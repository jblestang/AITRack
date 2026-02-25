//! Joint Probabilistic Data Association (JPDA).
//!
//! JPDA avoids making hard decisions (like Hungarian GNN) by enumerating all plausible
//! joint events (hypothesis) in a connected component, calculating their probabilities,
//! and computing the marginal association probability β_{tj} for each track-measurement pair.

use crate::association::{BipartiteGraph, Component};

/// Result of JPDA marginalization.
pub struct JpdaResult {
    /// Marginal probability β_{tj} for track `ti` associating to measurement `mj`.
    /// The index within the vector matches the input `comp.track_indices`.
    /// `meas_probs[i][j]` = P(track `i` associates to `mj`), where `mj` is the
    /// globally mapped measurement index in `comp.meas_indices`.
    pub meas_probs: Vec<Vec<(usize, f64)>>,
    /// Marginal probability β_{i0} that track `i` missed detection.
    pub miss_probs: Vec<f64>,
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_meas: Vec<usize>,
}

#[derive(Clone)]
struct Event {
    /// map from local track index to local measurement index (or None for miss)
    assignments: Vec<Option<usize>>,
    prob: f64,
}

/// Solves JPDA for a single connected component.
///
/// `comp`: the spatial component.
/// `graph`: the full bipartite graph (for accessing edge costs).
/// `pd`: probability of detection (e.g. 0.95).
/// `lambda_clutter`: spatial density of false alarms.
///
/// Note: The `graph.edges` stores `d2` (Mahalanobis distance squared).
/// The likelihood of a valid gate association is proportional to exp(-d2 / 2).
pub fn jpda_solve(
    comp: &Component,
    graph: &BipartiteGraph,
    pd: f64,
    lambda_clutter: f64,
    dummy_cost: f64,
) -> JpdaResult {
    let n_tracks = comp.track_indices.len();
    let n_meas = comp.meas_indices.len();

    // Safety fallback: exact JPDA scales factorially O(N!).
    // For large components, the approximation truncation biases towards track deletion.
    // Instead, definitively associate large clusters using robust O(N^3) assignment.
    if n_tracks > 10 || n_meas > 10 {
        let ass_result = crate::association::hungarian_solve(comp, dummy_cost);
        let mut meas_probs = vec![Vec::new(); n_tracks];
        let mut miss_probs = vec![1.0; n_tracks];

        for &(ti, mi) in &ass_result.pairs {
            if let Some(local_ti) = comp.track_indices.iter().position(|&x| x == ti) {
                meas_probs[local_ti].push((mi, 1.0));
                miss_probs[local_ti] = 0.0;
            }
        }

        return JpdaResult {
            meas_probs,
            miss_probs,
            unmatched_tracks: ass_result.unmatched_tracks,
            unmatched_meas: ass_result.unmatched_meas,
        };
    }

    // 1. Build adjacency list for fast event generation
    // local_adj[i] = list of local measurement indices `j` that track `i` gates with
    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); n_tracks];
    // local_cost[i][j] = Mahalanobis distance squared d^2
    let mut local_cost: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_tracks];

    // Build reverse maps
    use std::collections::HashMap;
    let mut global_to_local_meas: HashMap<usize, usize> = HashMap::new();
    for (lj, &gj) in comp.meas_indices.iter().enumerate() {
        global_to_local_meas.insert(gj, lj);
    }
    let mut global_to_local_track: HashMap<usize, usize> = HashMap::new();
    for (li, &gi) in comp.track_indices.iter().enumerate() {
        global_to_local_track.insert(gi, li);
    }

    // Extract edges for this component
    for edge in &graph.edges {
        if let Some(&li) = global_to_local_track.get(&edge.track_idx) {
            if let Some(&lj) = global_to_local_meas.get(&edge.meas_idx) {
                local_adj[li].push(lj);
                local_cost[li].insert(lj, edge.cost);
            }
        }
    }

    // 2. Enumerate all joint events recursively.
    // An event is a set of assignments such that no measurement is assigned to more than one track.
    let mut valid_events = Vec::new();

    // Backtracking state
    let mut current_assignment = vec![None; n_tracks];
    let mut meas_used = vec![false; n_meas];

    // Safety limit to prevent computationally explosive dense components.
    // If a component becomes massively tangled, fallback to subset or truncate.
    // We limit JPDA enumeration to computationally feasible bounds.
    let max_events = 10_000;

    fn generate_events(
        t_idx: usize,
        n_tracks: usize,
        local_adj: &[Vec<usize>],
        meas_used: &mut [bool],
        current_assignment: &mut Vec<Option<usize>>,
        valid_events: &mut Vec<Event>,
        max_events: usize,
    ) {
        if valid_events.len() >= max_events {
            return;
        }

        if t_idx == n_tracks {
            valid_events.push(Event {
                assignments: current_assignment.clone(),
                prob: 0.0, // Calculated later
            });
            return;
        }

        // Option A: Track t_idx is missed (assigned to None)
        current_assignment[t_idx] = None;
        generate_events(
            t_idx + 1,
            n_tracks,
            local_adj,
            meas_used,
            current_assignment,
            valid_events,
            max_events,
        );

        if valid_events.len() >= max_events {
            return;
        }

        // Option B: Track t_idx is assigned to a valid gated measurement
        for &m_idx in &local_adj[t_idx] {
            if !meas_used[m_idx] {
                meas_used[m_idx] = true;
                current_assignment[t_idx] = Some(m_idx);

                generate_events(
                    t_idx + 1,
                    n_tracks,
                    local_adj,
                    meas_used,
                    current_assignment,
                    valid_events,
                    max_events,
                );

                meas_used[m_idx] = false;

                if valid_events.len() >= max_events {
                    return;
                }
            }
        }
    }
    
    // Start recursion
    generate_events(
        0,
        n_tracks,
        &local_adj,
        &mut meas_used,
        &mut current_assignment,
        &mut valid_events,
        max_events,
    );

    // 3. Calculate probabilies of each event
    // The probability of an event E is proportional to:
    // P(E) ∝ ∏_{i detected} ( \frac{P_D}{V_c} * L_{i,j} ) * ∏_{i missed} (1 - P_D) * (λ_{clutter})^{N_fa}
    // Alternatively (simplifying constants across all valid events):
    // P(E) ∝ ∏_{i detected}  \frac{P_D}{λ_{clutter}} L_{i,j} * ∏_{i missed} (1 - P_D)
    // where L_{i,j} = \frac{1}{\sqrt{|2\pi S|}} exp(-d^2 / 2)
    // To avoid tracking S determinant explicitly here (since it's roughly constant per sensor),
    // we use a simplified volumetric likelihood proportional to exp(-d2 / 2).
    //
    // L_{i,j} ~ (2\pi)^{-M/2} |S|^{-1/2} exp(-0.5 d2)
    // For simplicity, we approximate |S| as a constant tuning param c_s, giving a fixed 
    // scaling factor for hits.
    
    let mut total_prob = 0.0;
    
    // Lambda is typically very small. To avoid division by zero or infinites, we use log-sum-exp 
    // or just direct scale, assuming lambda > 0.
    let lambda = lambda_clutter.max(1e-9);
    
    // Empirical volume normalizer (derived from average gate size, e.g. 2D radar ~100m^2).
    // The exact scaling affects how aggressively the filter prefers misses vs low-probability hits.
    let volume_norm = 1.0 / (2.0 * std::f64::consts::PI * 1000.0);

    for event in &mut valid_events {
        let mut p = 1.0;
        for (i, opt_j) in event.assignments.iter().enumerate() {
            if let Some(j) = opt_j {
                let d2 = local_cost[i][j];
                let likelihood = volume_norm * (-0.5 * d2).exp();
                p *= (pd / lambda) * likelihood;
            } else {
                p *= 1.0 - pd;
            }
        }
        event.prob = p;
        total_prob += p;
    }

    // 4. Marginalize associations
    let mut marginals = vec![vec![0.0; n_meas]; n_tracks];
    let mut miss_probs = vec![0.0; n_tracks];

    if total_prob > 1e-30 {
        for event in &valid_events {
            let normalized_p = event.prob / total_prob;
            for (i, opt_j) in event.assignments.iter().enumerate() {
                if let Some(j) = opt_j {
                    marginals[i][*j] += normalized_p;
                } else {
                    miss_probs[i] += normalized_p;
                }
            }
        }
    } else {
        // Fallback if all probabilities underflowed completely
        for i in 0..n_tracks {
            miss_probs[i] = 1.0;
        }
    }

    // 5. Gather unmatched (for birth / pruning)
    // In JPDA, a track is "unmatched" if its miss_prob is near 1.0.
    // A measurement is "unmatched" if no track claims it with high probability.
    let mut unmatched_tracks = Vec::new();
    let mut unmatched_meas = Vec::new();

    let mut meas_claimed_prob = vec![0.0; n_meas];
    for i in 0..n_tracks {
        if miss_probs[i] > 0.99 {
            unmatched_tracks.push(comp.track_indices[i]);
        }
        for j in 0..n_meas {
            meas_claimed_prob[j] += marginals[i][j];
        }
    }

    for (j, &prob) in meas_claimed_prob.iter().enumerate() {
        if prob < 0.1 {
            unmatched_meas.push(comp.meas_indices[j]);
        }
    }

    // Convert local marginals to global structure
    let mut meas_probs = vec![Vec::new(); n_tracks];
    for (i, track_marginals) in marginals.iter().enumerate() {
        for (j, &prob) in track_marginals.iter().enumerate() {
            if prob > 0.001 { // Filter out negligible associations
                meas_probs[i].push((comp.meas_indices[j], prob));
            }
        }
    }

    JpdaResult {
        meas_probs,
        miss_probs,
        unmatched_tracks,
        unmatched_meas,
    }
}
