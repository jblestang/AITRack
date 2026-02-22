//! Data association: bipartite graph construction, connected-component
//! partitioning (union-find), and Hungarian assignment.
//!
//! # Algorithm pipeline
//! 1. For each (track, measurement) pair that passed gating, add an edge
//!    to the sparse bipartite graph.
//! 2. Partition the graph into **connected components** using union-find.
//!    Components are independent — they can be solved in parallel.
//! 3. Solve each component with the **Hungarian algorithm** (Jonker-Volgenant
//!    style O(n³) implementation).
//!
//! Phase B will add kd-tree spatial lookup; Phase D will add JPDA on dense
//! components.

// use crate::types::MeasurementId;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Bipartite graph
// ---------------------------------------------------------------------------

/// An edge in the dense assignment cost matrix.
#[derive(Clone, Debug)]
pub struct AssignEdge {
    pub track_idx: usize,
    pub meas_idx: usize,
    /// Mahalanobis distance squared (used as cost)
    pub cost: f64,
}

/// Sparse bipartite graph: edges between track indices and measurement indices.
#[derive(Clone, Debug, Default)]
pub struct BipartiteGraph {
    pub edges: Vec<AssignEdge>,
    pub n_tracks: usize,
    pub n_meas: usize,
}

impl BipartiteGraph {
    pub fn new(n_tracks: usize, n_meas: usize) -> Self {
        Self {
            edges: Vec::new(),
            n_tracks,
            n_meas,
        }
    }

    /// Add an edge (gate-passed association candidate).
    pub fn add_edge(&mut self, track_idx: usize, meas_idx: usize, cost: f64) {
        self.edges.push(AssignEdge {
            track_idx,
            meas_idx,
            cost,
        });
    }

    /// True if no edges exist — all tracks/meas are independent.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Union-Find (path compression + union by rank)
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// A single connected component in the bipartite graph.
#[derive(Clone, Debug)]
pub struct Component {
    pub track_indices: Vec<usize>,
    pub meas_indices: Vec<usize>,
    pub edges: Vec<AssignEdge>,
}

/// Partition the bipartite graph into connected components.
/// Each component can be solved independently.
///
/// We treat tracks and measurements as nodes in a combined graph:
/// - Track i   → node i
/// - Measure j → node n_tracks + j
pub fn partition_components(graph: &BipartiteGraph) -> Vec<Component> {
    let n_total = graph.n_tracks + graph.n_meas;
    let mut uf = UnionFind::new(n_total);

    for e in &graph.edges {
        uf.union(e.track_idx, graph.n_tracks + e.meas_idx);
    }

    // Group edges by component root
    let mut comp_map: HashMap<usize, Component> = HashMap::new();
    for e in &graph.edges {
        let root = uf.find(e.track_idx);
        let comp = comp_map.entry(root).or_insert_with(|| Component {
            track_indices: Vec::new(),
            meas_indices: Vec::new(),
            edges: Vec::new(),
        });
        comp.edges.push(e.clone());
    }

    // Fill track and meas index lists (deduplicated)
    for comp in comp_map.values_mut() {
        comp.track_indices = comp.edges.iter().map(|e| e.track_idx).collect();
        comp.track_indices.sort_unstable();
        comp.track_indices.dedup();
        comp.meas_indices = comp.edges.iter().map(|e| e.meas_idx).collect();
        comp.meas_indices.sort_unstable();
        comp.meas_indices.dedup();
    }

    comp_map.into_values().collect()
}

// ---------------------------------------------------------------------------
// Hungarian algorithm — O(n³) Kuhn-Munkres
// ---------------------------------------------------------------------------

/// Assignment result: (track_idx, meas_idx) matched pairs.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    pub pairs: Vec<(usize, usize)>,
    /// Track indices that were NOT matched (missed detections)
    pub unmatched_tracks: Vec<usize>,
    /// Measurement indices not matched (false alarms / clutter)
    pub unmatched_meas: Vec<usize>,
}

/// Solve the assignment problem for a single component using the Hungarian
/// algorithm on a rectangular cost matrix.
///
/// Dummy assignments (missed detection and false alarm) are handled by
/// adding a `dummy_cost` row at the bottom (for each extra measurement)
/// and a `dummy_cost` column on the right (for each extra track).
/// The algorithm naturally finds the globally optimal assignment.
pub fn hungarian_solve(component: &Component, dummy_cost: f64) -> Assignment {
    let nt = component.track_indices.len();
    let nm = component.meas_indices.len();

    if nt == 0 || nm == 0 {
        return Assignment {
            pairs: vec![],
            unmatched_tracks: component.track_indices.clone(),
            unmatched_meas: component.meas_indices.clone(),
        };
    }

    // Build local cost matrix (nt × nm) — fill with dummy_cost, then overwrite
    // with actual costs from graph edges.
    let n = nt.max(nm);
    let mut cost = vec![dummy_cost; n * n];

    // Local index maps: track_indices[i] → row i,  meas_indices[j] → col j
    let track_local: HashMap<usize, usize> = component
        .track_indices
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();
    let meas_local: HashMap<usize, usize> = component
        .meas_indices
        .iter()
        .enumerate()
        .map(|(j, &m)| (m, j))
        .collect();

    for e in &component.edges {
        if let (Some(&ri), Some(&ci)) = (track_local.get(&e.track_idx), meas_local.get(&e.meas_idx))
        {
            cost[ri * n + ci] = e.cost;
        }
    }

    // Run Hungarian on square n×n matrix
    let row_assign = run_hungarian(&cost, n);

    // Decode result
    let mut pairs = Vec::new();
    let mut unmatched_tracks = Vec::new();
    let mut matched_meas = vec![false; n];

    for (ri, ci) in row_assign.iter().enumerate() {
        let ci = *ci;
        if ri < nt && ci < nm {
            // Both real track and real measurement
            let track_global = component.track_indices[ri];
            let meas_global = component.meas_indices[ci];
            pairs.push((track_global, meas_global));
            matched_meas[ci] = true;
        } else if ri < nt {
            // Track assigned to dummy column → missed detection
            unmatched_tracks.push(component.track_indices[ri]);
        } else {
            // Dummy row → irrelevant
        }
    }

    let unmatched_meas: Vec<usize> = (0..nm)
        .filter(|&j| !matched_meas[j])
        .map(|j| component.meas_indices[j])
        .collect();

    Assignment {
        pairs,
        unmatched_tracks,
        unmatched_meas,
    }
}

/// Core Hungarian algorithm on a square n×n cost matrix (row-major).
/// Returns row_assignment[row] = assigned_column.
fn run_hungarian(cost: &[f64], n: usize) -> Vec<usize> {
    // Potentials for rows (u) and columns (v)
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    // p[j] = row assigned to column j (1-indexed, 0 = none)
    let mut p = vec![0usize; n + 1];
    // way[j] = previous column in augmenting path
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let val = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if val < minv[j] {
                        minv[j] = val;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Augment
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Decode: p[j] = row for column j (1-indexed)
    let mut row_assign = vec![0usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            row_assign[p[j] - 1] = j - 1;
        }
    }
    row_assign
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_3x3_known() {
        // Cost matrix:
        // [4, 1, 3]
        // [2, 0, 5]
        // [3, 2, 2]
        // Optimal: row0→col1 (1), row1→col0 (2), row2→col2 (2) = 5
        let cost = vec![4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let assign = run_hungarian(&cost, 3);
        let total: f64 = assign
            .iter()
            .enumerate()
            .map(|(r, &c)| cost[r * 3 + c])
            .sum();
        assert!(
            (total - 5.0).abs() < 1e-9,
            "Expected total cost 5, got {total}"
        );
    }

    #[test]
    fn partition_two_independent_components() {
        let mut graph = BipartiteGraph::new(4, 4);
        // Component 1: track 0 ↔ meas 0
        graph.add_edge(0, 0, 1.0);
        // Component 2: track 2 ↔ meas 3
        graph.add_edge(2, 3, 2.0);

        let comps = partition_components(&graph);
        assert_eq!(comps.len(), 2, "Should have 2 independent components");
    }

    #[test]
    fn hungarian_solve_simple() {
        let comp = Component {
            track_indices: vec![0, 1],
            meas_indices: vec![0, 1],
            edges: vec![
                AssignEdge {
                    track_idx: 0,
                    meas_idx: 0,
                    cost: 1.0,
                },
                AssignEdge {
                    track_idx: 0,
                    meas_idx: 1,
                    cost: 10.0,
                },
                AssignEdge {
                    track_idx: 1,
                    meas_idx: 0,
                    cost: 10.0,
                },
                AssignEdge {
                    track_idx: 1,
                    meas_idx: 1,
                    cost: 1.0,
                },
            ],
        };
        let ass = hungarian_solve(&comp, 100.0);
        assert_eq!(ass.pairs.len(), 2);
        // Both should be assigned to their close match
        assert!(ass.pairs.contains(&(0, 0)));
        assert!(ass.pairs.contains(&(1, 1)));
    }
}
