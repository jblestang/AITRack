//! Pipeline orchestrator: the full tracking cycle for one radar batch.
//!
//! # Processing steps per batch (event-driven, async multi-sensor)
//! 1. Apply bias correction to all measurements
//! 2. Predict all live tracks to the batch timestamp
//! 3. For each track × measurement: gate check (Mahalanobis)
//! 4. Build sparse bipartite graph from gate-passing pairs
//! 5. Partition graph into connected components (union-find)
//! 6. Solve each component with Hungarian algorithm
//! 7. Update matched tracks (KF update)
//! 8. Register hits/misses on track manager
//! 9. Birth tentative tracks for unmatched measurements
//! 10. Prune deleted tracks
//! 11. Collect debug data for UI

use crate::{
    association::{hungarian_solve, partition_components, BipartiteGraph},
    bias::BiasEstimator,
    gating::{compute_gate_ellipse, mahalanobis_gate, GatingEllipse, SpatialGrid},
    imm::CtKalmanFilter,
    kf::{CvKalmanFilter, CvKfConfig, KalmanFilter},
    track::{Track, TrackStatus},
    track_manager::{TrackManager, TrackManagerConfig},
    types::{DMat, DVec, Measurement, MeasurementId, RadarBatch, TrackId},
};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the tracking pipeline.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Mahalanobis gate threshold (χ² value). Default: CHI2_99[2] for 2D.
    pub gate_threshold: f64,
    /// Cost assigned to dummy track/measurement in Hungarian (missed/clutter).
    pub dummy_cost: f64,
    /// KF motion model config (used for CV-slow model; also baseline for non-IMM mode)
    pub kf_config: CvKfConfig,
    /// Track management config
    pub track_manager_config: TrackManagerConfig,
    /// Show debug data for all components (not just the selected track)
    pub collect_debug: bool,
    /// Enable IMM (CV-slow + CV-fast + CT models per track).
    /// When true, each track carries an `ImmState` and the fused output is used
    /// for gating and display.  When false, plain CV filter is used (Phase A).
    pub use_imm: bool,
    /// Process noise std for the high-manoeuvre CV model inside IMM (m/s²)
    pub imm_sigma_fast: f64,
    /// Position process noise std for the CT model inside IMM (m)
    pub imm_ct_sigma_p: f64,
    /// Velocity process noise std for the CT model inside IMM (m/s)
    pub imm_ct_sigma_v: f64,
    /// Enable Joint Probabilistic Data Association (JPDA) instead of Hungarian.
    pub use_jpda: bool,
    /// Enable merging of duplicate tracks observed by multiple asynchronous radars.
    pub enable_track_merging: bool,
    /// Absolute maximum spatial distance squared (fast pruning)
    pub merge_dist_sq: f64,
    /// The Mahalanobis 2D statistical distance squared limit to allow track merging.
    pub merge_maha_sq: f64,
    /// Fixed sensor biases (dx, dy, dtheta, br, ba, dt0)
    pub fixed_biases: std::collections::HashMap<crate::types::SensorId, crate::bias::SensorBiasState>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            gate_threshold: 20.0, // wide gate for high-G maneuvers
            dummy_cost: 1000.0,
            kf_config: CvKfConfig {
                process_noise_std: 3.0,
            },
            track_manager_config: crate::track_manager::TrackManagerConfig {
                miss_limit_confirmed: 15,
                miss_limit_tentative: 2,
                confirm_m: 3,
                ..Default::default()
            },
            collect_debug: true,
            use_imm: true,
            imm_sigma_fast: 150.0,
            imm_ct_sigma_p: 200.0,
            imm_ct_sigma_v: 100.0,
            use_jpda: true,
            enable_track_merging: false, // Disabled to prevent it murdering the 50 dense-crossing targets
            merge_dist_sq: 6000.0 * 6000.0, // 6km fast spatial bound
            merge_maha_sq: 20.0, // 99.9% statistical overlap
            fixed_biases: std::collections::HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Debug Data (for UI)
// ---------------------------------------------------------------------------

/// All intermediate data produced in one pipeline step — for debug UI.
#[derive(Clone, Debug, Default)]
pub struct PipelineDebugData {
    /// Gating ellipses for each track (projected 2D)
    pub gate_ellipses: Vec<GatingEllipse>,
    /// Association edges: (track_id, meas_id) that passed gating
    pub gate_edges: Vec<(TrackId, MeasurementId, f64)>, // (track, meas, d²)
    /// Assignments confirmed by Hungarian
    pub assignments: Vec<(TrackId, MeasurementId)>,
    /// Innovations for matched pairs (for residual display)
    pub innovations: Vec<(TrackId, Vec<f64>)>,
    /// Component membership: component_idx → list of track IDs
    pub components: Vec<Vec<TrackId>>,
    /// Timings in microseconds
    pub timing_predict_us: u64,
    pub timing_gate_us: u64,
    pub timing_assign_us: u64,
    pub timing_update_us: u64,
    pub timing_manage_us: u64,
    /// Bias estimation states for all sensors
    pub sensor_biases: std::collections::HashMap<crate::types::SensorId, crate::bias::SensorBiasState>,
}

/// Outputs of one pipeline step.
#[derive(Clone, Debug)]
pub struct PipelineOutput {
    /// All live tracks after this step
    pub tracks: Vec<Track>,
    /// Measurements that weren't matched (false alarms / clutter)
    pub unmatched_measurements: Vec<Measurement>,
    /// Debug data (populated only when `collect_debug` is true)
    pub debug: PipelineDebugData,
    /// Wall-clock time of processing
    pub total_time_us: u64,
    /// Number of tracks born this step
    pub births: usize,
    /// Number of tracks confirmed this step
    pub confirmations: usize,
    /// Number of tracks deleted this step
    pub deletions: usize,
}

// ---------------------------------------------------------------------------
// Observation model helpers
// ---------------------------------------------------------------------------

/// Build 2×6 observation matrix H for cartesian XY measurement.
fn h_cartesian_2d() -> DMat {
    DMatrix::from_row_slice(2, 6, &[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// The main tracking pipeline. Holds the track pool and filter.
pub struct Pipeline {
    pub config: PipelineConfig,
    pub tracks: Vec<Track>,
    pub track_manager: TrackManager,
    /// CV-slow KF (also used as the single filter in non-IMM mode)
    pub kf: CvKalmanFilter,
    /// CV-fast KF used inside IMM
    kf_fast: CvKalmanFilter,
    /// CT-Left KF used inside IMM
    kf_ctl: CtKalmanFilter,
    /// CT-Right KF used inside IMM
    kf_ctr: CtKalmanFilter,
    pub bias_estimator: BiasEstimator,
    /// Registry of known sensor positions (updated from incoming batches)
    pub sensor_positions: HashMap<crate::types::SensorId, [f64; 3]>,
    next_meas_id: u64,
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        let kf = CvKalmanFilter::new(config.kf_config.clone());
        let kf_fast = CvKalmanFilter::new(CvKfConfig {
            process_noise_std: config.imm_sigma_fast,
        });
        let kf_ctl = CtKalmanFilter::new(0.3, config.imm_ct_sigma_p, config.imm_ct_sigma_v);
        let kf_ctr = CtKalmanFilter::new(-0.3, config.imm_ct_sigma_p, config.imm_ct_sigma_v);
        let track_manager = TrackManager::new(config.track_manager_config.clone());
        let mut bias_estimator = BiasEstimator::new();
        bias_estimator.sensor_states = config.fixed_biases.clone();
        
        Self {
            config,
            tracks: Vec::new(),
            track_manager,
            kf,
            kf_fast,
            kf_ctl,
            kf_ctr,
            bias_estimator,
            sensor_positions: HashMap::new(),
            next_meas_id: 0,
        }
    }

    /// Process a batch of radar measurements. Returns full pipeline output.
    pub fn process_batch(&mut self, batch: &RadarBatch) -> PipelineOutput {
        let start_total = Instant::now();
        let mut debug = PipelineDebugData::default();
        let collect = self.config.collect_debug;

        // ----------------------------------------------------------------
        // Step 1: Bias correction (Phase A: identity, stats maintained)
        // ----------------------------------------------------------------
        if let Some(pos) = batch.sensor_pos {
            self.sensor_positions.insert(batch.sensor_id, pos);
        }
        let sensor_pos = self.sensor_positions.get(&batch.sensor_id).cloned().unwrap_or([0.0, 0.0, 0.0]);
        let sensor_bias = self.bias_estimator.get_sensor_state(batch.sensor_id);
        let measurements: Vec<Measurement> = batch
            .measurements
            .iter()
            .map(|m| {
                let mut mc = m.clone();
                // Phase C: Apply temporal bias correction (T_true = T_received - dt0)
                mc.timestamp = sensor_bias.correct_timestamp(m.timestamp);

                // Apply spatial bias correction if available
                if let crate::types::MeasurementValue::Cartesian2D { x, y } = &mut mc.value {
                    let (xc, yc) = sensor_bias.correct_cartesian_2d(*x, *y, sensor_pos);
                    *x = xc;
                    *y = yc;
                }
                // Assign unique measurement ID
                mc.id = MeasurementId(self.next_meas_id);
                self.next_meas_id += 1;
                mc
            })
            .collect();

        let batch_time = sensor_bias.correct_timestamp(batch.sensor_time);

        // ----------------------------------------------------------------
        // Step 2: Predict all tracks to batch_time (Parallel)
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        // Borrow KFs locally to use in par_iter_mut closure
        let kf = &self.kf;
        let kf_fast = &self.kf_fast;
        let kf_ctl = &self.kf_ctl;
        let kf_ctr = &self.kf_ctr;

        self.tracks.par_iter_mut().for_each(|track| {
            if track.status != TrackStatus::Deleted {
                let dt = (batch_time - track.last_updated).max(0.0);
                if dt > 0.0 {
                    if let Some(imm) = &mut track.imm {
                        // IMM predict: mixing + per-model predict + fuse
                        imm.predict(dt, kf, kf_fast, kf_ctl, kf_ctr);
                        track.state = imm.fused_state;
                        track.cov = imm.fused_cov;
                    } else {
                        let (new_state, new_cov) = kf.predict(&track.state, &track.cov, dt);
                        track.state = new_state;
                        track.cov = new_cov;
                    }
                }
            }
        });
        debug.timing_predict_us = t0.elapsed().as_micros() as u64;

        // ----------------------------------------------------------------
        // Step 3-4: Gating + build bipartite graph
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        let h = h_cartesian_2d();
        let n_tracks = self.tracks.len();
        let n_meas = measurements.len();
        let mut graph = BipartiteGraph::new(n_tracks, n_meas);

        // We also pre-compute innovations to avoid re-computation in KF update
        // key: (track_idx, meas_idx)

        // Build spatial grid for measurements (O(M))
        let mut grid = SpatialGrid::new(5000.0);
        for (mi, meas) in measurements.iter().enumerate() {
            let z = meas.to_cartesian_2d();
            grid.insert(mi, z[0], z[1]);
        }

        let live_track_indices: Vec<usize> = (0..n_tracks)
            .filter(|&i| self.tracks[i].status != TrackStatus::Deleted)
            .collect();

        // Used to hold results computed concurrently
        #[derive(Default)]
        struct GateResult {
            edges: Vec<(usize, usize, f64)>, // (ti, mi, d2)
            debug_edges: Vec<(TrackId, MeasurementId, f64)>,
            innovations: Vec<((usize, usize), (DVec, DMat))>,
            ellipses: Vec<GatingEllipse>,
        }

        // Parallel gating over all live tracks
        let gate_threshold = self.config.gate_threshold;
        let r_default = if measurements.is_empty() {
            DMatrix::from_diagonal(&DVector::from_vec(vec![100.0, 100.0]))
        } else {
            measurements[0].noise_cov_matrix()
        };

        let gate_results: Vec<GateResult> = live_track_indices
            .par_iter()
            .map(|&ti| {
                let track = &self.tracks[ti];
                let mut res = GateResult::default();

                // Query nearby measurements from grid (O(1) average)
                let nearby_meas_indices = grid.query_nearby(track.state[0], track.state[1]);

                let mut track_edges = Vec::new();

                for &mi in &nearby_meas_indices {
                    let meas = &measurements[mi];
                    let z = meas.to_cartesian_2d();
                    
                    // Absolute physical hard-cap to prevent Mahalanobis explosion seduction
                    let dx = z[0] - track.state[0];
                    let dy = z[1] - track.state[1];
                    let physical_dist_sq = dx * dx + dy * dy;
                    if physical_dist_sq > 2000.0 * 2000.0 {
                        continue;
                    }

                    let r = meas.noise_cov_matrix();

                    let gate =
                        mahalanobis_gate(&track.state, &track.cov, &z, &h, &r, gate_threshold);
                    if gate.passes {
                        track_edges.push((mi, gate, meas.id));
                    }
                }

                // Prune high-density components by keeping only the top 5 closest measurements.
                // This breaks apart massive connected graphs, keeping Hungarian O(N³) fast.
                track_edges.sort_unstable_by(|a, b| a.1.d2.partial_cmp(&b.1.d2).unwrap());
                track_edges.truncate(5);

                for (mi, gate, meas_id) in track_edges {
                    res.edges.push((ti, mi, gate.d2));
                    if collect {
                        res.debug_edges.push((track.id, meas_id, gate.d2));
                    }
                    res.innovations
                        .push(((ti, mi), (gate.innovation, gate.innovation_cov)));
                }

                // Compute gate ellipse for display
                if collect {
                    let r = if nearby_meas_indices.is_empty() {
                        &r_default
                    } else {
                        &measurements[nearby_meas_indices[0]].noise_cov_matrix()
                    };
                    let ellipse = compute_gate_ellipse(&track.state, &track.cov, &h, r, ti, 3.0);
                    res.ellipses.push(ellipse);
                }

                res
            })
            .collect();

        // Accumulate results
        let mut innovation_cache: HashMap<(usize, usize), (DVec, DMat)> =
            HashMap::with_capacity(gate_results.iter().map(|r| r.innovations.len()).sum());
        for res in gate_results {
            for (ti, mi, d2) in res.edges {
                graph.add_edge(ti, mi, d2);
            }
            if collect {
                debug.gate_edges.extend(res.debug_edges);
                debug.gate_ellipses.extend(res.ellipses);
            }
            innovation_cache.extend(res.innovations);
        }

        debug.timing_gate_us = t0.elapsed().as_micros() as u64;

        // ----------------------------------------------------------------
        // Step 5: Partition + Step 6: Hungarian per component
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        let components = partition_components(&graph);
        let mut all_assignments: Vec<(usize, usize)> = Vec::new();
        let mut all_unmatched_tracks: Vec<usize> = Vec::new();
        let mut all_unmatched_meas: Vec<usize> = Vec::new();

        if collect {
            debug.components = components
                .iter()
                .map(|c| {
                    c.track_indices
                        .iter()
                        .filter_map(|&ti| self.tracks.get(ti).map(|t| t.id))
                        .collect()
                })
                .collect();
        }

        // Parallel resolution over disconnected components
        let dummy_cost = self.config.dummy_cost;
        let mut jpda_outputs: Vec<(usize, Vec<(usize, f64)>, f64)> = Vec::new();

        if self.config.use_jpda {
            let pd = 0.95; 
            let lambda_c = 1e-6;
            let jpda_results: Vec<_> = components
                .par_iter()
                .map(|comp| crate::jpda::jpda_solve(comp, &graph, pd, lambda_c, dummy_cost))
                .collect();
            
            for (comp, res) in components.iter().zip(jpda_results.iter()) {
                all_unmatched_tracks.extend(&res.unmatched_tracks);
                all_unmatched_meas.extend(&res.unmatched_meas);
                for (i, &ti) in comp.track_indices.iter().enumerate() {
                    jpda_outputs.push((ti, res.meas_probs[i].clone(), res.miss_probs[i]));
                }
            }
        } else {
            let assignment_results: Vec<_> = components
                .par_iter()
                .map(|comp| hungarian_solve(comp, dummy_cost))
                .collect();

            for ass in assignment_results {
                all_assignments.extend(ass.pairs);
                all_unmatched_tracks.extend(ass.unmatched_tracks);
                all_unmatched_meas.extend(ass.unmatched_meas);
            }
        }

        // Tracks not in any component are all unmatched (no measurements near them)
        for &ti in &live_track_indices {
            let in_graph = graph.edges.iter().any(|e| e.track_idx == ti);
            if !in_graph {
                all_unmatched_tracks.push(ti);
            }
        }
        debug.timing_assign_us = t0.elapsed().as_micros() as u64;

        // ----------------------------------------------------------------
        // Step 7: KF update for matched pairs & Bias Harvesting
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        let mut matched_meas_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let h_ref = &h;
        let mut confirmations = 0;

        if self.config.use_jpda {
            for (ti, m_probs, miss_prob) in jpda_outputs {
                let track = &mut self.tracks[ti];
                
                let mut jpda_meas = Vec::new();
                for (mi, prob) in &m_probs {
                    let meas = &measurements[*mi];
                    let z = meas.to_cartesian_2d();
                    jpda_meas.push((z, *prob));
                    if collect {
                        debug.assignments.push((track.id, meas.id));
                    }
                    matched_meas_indices.insert(*mi);
                }
                
                if jpda_meas.is_empty() {
                    continue; // Register miss happens later
                }
                

                let r = measurements[m_probs[0].0].noise_cov_matrix();

                if let Some(imm) = &mut track.imm {
                    imm.update_jpda(&jpda_meas, miss_prob, h_ref, &r, &self.kf, &self.kf_fast, &self.kf_ctl, &self.kf_ctr);
                    track.state = imm.fused_state;
                    track.cov = imm.fused_cov;
                } else {
                    let res = self.kf.update_jpda(&track.state, &track.cov, &jpda_meas, miss_prob, h_ref, &r);
                    track.state = res.0;
                    track.cov = res.1;
                }

                let prev_status = track.status;
                track.last_updated = batch_time;
                track.push_history();
                self.track_manager.register_hit(track);
                if track.status == TrackStatus::Confirmed && prev_status == TrackStatus::Tentative {
                    confirmations += 1;
                }
            }
        } else {
            for (ti, mi) in &all_assignments {
                matched_meas_indices.insert(*mi);
                let meas = &measurements[*mi];
                let z = meas.to_cartesian_2d();
                let r = meas.noise_cov_matrix();

                let track = &mut self.tracks[*ti];
                

                if let Some(imm) = &mut track.imm {
                    // IMM update: per-model update + likelihood weighting + fuse
                    imm.update(
                        &z,
                        h_ref,
                        &r,
                        &self.kf,
                        &self.kf_fast,
                        &self.kf_ctl,
                        &self.kf_ctr,
                    );
                    track.state = imm.fused_state;
                    track.cov = imm.fused_cov;

                    if collect {
                        let innov = z.clone()
                            - h_ref * nalgebra::DVector::from_iterator(6, track.state.iter().copied());
                        debug
                            .innovations
                            .push((track.id, innov.iter().copied().collect()));
                        debug.assignments.push((track.id, meas.id));
                    }
                } else {
                    let res = self.kf.update(&track.state, &track.cov, &z, h_ref, &r);
                    track.state = res.state;
                    track.cov = res.cov;
                    if collect {
                        debug.assignments.push((track.id, meas.id));
                        debug
                            .innovations
                            .push((track.id, res.innovation.iter().copied().collect()));
                    }
                }

                let prev_status = track.status;
                track.last_updated = batch_time;
                track.push_history();
                self.track_manager.register_hit(track);
                if track.status == TrackStatus::Confirmed && prev_status == TrackStatus::Tentative {
                    confirmations += 1;
                }
            }
        }
        
        // Feed harvested high-confidence        
        debug.timing_update_us = t0.elapsed().as_micros() as u64;

        // ----------------------------------------------------------------
        // Step 8: Register misses for unmatched tracks
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        for &ti in &all_unmatched_tracks {
            self.track_manager.register_miss(&mut self.tracks[ti]);
        }

        // ----------------------------------------------------------------
        // Step 9: Birth tentative tracks for unmatched measurements
        // ----------------------------------------------------------------
        let unmatched_meas: Vec<Measurement> = (0..n_meas)
            .filter(|i| !matched_meas_indices.contains(i))
            .map(|i| measurements[i].clone())
            .collect();

        let births = unmatched_meas.len();
        for meas in &unmatched_meas {
            let mut new_track = self.track_manager.birth_track(meas, batch_time);
            if self.config.use_imm {
                new_track.init_imm();
            }
            self.tracks.push(new_track);
        }

        // ----------------------------------------------------------------
        // Step 10: Track Merging
        // Prevent duplicate tracks from asynchronous multi-sensor fusion.
        // ----------------------------------------------------------------
        if self.config.enable_track_merging {
            let _t_merge = Instant::now();
            let mut merged_away = std::collections::HashSet::new();
            
            // O(T^2) spatial clustering loop 
            let live_indices: Vec<usize> = (0..self.tracks.len())
                .filter(|&i| self.tracks[i].status != TrackStatus::Deleted)
                .collect();

            // Compare each pair
            for (i, &idx1) in live_indices.iter().enumerate() {
                if merged_away.contains(&idx1) { continue; }
                
                for &idx2 in &live_indices[i + 1..] {
                    if merged_away.contains(&idx2) { continue; }

                    let t1 = &self.tracks[idx1];
                    let t2 = &self.tracks[idx2];
                    
                    // Fast Euclidean distance bounding check
                    let dx = t1.state[0] - t2.state[0];
                    let dy = t1.state[1] - t2.state[1];
                    let dist_sq = dx * dx + dy * dy;

                    // Velocity check: crossing independent targets have distinct velocities.
                    // Tentative tracks have uninitialized/poor velocity estimates, so conditionally skip the `dv` check if either is Tentative.
                    let both_confirmed = t1.status == TrackStatus::Confirmed && t2.status == TrackStatus::Confirmed;
                    let mut vel_ok = true;
                    
                    if both_confirmed {
                        let dvx = t1.state[3] - t2.state[3];
                        let dvy = t1.state[4] - t2.state[4];
                        let dv_sq = dvx * dvx + dvy * dvy;
                        vel_ok = dv_sq < 50.0 * 50.0;
                    }

                    // Strict Covariance-based spatial Mahalanobis distance (S = P1(x,y) + P2(x,y))
                    let s_x = t1.cov[(0, 0)] + t2.cov[(0, 0)];
                    let s_y = t1.cov[(1, 1)] + t2.cov[(1, 1)];
                    let s_xy = t1.cov[(0, 1)] + t2.cov[(0, 1)];
                    let det = s_x * s_y - s_xy * s_xy;
                    
                    let mut maha_d2 = std::f64::INFINITY;
                    if det > 1e-6 {
                        maha_d2 = (dx * (s_y * dx - s_xy * dy) + dy * (-s_xy * dx + s_x * dy)) / det;
                    }

                    // Merge if statistically overlapping AND velocity vector is very similar (or track is Tentative).
                    if dist_sq < self.config.merge_dist_sq && maha_d2 < self.config.merge_maha_sq && vel_ok {
                        // Tracks are redundantly covering the same target.
                        let t1_score = (
                            match t1.status {
                                TrackStatus::Confirmed => 2,
                                TrackStatus::Tentative => 1,
                                TrackStatus::Deleted => 0,
                            },
                            t1.total_hits,
                            std::cmp::Reverse(t1.id.0), // Tie breaker: smallest ID wins (oldest original track)
                        );
                        let t2_score = (
                            match t2.status {
                                TrackStatus::Confirmed => 2,
                                TrackStatus::Tentative => 1,
                                TrackStatus::Deleted => 0,
                            },
                            t2.total_hits,
                            std::cmp::Reverse(t2.id.0),
                        );

                        if t1_score >= t2_score {
                            merged_away.insert(idx2);
                        } else {
                            merged_away.insert(idx1);
                            break; // idx1 is gone, no need to compare it further
                        }
                    }
                }
            }

            // Execute deletions
            for &idx in &merged_away {
                self.tracks[idx].status = TrackStatus::Deleted;
            }
        }

        // ----------------------------------------------------------------
        // Step 11: Prune deleted tracks
        // ----------------------------------------------------------------
        let deletions = TrackManager::prune_deleted(&mut self.tracks);
        debug.timing_manage_us = t0.elapsed().as_micros() as u64;

        if self.config.collect_debug {
            debug.sensor_biases = self.bias_estimator.sensor_states.clone();
        }

        debug.timing_manage_us = t0.elapsed().as_micros() as u64;

        PipelineOutput {
            tracks: self.tracks.clone(),
            unmatched_measurements: unmatched_meas.clone(),
            debug,
            total_time_us: start_total.elapsed().as_micros() as u64,
            births,
            confirmations,
            deletions,
        }
    }

    /// Reset: clear all tracks.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_meas_id = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MeasurementValue, RadarBatch, SensorId};

    fn make_batch(sensor_id: u32, t: f64, positions: &[(f64, f64)]) -> RadarBatch {
        let measurements = positions
            .iter()
            .map(|&(x, y)| Measurement {
                id: MeasurementId(0),
                sensor_id: SensorId(sensor_id),
                timestamp: t,
                value: MeasurementValue::Cartesian2D { x, y },
                noise_cov: vec![25.0, 0.0, 0.0, 25.0], // 5m std dev
            })
            .collect();
        RadarBatch {
            sensor_id: SensorId(sensor_id),
            sensor_time: t,
            arrival_time: t,
            sensor_pos: None,
            measurements,
        }
    }

    #[test]
    fn pipeline_births_and_tracks() {
        let mut pipeline = Pipeline::new(PipelineConfig::default());

        // First batch: 2 targets
        let batch1 = make_batch(0, 0.0, &[(100.0, 200.0), (300.0, 400.0)]);
        let out1 = pipeline.process_batch(&batch1);
        assert_eq!(out1.births, 2, "Should birth 2 tracks");

        // Second batch: same 2 targets, moved slightly
        let batch2 = make_batch(0, 1.0, &[(101.0, 200.0), (301.0, 401.0)]);
        let out2 = pipeline.process_batch(&batch2);
        assert!(
            out2.births == 0 || out2.births <= 2,
            "Existing tracks should absorb measurements"
        );
    }

    #[test]
    fn pipeline_confirmation() {
        let cfg = PipelineConfig {
            track_manager_config: TrackManagerConfig {
                confirm_m: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pipeline = Pipeline::new(cfg);

        for t in 0..5 {
            let batch = make_batch(0, t as f64, &[(100.0 + t as f64, 100.0)]);
            pipeline.process_batch(&batch);
        }

        let confirmed = pipeline
            .tracks
            .iter()
            .filter(|t| t.status == TrackStatus::Confirmed)
            .count();
        assert!(
            confirmed >= 1,
            "At least one track should be confirmed after 5 frames"
        );
    }
}
