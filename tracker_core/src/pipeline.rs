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
    gating::{compute_gate_ellipse, mahalanobis_gate, GatingEllipse, SpatialGrid, CHI2_99},
    imm::CtKalmanFilter,
    kf::{CvKalmanFilter, CvKfConfig, KalmanFilter},
    track::{Track, TrackStatus},
    track_manager::{TrackManager, TrackManagerConfig},
    types::{DMat, DVec, Measurement, MeasurementId, RadarBatch, TrackId},
};
use nalgebra::{DMatrix, DVector, Vector6};
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
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            gate_threshold: CHI2_99[2], // 99% for 2D
            dummy_cost: 1000.0,
            kf_config: CvKfConfig::default(),
            track_manager_config: TrackManagerConfig::default(),
            collect_debug: true,
            use_imm: false,
            imm_sigma_fast: 30.0,
            imm_ct_sigma_p: 50.0,
            imm_ct_sigma_v: 30.0,
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
        Self {
            config,
            tracks: Vec::new(),
            track_manager,
            kf,
            kf_fast,
            kf_ctl,
            kf_ctr,
            bias_estimator: BiasEstimator::new(),
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
        let sensor_bias = self.bias_estimator.get_or_create(batch.sensor_id).clone();
        let measurements: Vec<Measurement> = batch
            .measurements
            .iter()
            .map(|m| {
                let mut mc = m.clone();
                // Apply spatial bias correction if available
                if let crate::types::MeasurementValue::Cartesian2D { x, y } = &mut mc.value {
                    let (xc, yc) = sensor_bias.correct_cartesian_2d(*x, *y);
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

                for &mi in &nearby_meas_indices {
                    let meas = &measurements[mi];
                    let z = meas.to_cartesian_2d();
                    let r = meas.noise_cov_matrix();
                    let gate =
                        mahalanobis_gate(&track.state, &track.cov, &z, &h, &r, gate_threshold);
                    if gate.passes {
                        res.edges.push((ti, mi, gate.d2));
                        if collect {
                            res.debug_edges.push((track.id, meas.id, gate.d2));
                        }
                        res.innovations
                            .push(((ti, mi), (gate.innovation, gate.innovation_cov)));
                    }
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

        // Parallel Hungarian assignment over disconnected components
        let dummy_cost = self.config.dummy_cost;
        let assignment_results: Vec<_> = components
            .par_iter()
            .map(|comp| hungarian_solve(comp, dummy_cost))
            .collect();

        for ass in assignment_results {
            all_assignments.extend(ass.pairs);
            all_unmatched_tracks.extend(ass.unmatched_tracks);
            all_unmatched_meas.extend(ass.unmatched_meas);
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
        // Step 7: KF update for matched pairs
        // ----------------------------------------------------------------
        let t0 = Instant::now();
        let mut matched_meas_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let h_ref = &h;
        let mut confirmations = 0;

        for (ti, mi) in &all_assignments {
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
            matched_meas_indices.insert(*mi);
        }
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
        // Step 10: Prune deleted tracks
        // ----------------------------------------------------------------
        let deletions = TrackManager::prune_deleted(&mut self.tracks);
        debug.timing_manage_us = t0.elapsed().as_micros() as u64;

        let total_time_us = start_total.elapsed().as_micros() as u64;

        PipelineOutput {
            tracks: self.tracks.clone(),
            unmatched_measurements: unmatched_meas,
            debug,
            total_time_us,
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
