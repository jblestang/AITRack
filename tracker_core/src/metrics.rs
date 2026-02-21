//! Tracking metrics: RMSE position/velocity, ID-switch rate, precision/recall.

use crate::types::{StateVec, TrackId};
use serde::{Deserialize, Serialize};

/// Ground-truth state of one target at a given time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundTruth {
    /// True target ID (from simulator)
    pub target_id: u64,
    pub time: f64,
    pub state: [f64; 6],
}

/// Per-frame association between a track and a ground-truth target.
#[derive(Clone, Debug)]
pub struct TrackAssociation {
    pub track_id: TrackId,
    pub target_id: u64,
}

/// Accumulated metric statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrackingMetrics {
    /// Number of frames evaluated
    pub n_frames: u64,
    /// Total number of matched (track, target) pairs evaluated
    pub n_matched: u64,
    /// Sum of squared position errors (for RMSE)
    pub sum_sq_pos_err: f64,
    /// Sum of squared velocity errors (for RMSE)
    pub sum_sq_vel_err: f64,
    /// Total true positives (track matched to correct target)
    pub true_positives: u64,
    /// False positives (unmatched tracks)
    pub false_positives: u64,
    /// False negatives (unmatched targets)
    pub false_negatives: u64,
    /// ID switches detected
    pub id_switches: u64,
}

impl TrackingMetrics {
    /// Root-mean-square position error (meters, 2D).
    pub fn rmse_position(&self) -> f64 {
        if self.n_matched == 0 {
            return 0.0;
        }
        (self.sum_sq_pos_err / self.n_matched as f64).sqrt()
    }

    /// Root-mean-square velocity error (m/s, 2D).
    pub fn rmse_velocity(&self) -> f64 {
        if self.n_matched == 0 {
            return 0.0;
        }
        (self.sum_sq_vel_err / self.n_matched as f64).sqrt()
    }

    /// Precision = TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denom = (self.true_positives + self.false_positives) as f64;
        if denom == 0.0 { 1.0 } else { self.true_positives as f64 / denom }
    }

    /// Recall = TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denom = (self.true_positives + self.false_negatives) as f64;
        if denom == 0.0 { 1.0 } else { self.true_positives as f64 / denom }
    }

    /// Accumulate one frame's worth of associations.
    pub fn accumulate(
        &mut self,
        associations: &[TrackAssociation],
        track_states: &[(TrackId, StateVec)],
        ground_truths: &[GroundTruth],
    ) {
        self.n_frames += 1;

        // Build lookup: target_id -> GT state
        let gt_map: std::collections::HashMap<u64, &GroundTruth> =
            ground_truths.iter().map(|g| (g.target_id, g)).collect();

        let mut matched_targets = std::collections::HashSet::new();

        for assoc in associations {
            if let Some(gt) = gt_map.get(&assoc.target_id) {
                // Find track state
                if let Some((_, state)) = track_states.iter().find(|(id, _)| *id == assoc.track_id) {
                    let dx = state[0] - gt.state[0];
                    let dy = state[1] - gt.state[1];
                    let dvx = state[3] - gt.state[3];
                    let dvy = state[4] - gt.state[4];
                    self.sum_sq_pos_err += dx * dx + dy * dy;
                    self.sum_sq_vel_err += dvx * dvx + dvy * dvy;
                    self.n_matched += 1;
                    self.true_positives += 1;
                    matched_targets.insert(assoc.target_id);
                }
            } else {
                self.false_positives += 1;
            }
        }

        self.false_negatives += ground_truths
            .iter()
            .filter(|g| !matched_targets.contains(&g.target_id))
            .count() as u64;
    }
}
