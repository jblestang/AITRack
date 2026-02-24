//! Bias estimation module (stub for Phase A, full implementation in Phase C).
//!
//! # Phase A
//! This module exposes the types but all estimators are no-ops.
//!
//! # Phase C (planned)
//! - **Spatial bias**: estimate (dx, dy, dtheta) per radar using innovation
//!   residuals from high-confidence associations (Mahalanobis d² < threshold).
//! - **Temporal bias**: estimate clock offset `dt0` per radar using dynamical
//!   coherence of innovations across time.
//! - **Outlier rejection**: ignore associations with d² > outlier threshold.

use crate::types::SensorId;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Spatial bias parameters for a single radar.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RadarSpatialBias {
    /// Position offset X (meters)
    pub dx: f64,
    /// Position offset Y (meters)
    pub dy: f64,
    /// Angular rotation offset (radians)
    pub dtheta: f64,
    /// Range bias (meters)
    pub br: f64,
    /// Azimuth bias (radians)
    pub ba: f64,
}

/// Temporal synchronization bias for a single radar.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RadarTimeBias {
    /// Clock offset (seconds): timestamp_true = timestamp_received - dt0
    pub dt0: f64,
    /// Clock drift (s/s): how fast the clock deviates per second
    pub dt_dot: f64,
}

/// Per-sensor bias state, updated by the bias estimator.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SensorBiasState {
    pub sensor_id: SensorId,
    pub spatial: RadarSpatialBias,
    pub temporal: RadarTimeBias,
    /// Confidence in the current estimate (0..1)
    pub confidence: f64,
}

impl SensorBiasState {
    pub fn new(sensor_id: SensorId) -> Self {
        Self {
            sensor_id,
            ..Default::default()
        }
    }

    /// Apply spatial bias correction to a cartesian 2D measurement.
    /// Returns (x_corrected, y_corrected).
    pub fn correct_cartesian_2d(&self, x: f64, y: f64) -> (f64, f64) {
        let cos_t = self.spatial.dtheta.cos();
        let sin_t = self.spatial.dtheta.sin();
        // Correct rotation then translation
        let xr = cos_t * x - sin_t * y - self.spatial.dx;
        let yr = sin_t * x + cos_t * y - self.spatial.dy;
        (xr, yr)
    }

    pub fn correct_timestamp(&self, t: f64) -> f64 {
        t - self.dt0_at(t)
    }

    fn dt0_at(&self, t: f64) -> f64 {
        self.temporal.dt0 + self.temporal.dt_dot * t
    }
    
    /// Returns a 2D bias vector [bx, by] evaluated at this timestamp.
    /// Used directly in the EKF measurement innovation step.
    pub fn get_spatial_bias_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.spatial.dx, self.spatial.dy])
    }
}

/// Bias estimator — holds per-sensor state and updates it.
/// Phase A: no-op. Phase C: incremental innovation-based estimation.
#[derive(Clone, Debug, Default)]
pub struct BiasEstimator {
    pub sensor_states: std::collections::HashMap<SensorId, SensorBiasState>,
}

impl BiasEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create bias state for a sensor.
    pub fn get_or_create(&mut self, sensor_id: SensorId) -> &mut SensorBiasState {
        self.sensor_states
            .entry(sensor_id)
            .or_insert_with(|| SensorBiasState::new(sensor_id))
    }

    /// Execute online estimation logic (Phase C).
    /// `innovations` holds `(track_state, innovation_vector)` for high-confidence associated pairs.
    pub fn update(
        &mut self,
        sensor_id: SensorId,
        high_confidence_pairs: &[(crate::types::StateVec, DVector<f64>)],
    ) {
        if high_confidence_pairs.is_empty() {
            return;
        }

        let state = self.get_or_create(sensor_id);
        
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;
        let mut sum_dt0 = 0.0;
        let mut temporal_votes = 0.0;

        for (track_state, innov) in high_confidence_pairs {
            // innovation ν = z_raw - H * x_pred
            // Under a translation bias (bx, by), E[z_raw] = H*x_true + B
            // Therefore, the raw innovation ν averages out to the bias B.
            
            // Phase D: Outlier rejection via magnitude absolute clamping
            // A mis-associated clutter point passing the Mahalanobis gate might have a physically
            // unreasonable innovation (e.g. 2000m). We clamp it to safely bound the integration.
            let rx = innov[0].clamp(-800.0, 800.0);
            let ry = innov[1].clamp(-800.0, 800.0);
            
            sum_dx += rx;
            sum_dy += ry;

            // Temporal estimation:
            // If the radar clock is delayed by dt0 (it reports events later than they happened),
            // the target has already moved by v * dt0 physically.
            // So the spatial error is proportional to velocity: e_pos = -v * dt0
            // dt0 = -e_pos / v
            
            let vx = track_state[3];
            let vy = track_state[4];
            let speed_sq = vx * vx + vy * vy;
            
            // Only update clocks from fast-moving targets (avoid divide-by-zero noise)
            if speed_sq > 25.0 {
                // Project the spatial error onto the velocity vector to isolate timing delays
                // from pure orthogonal spatial translation biases.
                // Outlier rejection: cap max dt0 estimation per frame to 1.5 seconds.
                let dt0_est = (-(rx * vx + ry * vy) / speed_sq).clamp(-1.5, 1.5);
                sum_dt0 += dt0_est;
                temporal_votes += 1.0;
            }
        }

        let n = high_confidence_pairs.len() as f64;
        let mean_dx = sum_dx / n;
        let mean_dy = sum_dy / n;

        // Exponential Moving Average (EMA) update rules
        // Since `innov` is the *residual* between the PRE-CORRECTED measurement and the prediction,
        // it represents the *remaining* uncompensated bias. 
        // We must cumulatively add this residual to the bias states, acting as an Integral controller.
        let alpha_spatial = 0.1; // Learning rate for spatial bias (increased for faster convergence)
        let alpha_temporal = 0.05; // Learning rate for clocks
        
        state.spatial.dx += alpha_spatial * mean_dx;
        state.spatial.dy += alpha_spatial * mean_dy;

        if temporal_votes > 0.0 {
            let mean_dt0 = sum_dt0 / temporal_votes;
            state.temporal.dt0 += alpha_temporal * mean_dt0;
        }
        
        // Asymptotically increase confidence towards 1.0 as more hits are processed
        state.confidence = 1.0 - (1.0 - state.confidence) * 0.99;
    }
}
