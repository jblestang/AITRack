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

    /// Apply temporal bias correction to a measurement timestamp.
    pub fn correct_timestamp(&self, t: f64) -> f64 {
        t - self.dt0_at(t)
    }

    fn dt0_at(&self, t: f64) -> f64 {
        self.temporal.dt0 + self.temporal.dt_dot * t
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

    /// Phase A stub — will do online estimation in Phase C.
    /// `innovations` holds (track_id_u64, innovation_vec) for high-confidence pairs.
    pub fn update(&mut self, _innovations: &[(u64, Vec<f64>)]) {}
}
