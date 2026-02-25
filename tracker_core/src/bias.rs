//! Fixed bias compensation module.
//!
//! This module provides structures to store and apply fixed sensor biases
//! provided via configuration. Dynamic estimation has been disabled.

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

/// Per-sensor bias state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorBiasState {
    pub sensor_id: SensorId,
    pub spatial: RadarSpatialBias,
    pub temporal: RadarTimeBias,
}

impl Default for SensorBiasState {
    fn default() -> Self {
        Self {
            sensor_id: SensorId(0),
            spatial: RadarSpatialBias::default(),
            temporal: RadarTimeBias::default(),
        }
    }
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
    pub fn correct_cartesian_2d(&self, x: f64, y: f64, sensor_pos: [f64; 3]) -> (f64, f64) {
        let xt = x - sensor_pos[0] - self.spatial.dx;
        let yt = y - sensor_pos[1] - self.spatial.dy;

        // Apply negative rotation
        let cos_t = (-self.spatial.dtheta).cos();
        let sin_t = (-self.spatial.dtheta).sin();
        
        let mut xr_local = cos_t * xt - sin_t * yt;
        let mut yr_local = sin_t * xt + cos_t * yt;

        // Apply negative range bias
        let r = (xr_local * xr_local + yr_local * yr_local).sqrt();
        if r > 1.0 {
            let r_corr = (r - self.spatial.br) / r;
            xr_local *= r_corr;
            yr_local *= r_corr;
        }

        (xr_local + sensor_pos[0], yr_local + sensor_pos[1])
    }

    pub fn correct_timestamp(&self, t: f64) -> f64 {
        t - self.dt0_at(t)
    }

    fn dt0_at(&self, t: f64) -> f64 {
        self.temporal.dt0 + self.temporal.dt_dot * t
    }
    
    pub fn get_spatial_bias_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.spatial.dx, self.spatial.dy])
    }
}

/// Bias lookup service — holds fixed per-sensor biases.
#[derive(Clone, Debug, Default)]
pub struct BiasEstimator {
    pub sensor_states: std::collections::HashMap<SensorId, SensorBiasState>,
}

impl BiasEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_sensor_state(&self, sensor_id: SensorId) -> &SensorBiasState {
        self.sensor_states.get(&sensor_id).unwrap_or(&STUB_BIAS)
    }
}

lazy_static::lazy_static! {
    static ref STUB_BIAS: SensorBiasState = SensorBiasState::default();
}
