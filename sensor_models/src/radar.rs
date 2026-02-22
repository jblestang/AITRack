//! Radar sensor parameters.

use serde::{Deserialize, Serialize};

/// Physical configuration of a radar sensor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadarParams {
    /// Radar position (x, y, z) in world coordinates (meters)
    pub position: [f64; 3],
    /// Heading (yaw) of the radar boresight (radians)
    pub heading: f64,
    /// Maximum detection range (meters)
    pub max_range: f64,
    /// Field of view half-angle (radians)
    pub fov_half: f64,
    /// Probability of detection per target per scan
    pub p_detection: f64,
    /// Mean number of clutter returns per scan per km²
    pub lambda_clutter: f64,
    /// Update rate (Hz) — average time between batches = 1.0 / refresh_rate
    pub refresh_rate: f64,
    /// Measurement noise: range standard deviation (meters)
    pub range_noise_std: f64,
    /// Measurement noise: azimuth standard deviation (radians)
    pub azimuth_noise_std: f64,
    /// Whether this radar outputs cartesian (true) or polar (false) measurements
    pub output_cartesian: bool,
}

impl Default for RadarParams {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            heading: 0.0,
            max_range: 100_000.0,           // 100 km
            fov_half: std::f64::consts::PI, // 360° coverage
            p_detection: 0.9,
            lambda_clutter: 0.5e-6,  // 0.5 returns / km² / scan
            refresh_rate: 1.0,       // 1 Hz default
            range_noise_std: 50.0,   // 50 m
            azimuth_noise_std: 0.01, // ~0.6° ≈ 1 mrad
            output_cartesian: true,
        }
    }
}
