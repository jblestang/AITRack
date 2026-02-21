//! Observation models: H matrix, R matrix, polar↔cartesian conversion, Jacobians.
//!
//! # Measurement types supported
//! - **Cartesian 2D**: z = [x, y], H is constant 2×6 matrix
//! - **Polar 2D**: z = [range, azimuth], H is the linearised Jacobian at x̂

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

pub type DMat = DMatrix<f64>;
pub type DVec = DVector<f64>;

/// Trait for a radar observation model.
pub trait ObservationModel {
    /// Observation matrix H (linear approx. at `state`)
    fn h_matrix(&self, state: &[f64; 6]) -> DMat;
    /// Measurement noise covariance R
    fn r_matrix(&self) -> DMat;
    /// Map state to expected measurement h(x)
    fn apply(&self, state: &[f64; 6]) -> DVec;
}

// ---------------------------------------------------------------------------
// Cartesian 2D
// ---------------------------------------------------------------------------

/// Cartesian XY observation model (range pre-converted to x,y).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartesianXY {
    /// Std deviation in x (meters)
    pub sigma_x: f64,
    /// Std deviation in y (meters)
    pub sigma_y: f64,
}

impl CartesianXY {
    pub fn new(sigma_x: f64, sigma_y: f64) -> Self {
        Self { sigma_x, sigma_y }
    }
}

impl ObservationModel for CartesianXY {
    fn h_matrix(&self, _state: &[f64; 6]) -> DMat {
        // Linear: z = [px, py]
        DMatrix::from_row_slice(2, 6, &[
            1., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0.,
        ])
    }

    fn r_matrix(&self) -> DMat {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.sigma_x * self.sigma_x,
            self.sigma_y * self.sigma_y,
        ]))
    }

    fn apply(&self, state: &[f64; 6]) -> DVec {
        DVector::from_vec(vec![state[0], state[1]])
    }
}

// ---------------------------------------------------------------------------
// Polar 2D (radar: range + azimuth)
// ---------------------------------------------------------------------------

/// Polar observation model for a ground-based 2D radar.
/// z = [range, azimuth] where azimuth is measured from north (0 = East in our convention).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolarObservation {
    /// Radar position in world frame (x, y)
    pub radar_pos: [f64; 2],
    /// Range noise std dev (meters)
    pub sigma_r: f64,
    /// Azimuth noise std dev (radians)
    pub sigma_az: f64,
}

impl PolarObservation {
    pub fn new(radar_pos: [f64; 2], sigma_r: f64, sigma_az: f64) -> Self {
        Self { radar_pos, sigma_r, sigma_az }
    }

    /// Convert polar [range, az] to cartesian [x, y] relative to world origin.
    pub fn polar_to_cartesian(&self, range: f64, azimuth: f64) -> (f64, f64) {
        let x = self.radar_pos[0] + range * azimuth.cos();
        let y = self.radar_pos[1] + range * azimuth.sin();
        (x, y)
    }

    /// Compute [range, azimuth] from a state vector.
    pub fn state_to_polar(&self, state: &[f64; 6]) -> (f64, f64) {
        let dx = state[0] - self.radar_pos[0];
        let dy = state[1] - self.radar_pos[1];
        let range = (dx * dx + dy * dy).sqrt();
        let az = dy.atan2(dx);
        (range, az)
    }
}

impl ObservationModel for PolarObservation {
    fn h_matrix(&self, state: &[f64; 6]) -> DMat {
        // Jacobian of h = [range, az] w.r.t. x = [px, py, ...]
        let dx = state[0] - self.radar_pos[0];
        let dy = state[1] - self.radar_pos[1];
        let r2 = dx * dx + dy * dy;
        let r = r2.sqrt().max(1e-3);

        // ∂range/∂px = dx/r,   ∂range/∂py = dy/r,   others = 0
        // ∂az/∂px = -dy/r²,    ∂az/∂py = dx/r²,     others = 0
        DMatrix::from_row_slice(2, 6, &[
             dx / r,  dy / r,  0., 0., 0., 0.,
            -dy / r2,  dx / r2, 0., 0., 0., 0.,
        ])
    }

    fn r_matrix(&self) -> DMat {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.sigma_r * self.sigma_r,
            self.sigma_az * self.sigma_az,
        ]))
    }

    fn apply(&self, state: &[f64; 6]) -> DVec {
        let (r, az) = self.state_to_polar(state);
        DVector::from_vec(vec![r, az])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartesian_h_shape() {
        let model = CartesianXY::new(5.0, 5.0);
        let state = [100.0, 200.0, 0.0, 10.0, 0.0, 0.0];
        let h = model.h_matrix(&state);
        assert_eq!((h.nrows(), h.ncols()), (2, 6));
    }

    #[test]
    fn polar_roundtrip() {
        let model = PolarObservation::new([0.0, 0.0], 50.0, 0.01);
        let (r, az) = model.state_to_polar(&[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((r - 1000.0).abs() < 1e-6);
        assert!(az.abs() < 1e-6);
        let (x, y) = model.polar_to_cartesian(r, az);
        assert!((x - 1000.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }
}
