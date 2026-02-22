//! Mahalanobis gating: determines whether a measurement is "close enough"
//! to a predicted track to be considered as a potential association.
//!
//! # Gating criterion
//! d²(z, track) = νᵀ S⁻¹ ν  where ν = z − H·x̂_pred,  S = H·P_pred·Hᵀ + R
//!
//! Accept if d² < χ²(p, dof) for confidence level p.
//!
//! # Gate threshold table (99.7% ≈ 3σ for 2D)
//! dof=2: χ²(0.99, 2) ≈ 9.21
//! dof=3: χ²(0.99, 3) ≈ 11.34

use crate::types::{DMat, DVec, StateCov, StateVec};
use nalgebra::Vector6;
use std::collections::HashMap;

/// A Uniform Grid spatial index for 2D points.
/// Used to accelerate gating by only checking measurements in nearby cells.
pub struct SpatialGrid {
    cell_size: f64,
    /// Maps cell key (ix, iy) to a list of measurement indices.
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    /// Insert a measurement into the grid at position (x, y).
    pub fn insert(&mut self, idx: usize, x: f64, y: f64) {
        let ix = (x / self.cell_size).floor() as i32;
        let iy = (y / self.cell_size).floor() as i32;
        self.cells.entry((ix, iy)).or_default().push(idx);
    }

    /// Query measurement indices in cells surrounding (x, y).
    /// By default, we check the cell containing (x, y) and its 8 direct neighbors.
    pub fn query_nearby(&self, x: f64, y: f64) -> Vec<usize> {
        let ix = (x / self.cell_size).floor() as i32;
        let iy = (y / self.cell_size).floor() as i32;

        let mut results = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(indices) = self.cells.get(&(ix + dx, iy + dy)) {
                    results.extend_from_slice(indices);
                }
            }
        }
        results
    }
}

/// Pre-computed χ² gate thresholds indexed by dimension [1..=6].
/// Value at index `d` is χ²(0.99, d).
pub const CHI2_99: [f64; 7] = [0.0, 6.63, 9.21, 11.34, 13.28, 15.09, 16.81];

/// Result of a gate check for one (track, measurement) pair.
#[derive(Clone, Debug)]
pub struct GateResult {
    /// Mahalanobis distance squared
    pub d2: f64,
    /// True if d² < gate threshold
    pub passes: bool,
    /// Innovation vector ν (for debug display and KF update reuse)
    pub innovation: DVec,
    /// Innovation covariance S (reused for KF update)
    pub innovation_cov: DMat,
}

/// Gate ellipse descriptor — used by the UI to draw the ellipse.
#[derive(Clone, Debug)]
pub struct GatingEllipse {
    /// Track centre (projected 2D)
    pub center: (f64, f64),
    /// Semi-axes in x/y (approx from S diagonal)
    pub semi_x: f64,
    pub semi_y: f64,
    /// Rotation of the ellipse (radians, from SVD/eigen of S)
    pub angle: f64,
    /// Track this ellipse belongs to (for coloring)
    pub track_idx: usize,
}

/// Check whether measurement `z` passes the Mahalanobis gate for a predicted
/// track state `(state, cov)` using observation matrix `H` and measurement
/// noise `R`.
pub fn mahalanobis_gate(
    state: &StateVec,
    cov: &StateCov,
    z: &DVec,
    h: &DMat,
    r: &DMat,
    gate_threshold: f64,
) -> GateResult {
    // Innovation ν = z − H·x
    let hx = h * Vector6::from(*state);
    let innovation = z - hx;

    // Innovation covariance S = H·P·Hᵀ + R
    let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());
    let hp = h * &p_dyn;
    let s = &hp * h.transpose() + r;

    // Mahalanobis distance² = νᵀ S⁻¹ ν (use LU for numerical safety)
    let s_lu = s.clone().lu();
    let d2 = match s_lu.try_inverse() {
        Some(s_inv) => {
            let v = &s_inv * &innovation;
            innovation.dot(&v)
        }
        None => f64::INFINITY, // degenerate case — reject
    };

    GateResult {
        d2,
        passes: d2 < gate_threshold,
        innovation,
        innovation_cov: s,
    }
}

/// Compute a 2D GatingEllipse for display, given predicted state + covariance + H.
/// Projects the innovation covariance to 2D via SVD.
pub fn compute_gate_ellipse(
    state: &StateVec,
    cov: &StateCov,
    h: &DMat,
    r: &DMat,
    track_idx: usize,
    scale: f64, // multiply semi-axes by this (e.g., 3.0 for 3σ)
) -> GatingEllipse {
    let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());
    let s = h * &p_dyn * h.transpose() + r;

    // Extract 2×2 top-left block of S for 2D projection
    let s00 = s[(0, 0)].max(0.0);
    let s11 = s[(1, 1)].max(0.0);
    let s01 = if s.nrows() > 1 && s.ncols() > 1 {
        s[(0, 1)]
    } else {
        0.0
    };

    // Eigenvalues of 2×2: λ = (s00+s11)/2 ± sqrt(((s00-s11)/2)² + s01²)
    let mean = (s00 + s11) / 2.0;
    let diff = (s00 - s11) / 2.0;
    let disc = (diff * diff + s01 * s01).sqrt();
    let lambda1 = (mean + disc).max(0.0);
    let lambda2 = (mean - disc).max(0.0);

    let angle = s01.atan2(diff) / 2.0;

    GatingEllipse {
        center: (state[0], state[1]),
        semi_x: scale * lambda1.sqrt(),
        semi_y: scale * lambda2.sqrt(),
        angle,
        track_idx,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn simple_h() -> DMat {
        DMat::from_row_slice(2, 6, &[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
    }

    fn simple_r(sigma: f64) -> DMat {
        DMat::from_diagonal(&DVector::from_vec(vec![sigma * sigma, sigma * sigma]))
    }

    #[test]
    fn point_inside_gate() {
        use nalgebra::Vector6;
        let state = Vector6::new(10.0, 10.0, 0.0, 0.0, 0.0, 0.0);
        let cov = StateCov::identity() * 100.0;
        let h = simple_h();
        let r = simple_r(3.0);
        // Measurement right at prediction → d² ≈ 0
        let z = DVector::from_vec(vec![10.0, 10.0]);
        let res = mahalanobis_gate(&state, &cov, &z, &h, &r, CHI2_99[2]);
        assert!(res.passes, "Point at track location must pass gate");
        assert!(res.d2 < 1e-6);
    }

    #[test]
    fn point_outside_gate() {
        use nalgebra::Vector6;
        let state = Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        // Small covariance → tight gate
        let cov = StateCov::identity() * 0.01;
        let h = simple_h();
        let r = simple_r(0.1);
        // Measurement far away
        let z = DVector::from_vec(vec![1000.0, 1000.0]);
        let res = mahalanobis_gate(&state, &cov, &z, &h, &r, CHI2_99[2]);
        assert!(!res.passes, "Distant point must fail gate");
    }
}
