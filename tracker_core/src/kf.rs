//! Kalman filter: predict and update steps.
//!
//! # Design choices
//! - We use a **linear KF** with a constant-velocity (CV) motion model for Phase A.
//! - All math is done in `f64` via `nalgebra` for numerical stability.
//! - The `KalmanFilter` trait allows future IMM with multiple models (Phase D).
//!
//! ## State vector
//! x = [px, py, pz, vx, vy, vz]ᵀ  (6-dimensional)
//!
//! ## CV Transition model
//! F = I₆ + dt * [[0₃ I₃]; [0₃ 0₃]]
//! i.e. px += vx*dt, etc.
//!
//! ## Process noise Q (Singer model, simplified)
//! Q = q * diag([dt³/3, dt³/3, dt³/3, dt, dt, dt]) (block)

use crate::types::{DMat, DVec, StateCov, StateVec};
use nalgebra::Matrix6;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for a Kalman filter model (predict + update).
pub trait KalmanFilter {
    /// Predict state and covariance forward by `dt` seconds.
    fn predict(&self, state: &StateVec, cov: &StateCov, dt: f64) -> (StateVec, StateCov);

    /// Update state and covariance given an observation `z`, observation
    /// matrix `H` and measurement noise `R`.
    /// Returns (updated_state, updated_cov, innovation, innovation_cov).
    fn update(
        &self,
        state: &StateVec,
        cov: &StateCov,
        z: &DVec,
        h: &DMat,
        r: &DMat,
    ) -> KfUpdateResult;

    /// Update state and covariance utilizing Joint Probabilistic Data Association (JPDA).
    /// `meas_probs` is a list of (measurement z, association probability β).
    /// `miss_prob` is the probability β_0 that none of the measurements are correct (missed detection).
    /// Returns the updated state and covariance.
    fn update_jpda(
        &self,
        state: &StateVec,
        cov: &StateCov,
        meas_probs: &[(DVec, f64)],
        miss_prob: f64,
        h: &DMat,
        r: &DMat,
    ) -> (StateVec, StateCov, f64);
}

/// Result of a KF update step, exposed for debug UI.
#[derive(Clone, Debug)]
pub struct KfUpdateResult {
    pub state: StateVec,
    pub cov: StateCov,
    /// Innovation ν = z − H·x
    pub innovation: DVec,
    /// Innovation covariance S = H·P·Hᵀ + R
    pub innovation_cov: DMat,
    /// Kalman gain K
    pub kalman_gain: DMat,
}

// ---------------------------------------------------------------------------
// Constant Velocity model
// ---------------------------------------------------------------------------

/// Configuration for the CV Kalman filter.
#[derive(Clone, Debug)]
pub struct CvKfConfig {
    /// Process noise spectral density (acceleration variance, m²/s⁴).
    /// Higher = more maneuvering allowed.
    pub process_noise_std: f64,
}

impl Default for CvKfConfig {
    fn default() -> Self {
        Self {
            process_noise_std: 5.0, // 5 m/s² std dev (allows tracking slow turns)
        }
    }
}

/// Constant-Velocity Kalman filter (6-state, linear).
#[derive(Clone, Debug)]
pub struct CvKalmanFilter {
    pub config: CvKfConfig,
}

impl CvKalmanFilter {
    pub fn new(config: CvKfConfig) -> Self {
        Self { config }
    }

    /// Build state transition matrix F for timestep dt.
    pub fn transition_matrix(dt: f64) -> Matrix6<f64> {
        let mut f = Matrix6::<f64>::identity();
        // position += velocity * dt
        f[(0, 3)] = dt;
        f[(1, 4)] = dt;
        f[(2, 5)] = dt;
        f
    }

    /// Build process noise matrix Q for timestep dt.
    /// Uses discrete white noise acceleration model (DWNA).
    fn process_noise(dt: f64, q_std: f64) -> Matrix6<f64> {
        let q = q_std * q_std; // variance
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt3 * dt;

        // Block structure for [pos; vel] with acceleration noise
        // Q_pos = q * dt^4/4,  Q_pos_vel = q * dt^3/2,  Q_vel = q * dt^2
        let mut qm = Matrix6::<f64>::zeros();
        for i in 0..3usize {
            qm[(i, i)] = q * dt4 / 4.0;
            qm[(i + 3, i + 3)] = q * dt2;
            qm[(i, i + 3)] = q * dt3 / 2.0;
            qm[(i + 3, i)] = q * dt3 / 2.0;
        }
        qm
    }
}

impl KalmanFilter for CvKalmanFilter {
    fn predict(&self, state: &StateVec, cov: &StateCov, dt: f64) -> (StateVec, StateCov) {
        let f = Self::transition_matrix(dt);
        let q = Self::process_noise(dt, self.config.process_noise_std);
        let predicted_state = f * state;
        let predicted_cov = f * cov * f.transpose() + q;
        (predicted_state, predicted_cov)
    }

    fn update(
        &self,
        state: &StateVec,
        cov: &StateCov,
        z: &DVec,
        h: &DMat,
        r: &DMat,
    ) -> KfUpdateResult {
        // Convert fixed-size state to dynamic for DMat multiplication
        let x_dyn = DVec::from_iterator(6, state.iter().copied());
        let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());

        // Innovation: ν = z − H·x
        let hx = h * &x_dyn;
        let innovation = z - hx;

        // Innovation covariance: S = H·P·Hᵀ + R
        let hp = h * &p_dyn;
        let s = &hp * h.transpose() + r;

        // Kalman gain: K = P·Hᵀ·S⁻¹  (LU for numerical stability)
        let s_lu = s.clone().lu();
        let s_inv = s_lu
            .try_inverse()
            .expect("Innovation covariance S is singular");
        let k = &p_dyn * h.transpose() * &s_inv;

        // Updated state: x' = x + K·ν
        let state_update = &k * &innovation;
        let new_state = StateVec::from_fn(|r, _| state[r] + state_update[r]);

        // Updated covariance: Joseph form P' = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ
        let kh = &k * h;
        let i_kh = DMat::identity(6, 6) - kh;
        let new_p_dyn = &i_kh * &p_dyn * i_kh.transpose() + &k * r * k.transpose();
        let new_cov = StateCov::from_fn(|r, c| new_p_dyn[(r, c)]);

        KfUpdateResult {
            state: new_state,
            cov: new_cov,
            innovation,
            innovation_cov: s,
            kalman_gain: k,
        }
    }

    fn update_jpda(
        &self,
        state: &StateVec,
        cov: &StateCov,
        meas_probs: &[(DVec, f64)],
        miss_prob: f64,
        h: &DMat,
        r: &DMat,
    ) -> (StateVec, StateCov, f64) {
        let x_dyn = DVec::from_iterator(6, state.iter().copied());
        let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());

        let hx = h * &x_dyn;
        let hp = h * &p_dyn;
        let s = &hp * h.transpose() + r;

        let s_lu = s.clone().lu();
        // If S is singular, avoid crashing.
        let s_inv = match s_lu.try_inverse() {
            Some(inv) => inv,
            None => return (*state, *cov, 1e-30),
        };
        let k = &p_dyn * h.transpose() * &s_inv;

        let mut combined_innovation = DVec::zeros(2);
        let mut sum_beta_nu_nut = DMat::zeros(2, 2);

        // PDA combined innovation
        let mut total_hit_prob = 0.0;
        let mut innovations = Vec::with_capacity(meas_probs.len());
        for (z, beta) in meas_probs {
            let nu = z - &hx;
            combined_innovation += &nu * *beta;
            sum_beta_nu_nut += &nu * nu.transpose() * *beta;
            total_hit_prob += *beta;
            innovations.push(nu);
        }

        // State update: x' = x + K * (sum beta_j nu_j)
        let state_update = &k * &combined_innovation;
        let new_state = StateVec::from_fn(|r, _| state[r] + state_update[r]);

        // Covariance update (PDA format)
        // Pc = P - K*S*K^T (standard update cov)
        // P' = P - (1 - beta_0) * K*S*K^T + dP
        let k_s_kt = &k * s.clone() * k.transpose();
        // Spread of innovations: dP = K * [ sum(beta_j nu_j nu_j^T) - nu_combined * nu_combined^T ] * K^T
        let spread = sum_beta_nu_nut - &combined_innovation * combined_innovation.transpose();
        let dp = &k * spread * k.transpose();

        let new_p_dyn = &p_dyn - &k_s_kt * total_hit_prob + dp;
        let new_cov = StateCov::from_fn(|r, c| new_p_dyn[(r, c)]);

        // Compute a combined marginal likelihood for the track using the normal distribution 
        // to feed into the mixed IMM likelihood.
        let mut jpda_likelihood = miss_prob; // In PDA, base density accounts for missed
        let dim = 2.0;
        let det = s.determinant().abs().max(1e-30);
        let norm = 1.0 / ((2.0 * std::f64::consts::PI).powf(dim / 2.0) * det.sqrt());
        for (nu, (_, beta)) in innovations.iter().zip(meas_probs.iter()) {
            let mahalanobis_sq = (nu.transpose() * &s_inv * nu)[(0, 0)];
            let l = norm * (-0.5 * mahalanobis_sq).exp();
            jpda_likelihood += beta * l;
        }

        (new_state, new_cov, jpda_likelihood)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

    #[test]
    fn predict_constant_velocity() {
        let kf = CvKalmanFilter::new(CvKfConfig::default());
        // Object at (0,0,0) moving at (10,0,0) m/s
        let state = Vector6::new(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);
        let cov = StateCov::identity();

        let (pred_state, _pred_cov) = kf.predict(&state, &cov, 1.0);
        assert_abs_diff_eq!(pred_state[0], 10.0, epsilon = 1e-9); // x moved
        assert_abs_diff_eq!(pred_state[3], 10.0, epsilon = 1e-9); // vx unchanged
    }

    #[test]
    fn update_reduces_uncertainty() {
        let kf = CvKalmanFilter::new(CvKfConfig::default());
        let state = Vector6::new(100.0, 50.0, 0.0, 5.0, 2.0, 0.0);
        let cov = StateCov::identity() * 100.0;

        // 2D cartesian observation
        let h = DMat::from_row_slice(2, 6, &[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]);
        let r = DMat::from_diagonal(&DVec::from_vec(vec![9.0, 9.0])); // 3m std dev
        let z = DVec::from_vec(vec![101.0, 51.0]);

        let res = kf.update(&state, &cov, &z, &h, &r);
        // Posterior covariance trace must be less than prior
        let prior_trace: f64 = (0..6).map(|i| cov[(i, i)]).sum();
        let post_trace: f64 = (0..6).map(|i| res.cov[(i, i)]).sum();
        assert!(post_trace < prior_trace, "Update should reduce uncertainty");
    }
}
