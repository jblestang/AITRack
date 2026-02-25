//! Extended Kalman Filter (EKF) implementation.
//! 
//! Unlike the standard KF, the EKF linearizes non-linear observation models
//! by computing the Jacobian at the current state estimate.

use crate::types::{DMat, DVec, StateCov, StateVec};
use crate::kf::{KalmanFilter, KfUpdateResult, CvKalmanFilter, CvKfConfig};

/// Extended Kalman Filter implementation.
/// It reuses the Constant Velocity prediction from CvKalmanFilter but
/// supports non-linear updates.
#[derive(Clone, Debug)]
pub struct ExtendedKalmanFilter {
    pub cv_kf: CvKalmanFilter,
}

impl ExtendedKalmanFilter {
    pub fn new(config: CvKfConfig) -> Self {
        Self {
            cv_kf: CvKalmanFilter::new(config),
        }
    }

    /// Update step for EKF.
    /// `z` is the measurement.
    /// `hx` is the predicted measurement h(x_pred).
    /// `h_jacobian` is the Jacobian matrix H evaluated at x_pred.
    /// `r` is the measurement noise covariance.
    pub fn update_ekf(
        &self,
        state: &StateVec,
        cov: &StateCov,
        z: &DVec,
        hx: &DVec,
        h_jacobian: &DMat,
        r: &DMat,
    ) -> KfUpdateResult {
        let _x_dyn = DVec::from_iterator(6, state.iter().copied());
        let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());

        // Innovation: ν = z − h(x)
        let mut innovation = z - hx;
        
        // Handle azimuth wrap-around if it's a 2D polar measurement [r, az]
        if innovation.len() == 2 {
            // Assuming index 1 is azimuth in radians
            while innovation[1] > std::f64::consts::PI {
                innovation[1] -= 2.0 * std::f64::consts::PI;
            }
            while innovation[1] < -std::f64::consts::PI {
                innovation[1] += 2.0 * std::f64::consts::PI;
            }
        }

        // Innovation covariance: S = H·P·Hᵀ + R
        let hp = h_jacobian * &p_dyn;
        let s = &hp * h_jacobian.transpose() + r;

        // Kalman gain: K = P·Hᵀ·S⁻¹
        let s_lu = s.clone().lu();
        let s_inv = s_lu
            .try_inverse()
            .expect("Innovation covariance S is singular in EKF");
        let k = &p_dyn * h_jacobian.transpose() * &s_inv;

        // Updated state: x' = x + K·ν
        let state_update = &k * &innovation;
        let new_state = StateVec::from_fn(|r, _| state[r] + state_update[r]);

        // Updated covariance (Joseph form)
        let kh = &k * h_jacobian;
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
}

// Implement the basic KalmanFilter trait for EKF as well, 
// using linear approximation by default if called via standard trait.
impl KalmanFilter for ExtendedKalmanFilter {
    fn predict(&self, state: &StateVec, cov: &StateCov, dt: f64) -> (StateVec, StateCov) {
        self.cv_kf.predict(state, cov, dt)
    }

    fn update(
        &self,
        state: &StateVec,
        cov: &StateCov,
        z: &DVec,
        h: &DMat,
        r: &DMat,
    ) -> KfUpdateResult {
        // Fallback to linear: h(x) = H * x
        let _x_dyn = DVec::from_iterator(6, state.iter().copied());
        let hx = h * &_x_dyn;
        self.update_ekf(state, cov, z, &hx, h, r)
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
        self.cv_kf.update_jpda(state, cov, meas_probs, miss_prob, h, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

    #[test]
    fn test_ekf_polar_update() {
        let ekf = ExtendedKalmanFilter::new(CvKfConfig::default());
        let state = Vector6::new(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let cov = StateCov::identity() * 10.0;

        // Polar measurement at [105, 0.01]
        let z = DVec::from_vec(vec![105.0, 0.01]);
        
        // h(x) = [sqrt(x^2+y^2), atan2(y, x)]
        let hx = DVec::from_vec(vec![100.0, 0.0]);
        
        // Jacobian H at [100, 0]
        // dr/dx = x/r = 1, dr/dy = y/r = 0
        // daz/dx = -y/r^2 = 0, daz/dy = x/r^2 = 1/100
        let h_jac = DMat::from_row_slice(2, 6, &[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
        ]);
        
        let r = DMat::from_diagonal(&DVec::from_vec(vec![1.0, 0.0001]));

        let res = ekf.update_ekf(&state, &cov, &z, &hx, &h_jac, &r);
        
        // x should increase towards 105
        assert!(res.state[0] > 100.0);
        // y should increase towards something small positive due to azimuth 0.01
        assert!(res.state[1] > 0.0);
    }
}
