//! Unscented Kalman Filter (UKF) implementation.
//!
//! Uses sigma points to propagate mean and covariance through non-linear functions.
//! More accurate than EKF for highly non-linear systems and doesn't require Jacobians.

use crate::types::{DMat, DVec, StateCov, StateVec};
use crate::kf::{KalmanFilter, KfUpdateResult, CvKalmanFilter, CvKfConfig};

/// Unscented Kalman Filter.
#[derive(Clone, Debug)]
pub struct UnscentedKalmanFilter {
    pub cv_kf: CvKalmanFilter,
    /// Parameter alpha (usually small positive value, e.g., 0.001)
    pub alpha: f64,
    /// Parameter beta (usually 2.0 for Gaussian distributions)
    pub beta: f64,
    /// Parameter kappa (usually 0.0)
    pub kappa: f64,
}

impl UnscentedKalmanFilter {
    pub fn new(config: CvKfConfig) -> Self {
        Self {
            cv_kf: CvKalmanFilter::new(config),
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    /// Update step using Unscented Transform.
    /// `z` is the measurement.
    /// `h_func` is a closure that maps state to measurement.
    /// `r` is the measurement noise covariance.
    pub fn update_ukf<F>(
        &self,
        state: &StateVec,
        cov: &StateCov,
        z: &DVec,
        h_func: F,
        r: &DMat,
    ) -> KfUpdateResult
    where
        F: Fn(&StateVec) -> DVec,
    {
        let n: f64 = 6.0; // dimension of state
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        
        // 1. Generate Sigma Points
        let sigma_points = self.generate_sigma_points(state, cov, lambda);
        
        // 2. Predict measurements for each sigma point
        let mut predicted_measurements = Vec::new();
        for sp in &sigma_points {
            predicted_measurements.push(h_func(sp));
        }
        
        // 3. Recover predicted measurement mean and covariance
        let m_dim = z.len();
        let (z_mean, sz, pxz) = self.unscented_transform(
            &sigma_points,
            &predicted_measurements,
            state, // state is already the mean of sigma points if linear
            lambda,
            n,
            m_dim,
            r
        );

        // 4. Update
        let s_inv = sz.clone().lu().try_inverse().expect("Innovation covariance S is singular in UKF");
        let k = pxz * s_inv;
        
        let innovation = z - &z_mean;
        let state_update = &k * &innovation;
        let new_state = StateVec::from_fn(|r, _| state[r] + state_update[r]);
        
        let new_p_dyn = DMat::from_row_slice(6, 6, cov.as_slice()) - &k * sz.clone() * k.transpose();
        let new_cov = StateCov::from_fn(|r, c| new_p_dyn[(r, c)]);

        KfUpdateResult {
            state: new_state,
            cov: new_cov,
            innovation,
            innovation_cov: sz,
            kalman_gain: k,
        }
    }

    fn generate_sigma_points(&self, mean: &StateVec, cov: &StateCov, lambda: f64) -> Vec<StateVec> {
        let n = 6;
        let p_dyn = DMat::from_row_slice(6, 6, cov.as_slice());
        let spread = ( (n as f64 + lambda) * p_dyn ).cholesky().expect("UKF Covariance not positive definite").l();
        
        let mut sigma_points = Vec::with_capacity(2 * n + 1);
        sigma_points.push(*mean);
        
        for i in 0..n {
            let col = spread.column(i);
            sigma_points.push(StateVec::from_fn(|r, _| mean[r] + col[r]));
            sigma_points.push(StateVec::from_fn(|r, _| mean[r] - col[r]));
        }
        
        sigma_points
    }

    fn unscented_transform(
        &self,
        sigma_points: &[StateVec],
        z_points: &[DVec],
        _x_mean: &StateVec,
        lambda: f64,
        n: f64,
        m_dim: usize,
        r: &DMat
    ) -> (DVec, DMat, DMat) {
        let w_m0 = lambda / (n + lambda);
        let w_c0 = w_m0 + (1.0 - self.alpha * self.alpha + self.beta);
        let w_i = 1.0 / (2.0 * (n + lambda));
        
        // 1. Mean of measurements
        let mut z_mean = &z_points[0] * w_m0;
        for zp in z_points.iter().skip(1) {
            z_mean += zp * w_i;
        }
        
        // Handle azimuth wrap-around in mean if needed
        if m_dim == 2 {
            // This is a bit tricky for a mean. A better way is to move to complex 
            // but for narrow distributions we just normalize.
            while z_mean[1] > std::f64::consts::PI { z_mean[1] -= 2.0 * std::f64::consts::PI; }
            while z_mean[1] < -std::f64::consts::PI { z_mean[1] += 2.0 * std::f64::consts::PI; }
        }
        
        // 2. Innovation covariance and cross-covariance
        let mut sz = DMat::zeros(m_dim, m_dim);
        let mut pxz = DMat::zeros(6, m_dim);
        
        for (i, (sp, zp)) in sigma_points.iter().zip(z_points.iter()).enumerate() {
            let wi = if i == 0 { w_c0 } else { w_i };
            
            let mut dz = zp - &z_mean;
            if m_dim == 2 {
                while dz[1] > std::f64::consts::PI { dz[1] -= 2.0 * std::f64::consts::PI; }
                while dz[1] < -std::f64::consts::PI { dz[1] += 2.0 * std::f64::consts::PI; }
            }
            
            let dx = DVec::from_iterator(6, sp.iter().copied()) - DVec::from_iterator(6, _x_mean.iter().copied());
            
            sz += &dz * dz.transpose() * wi;
            pxz += &dx * dz.transpose() * wi;
        }
        
        sz += r;
        
        (z_mean, sz, pxz)
    }
}

impl KalmanFilter for UnscentedKalmanFilter {
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
        // Linear mode for trait compatibility
        self.update_ukf(state, cov, z, |s| h * DVec::from_iterator(6, s.iter().copied()), r)
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
    use nalgebra::Vector6;

    #[test]
    fn test_ukf_polar_update() {
        let ukf = UnscentedKalmanFilter::new(CvKfConfig::default());
        let state = Vector6::new(100.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let cov = StateCov::identity() * 10.0;

        // Polar measurement at [105, 0.01]
        let z = DVec::from_vec(vec![105.0, 0.01]);
        
        let h_func = |s: &StateVec| {
            let r = (s[0]*s[0] + s[1]*s[1]).sqrt();
            let az = s[1].atan2(s[0]);
            DVec::from_vec(vec![r, az])
        };
        
        let r = DMat::from_diagonal(&DVec::from_vec(vec![1.0, 0.0001]));

        let res = ukf.update_ukf(&state, &cov, &z, h_func, &r);
        
        // x should increase towards 105
        assert!(res.state[0] > 100.0);
        // y should increase towards something small positive due to azimuth 0.01
        assert!(res.state[1] > 0.0);
    }
}
