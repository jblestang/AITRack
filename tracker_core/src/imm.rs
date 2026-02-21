//! Interacting Multiple Models (IMM) filter.
//!
//! Maintains a bank of Kalman filters and a probability vector μ over models.
//!
//! Model bank (4 models):
//! 1. CV-steady: Low process noise, for straight flight.
//! 2. CV-agile:  High process noise, for random manoeuvres.
//! 3. CT-Left:   Constant turn left (+0.3 rad/s ≈ 9G at 250m/s).
//! 4. CT-Right:  Constant turn right (-0.3 rad/s).

use crate::{
    kf::{CvKalmanFilter, CvKfConfig, KalmanFilter, KfUpdateResult},
    types::{DMat, DVec, StateCov, StateVec},
};
use nalgebra::{DMatrix, Matrix6};

// ---------------------------------------------------------------------------
// CT (Constant Turn) Kalman filter
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CtKalmanFilter {
    pub omega: f64,
    pub sigma_p: f64,
    pub sigma_v: f64,
}

impl CtKalmanFilter {
    pub fn new(omega: f64, sigma_p: f64, sigma_v: f64) -> Self {
        Self { omega, sigma_p, sigma_v }
    }

    pub fn transition_matrix(omega: f64, dt: f64) -> Matrix6<f64> {
        let mut f = Matrix6::<f64>::identity();
        if omega.abs() < 1e-4 {
            // Near-zero turn rate: use CV
            f[(0, 3)] = dt;
            f[(1, 4)] = dt;
            f[(2, 5)] = dt;
        } else {
            let (s, c) = ((omega * dt).sin(), (omega * dt).cos());
            let (so, co) = (s / omega, (1.0 - c) / omega);
            f[(0, 3)] = so;
            f[(0, 4)] = -co;
            f[(1, 3)] = co;
            f[(1, 4)] = so;
            f[(2, 5)] = dt;
            f[(3, 3)] = c;
            f[(3, 4)] = -s;
            f[(4, 3)] = s;
            f[(4, 4)] = c;
        }
        f
    }

    pub fn predict(&self, state: &StateVec, cov: &StateCov, dt: f64) -> (StateVec, StateCov) {
        let f = Self::transition_matrix(self.omega, dt);
        let q_diag = nalgebra::Vector6::new(
            self.sigma_p * self.sigma_p * dt,
            self.sigma_p * self.sigma_p * dt,
            0.1, // altitude
            self.sigma_v * self.sigma_v * dt,
            self.sigma_v * self.sigma_v * dt,
            0.1,
        );
        let q = StateCov::from_diagonal(&q_diag);
        let new_state = f * state;
        let new_cov = f * cov * f.transpose() + q;
        (new_state, new_cov)
    }

    pub fn update(&self, state: &StateVec, cov: &StateCov, z: &DVec, h: &DMat, r: &DMat) -> KfUpdateResult {
        // Linear update (H is independent of omega)
        let kf = CvKalmanFilter::new(CvKfConfig { process_noise_std: 0.0 });
        kf.update(state, cov, z, h, r)
    }
}

// ---------------------------------------------------------------------------
// IMM Filter
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ImmModel {
    pub name: &'static str,
    pub prob: f64,
    pub state: StateVec,
    pub cov: StateCov,
}

#[derive(Clone, Debug)]
pub struct ImmState {
    pub models: Vec<ImmModel>,
    pub transition: Vec<Vec<f64>>,
    pub fused_state: StateVec,
    pub fused_cov: StateCov,
}

const N_MODELS: usize = 4;

impl ImmState {
    pub fn new(state: StateVec, cov: StateCov) -> Self {
        #[rustfmt::skip]
        let transition = vec![
            // to:  CV-st, CV-ag, CT-L,  CT-R
            vec![   0.92,  0.04,  0.02,  0.02 ], // from CV-st
            vec![   0.05,  0.85,  0.05,  0.05 ], // from CV-ag
            vec![   0.02,  0.08,  0.88,  0.02 ], // from CT-L
            vec![   0.02,  0.08,  0.02,  0.88 ], // from CT-R
        ];

        let models = vec![
            ImmModel { name: "CV-st", prob: 0.50, state, cov },
            ImmModel { name: "CV-ag", prob: 0.20, state, cov },
            ImmModel { name: "CT-L",  prob: 0.15, state, cov },
            ImmModel { name: "CT-R",  prob: 0.15, state, cov },
        ];

        Self {
            fused_state: state,
            fused_cov: cov,
            models,
            transition,
        }
    }

    pub fn predict(
        &mut self,
        dt: f64,
        kf_st: &CvKalmanFilter,
        kf_ag: &CvKalmanFilter,
        kf_ctl: &CtKalmanFilter,
        kf_ctr: &CtKalmanFilter,
    ) {
        let mixed = self.interaction();

        // Predict each model
        self.models[0].state = kf_st.predict(&mixed[0].0, &mixed[0].1, dt).0;
        self.models[0].cov   = kf_st.predict(&mixed[0].0, &mixed[0].1, dt).1;

        self.models[1].state = kf_ag.predict(&mixed[1].0, &mixed[1].1, dt).0;
        self.models[1].cov   = kf_ag.predict(&mixed[1].0, &mixed[1].1, dt).1;

        self.models[2].state = kf_ctl.predict(&mixed[2].0, &mixed[2].1, dt).0;
        self.models[2].cov   = kf_ctl.predict(&mixed[2].0, &mixed[2].1, dt).1;

        self.models[3].state = kf_ctr.predict(&mixed[3].0, &mixed[3].1, dt).0;
        self.models[3].cov   = kf_ctr.predict(&mixed[3].0, &mixed[3].1, dt).1;

        // Propagate probabilities
        let old_probs: Vec<f64> = self.models.iter().map(|m| m.prob).collect();
        for j in 0..N_MODELS {
            self.models[j].prob = (0..N_MODELS)
                .map(|i| self.transition[i][j] * old_probs[i])
                .sum::<f64>()
                .max(1e-30);
        }
        self.normalise_probs();
        self.fuse();
    }

    pub fn update(
        &mut self,
        z: &DVec,
        h: &DMat,
        r: &DMat,
        kf_st: &CvKalmanFilter,
        kf_ag: &CvKalmanFilter,
        kf_ctl: &CtKalmanFilter,
        kf_ctr: &CtKalmanFilter,
    ) {
        let mut likelihoods = Vec::with_capacity(N_MODELS);

        // Update each model and collect likelihoods
        for j in 0..N_MODELS {
            let res = match j {
                0 => kf_st.update(&self.models[j].state, &self.models[j].cov, z, h, r),
                1 => kf_ag.update(&self.models[j].state, &self.models[j].cov, z, h, r),
                2 => kf_ctl.update(&self.models[j].state, &self.models[j].cov, z, h, r),
                3 => kf_ctr.update(&self.models[j].state, &self.models[j].cov, z, h, r),
                _ => unreachable!(),
            };
            likelihoods.push(gaussian_likelihood(&res.innovation, &res.innovation_cov));
            self.models[j].state = res.state;
            self.models[j].cov = res.cov;
        }

        // μ_j ∝ L_j · μ̄_j
        let total: f64 = (0..N_MODELS)
            .map(|j| likelihoods[j] * self.models[j].prob)
            .sum::<f64>()
            .max(1e-30);

        for j in 0..N_MODELS {
            self.models[j].prob = (likelihoods[j] * self.models[j].prob / total).max(1e-10);
        }
        self.normalise_probs();
        self.fuse();
    }

    fn interaction(&self) -> Vec<(StateVec, StateCov)> {
        let mut mixed = Vec::with_capacity(N_MODELS);
        for j in 0..N_MODELS {
            let c_bar: f64 = (0..N_MODELS).map(|i| self.transition[i][j] * self.models[i].prob).sum();
            let c_bar = c_bar.max(1e-30);

            let mut x_mix = StateVec::zeros();
            for i in 0..N_MODELS {
                let mu_ij = self.transition[i][j] * self.models[i].prob / c_bar;
                x_mix += self.models[i].state * mu_ij;
            }

            let mut p_mix = StateCov::zeros();
            for i in 0..N_MODELS {
                let mu_ij = self.transition[i][j] * self.models[i].prob / c_bar;
                p_mix += self.models[i].cov * mu_ij;
                let dx = self.models[i].state - x_mix;
                p_mix += (dx * dx.transpose()) * mu_ij;
            }
            mixed.push((x_mix, p_mix));
        }
        mixed
    }

    fn fuse(&mut self) {
        let mut x_fused = StateVec::zeros();
        for m in &self.models { x_fused += m.state * m.prob; }
        
        let mut p_fused = StateCov::zeros();
        for m in &self.models {
            p_fused += m.cov * m.prob;
            let dx = m.state - x_fused;
            p_fused += (dx * dx.transpose()) * m.prob;
        }
        self.fused_state = x_fused;
        self.fused_cov = p_fused;
    }

    fn normalise_probs(&mut self) {
        let sum: f64 = self.models.iter().map(|m| m.prob).sum();
        let sum = sum.max(1e-30);
        for m in &mut self.models { m.prob /= sum; }
    }

    pub fn dominant_model(&self) -> &'static str {
        self.models.iter().max_by(|a, b| a.prob.partial_cmp(&b.prob).unwrap()).unwrap().name
    }
}

fn gaussian_likelihood(innovation: &DVec, s: &DMat) -> f64 {
    let dim = innovation.len();
    let det = s.determinant().abs();
    if det < 1e-30 { return 1e-30; }
    let s_inv = s.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(dim, dim));
    let maha2 = (innovation.transpose() * &s_inv * innovation)[0];
    let norm = ((2.0 * std::f64::consts::PI).powi(dim as i32) * det).sqrt();
    (-0.5 * maha2).exp() / norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kf::CvKfConfig;
    use nalgebra::{DMatrix, DVector, Vector6};

    #[test]
    fn imm_predict_does_not_diverge() {
        let state = Vector6::new(0.0, 0.0, 0.0, 100.0, 0.0, 0.0);
        let cov = StateCov::identity() * 100.0;
        let mut imm = ImmState::new(state, cov);

        let kf_st = CvKalmanFilter::new(CvKfConfig { process_noise_std: 1.0 });
        let kf_ag = CvKalmanFilter::new(CvKfConfig { process_noise_std: 30.0 });
        let kf_ctl = CtKalmanFilter::new(0.3, 10.0, 5.0);
        let kf_ctr = CtKalmanFilter::new(-0.3, 10.0, 5.0);

        for _ in 0..20 {
            imm.predict(0.5, &kf_st, &kf_ag, &kf_ctl, &kf_ctr);
        }
        assert!(imm.fused_state[0].is_finite());
    }

    #[test]
    fn imm_ct_model_gains_weight_during_turn() {
        let state = Vector6::new(0.0, 0.0, 0.0, 200.0, 0.0, 0.0);
        let cov = StateCov::identity() * 100.0;
        let mut imm = ImmState::new(state, cov);

        let kf_st = CvKalmanFilter::new(CvKfConfig { process_noise_std: 1.0 });
        let kf_ag = CvKalmanFilter::new(CvKfConfig { process_noise_std: 30.0 });
        let kf_ctl = CtKalmanFilter::new(0.2, 10.0, 5.0);
        let kf_ctr = CtKalmanFilter::new(-0.2, 10.0, 5.0);
        
        let h = DMatrix::from_row_slice(2, 6, &[1.,0.,0.,0.,0.,0., 0.,1.,0.,0.,0.,0.]);
        let r = DMatrix::from_diagonal(&DVector::from_vec(vec![1000.0, 1000.0]));

        let omega = 0.2f64;
        let dt = 0.5;
        let mut tx = 0.0f64; let mut ty = 0.0f64; let mut tvx = 200.0f64; let mut tvy = 0.0f64;

        for _ in 0..20 {
            let heading = tvy.atan2(tvx);
            let nh = heading + omega * dt;
            let v = (tvx*tvx + tvy*tvy).sqrt();
            tx += v * heading.cos() * dt;
            ty += v * heading.sin() * dt;
            tvx = v * nh.cos();
            tvy = v * nh.sin();

            imm.predict(dt, &kf_st, &kf_ag, &kf_ctl, &kf_ctr);
            let z = DVector::from_vec(vec![tx, ty]);
            imm.update(&z, &h, &r, &kf_st, &kf_ag, &kf_ctl, &kf_ctr);
        }

        let ctl_prob = imm.models[2].prob;
        let st_prob = imm.models[0].prob;
        assert!(ctl_prob > st_prob, "CT-L should dominate, got L={ctl_prob:.3} ST={st_prob:.3}");
    }
}
