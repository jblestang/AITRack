//! Track: state, covariance, status, history, optional IMM state.

use crate::{
    imm::ImmState,
    types::{StateCov, StateVec, TrackId},
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Maximum number of past states kept for display / smoothing.
/// Maximum length of the track state history (for drawing trails).
const HISTORY_LEN: usize = 1024;

/// Lifecycle status of a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackStatus {
    /// Not yet confirmed — may be spurious
    Tentative,
    /// Confirmed: seen M times out of last N frames
    Confirmed,
    /// Marked for removal
    Deleted,
}

/// A single multi-target track.
#[derive(Clone, Debug)]
pub struct Track {
    /// Unique identifier
    pub id: TrackId,
    /// Estimated state vector [px,py,pz,vx,vy,vz]  (fused if IMM active)
    pub state: StateVec,
    /// State estimation covariance  (fused if IMM active)
    pub cov: StateCov,
    /// Lifecycle status
    pub status: TrackStatus,
    /// Number of hits in the current M-of-N window
    pub hits: u8,
    /// Consecutive missed update count
    pub misses: u8,
    /// Total number of updates (for confidence display)
    pub total_hits: u32,
    /// Simulation time of last update
    pub last_updated: f64,
    /// Simulation time of birth
    pub born_at: f64,
    /// Past state snapshots (most recent last)
    pub history: VecDeque<StateVec>,
    /// Optional IMM state (Some when IMM mode is enabled in pipeline config)
    pub imm: Option<ImmState>,
}

impl Track {
    /// Create a new tentative track from an initial state estimate.
    /// `use_imm` initialises the IMM state alongside the classical state.
    pub fn new(id: TrackId, state: StateVec, cov: StateCov, birth_time: f64) -> Self {
        let mut history = VecDeque::with_capacity(HISTORY_LEN);
        history.push_back(state);
        Self {
            id,
            state,
            cov,
            status: TrackStatus::Tentative,
            hits: 1,
            misses: 0,
            total_hits: 1,
            last_updated: birth_time,
            born_at: birth_time,
            history,
            imm: None,
        }
    }

    /// Initialise the IMM state for this track (call once after birth when enabled).
    pub fn init_imm(&mut self) {
        self.imm = Some(ImmState::new(self.state, self.cov));
    }

    /// Push a new state snapshot to the history ring-buffer.
    pub fn push_history(&mut self) {
        if self.history.len() >= HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back(self.state);
    }

    /// Returns 2D projected [x, y] position (for rendering)
    pub fn position_2d(&self) -> (f64, f64) {
        (self.state[0], self.state[1])
    }

    /// Returns 2D velocity [vx, vy] (for rendering velocity arrow)
    pub fn velocity_2d(&self) -> (f64, f64) {
        (self.state[3], self.state[4])
    }

    /// Name of the dominant IMM model (or "CV" if IMM not active).
    pub fn motion_label(&self) -> &'static str {
        self.imm.as_ref().map(|i| i.dominant_model()).unwrap_or("CV")
    }
}
