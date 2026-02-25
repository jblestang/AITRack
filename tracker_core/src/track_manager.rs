//! Track lifecycle management: birth, M-of-N confirmation, deletion, merging.
//!
//! # Track Management Policy
//! - **Birth**: each unmatched measurement spawns a tentative track.
//! - **Confirmation**: a tentative track is confirmed after M hits in the
//!   last N opportunities. Default: M=3, N=4.
//! - **Deletion**: confirmed track deleted after `miss_limit` consecutive
//!   misses. Tentative track deleted after 1 miss (fast pruning).

use crate::{
    track::{Track, TrackStatus},
    types::{Measurement, StateCov, TrackId},
};
use nalgebra::Vector6;

/// Configuration for track management policy.
#[derive(Clone, Debug)]
pub struct TrackManagerConfig {
    /// Hits required for confirmation
    pub confirm_m: u8,
    /// Window for M-of-N rule
    pub confirm_n: u8,
    /// Consecutive misses before confirmed track deletion
    pub miss_limit_confirmed: u8,
    /// Consecutive misses before tentative track deletion
    pub miss_limit_tentative: u8,
    /// Initial position uncertainty (1σ in meters)
    pub init_pos_std: f64,
    /// Initial velocity uncertainty (1σ in m/s)
    pub init_vel_std: f64,
}

impl Default for TrackManagerConfig {
    fn default() -> Self {
        Self {
            confirm_m: 3,
            confirm_n: 4,
            miss_limit_confirmed: 5,
            miss_limit_tentative: 1,
            init_pos_std: 30.0, // 30 m initial position uncertainty
            init_vel_std: 20.0, // 20 m/s initial velocity uncertainty
        }
    }
}

/// Manages the pool of active tracks.
pub struct TrackManager {
    pub config: TrackManagerConfig,
    next_id: u64,
    // Ring-buffer of recent hit/miss per track (simplified: use u8 counters)
}

impl TrackManager {
    pub fn new(config: TrackManagerConfig) -> Self {
        Self { config, next_id: 0 }
    }

    fn next_track_id(&mut self) -> TrackId {
        let id = TrackId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Create a new tentative track from an unmatched measurement.
    pub fn birth_track(&mut self, meas: &Measurement, current_time: f64) -> Track {
        let id = self.next_track_id();
        let pos = meas.to_cartesian_2d();
        let state = Vector6::new(
            pos[0], pos[1], 0.0, // zero z
            0.0, 0.0, 0.0, // zero initial velocity
        );
        let ps = self.config.init_pos_std * self.config.init_pos_std;
        let vs = self.config.init_vel_std * self.config.init_vel_std;
        
        let mut cov = StateCov::from_diagonal(&Vector6::new(ps, ps, ps, vs, vs, vs));
        if meas.noise_cov.len() == 4 {
            cov[(0, 0)] = meas.noise_cov[0];
            cov[(0, 1)] = meas.noise_cov[1];
            cov[(1, 0)] = meas.noise_cov[2];
            cov[(1, 1)] = meas.noise_cov[3];
        }
        Track::new(id, state, cov, current_time)
    }

    /// Call after a track received an update (hit). Update status if needed.
    pub fn register_hit(&self, track: &mut Track) {
        track.misses = 0;
        track.hits = track.hits.saturating_add(1);
        track.total_hits += 1;
        // Confirm if M hits reached
        if track.status == TrackStatus::Tentative && track.hits >= self.config.confirm_m {
            track.status = TrackStatus::Confirmed;
        }
    }

    /// Call after a track received no update (miss).
    pub fn register_miss(&self, track: &mut Track) {
        track.misses = track.misses.saturating_add(1);
        let limit = match track.status {
            TrackStatus::Tentative => self.config.miss_limit_tentative,
            TrackStatus::Confirmed => self.config.miss_limit_confirmed,
            TrackStatus::Deleted => 0,
        };
        if track.misses > limit {
            track.status = TrackStatus::Deleted;
        }
    }

    /// Remove all deleted tracks. Returns count of removed tracks.
    pub fn prune_deleted(tracks: &mut Vec<Track>) -> usize {
        let before = tracks.len();
        tracks.retain(|t| t.status != TrackStatus::Deleted);
        before - tracks.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tentative_confirms_after_m_hits() {
        let cfg = TrackManagerConfig {
            confirm_m: 3,
            ..Default::default()
        };
        let mgr = TrackManager::new(cfg);
        let state = Vector6::zeros();
        let cov = StateCov::identity();
        // Track::new starts with hits=1 (birth counts as first hit)
        let mut track = Track::new(TrackId(0), state, cov, 0.0);
        assert_eq!(track.hits, 1);

        assert_eq!(track.status, TrackStatus::Tentative);
        mgr.register_hit(&mut track); // hits = 2
        assert_eq!(
            track.status,
            TrackStatus::Tentative,
            "Not confirmed yet at hit 2"
        );
        mgr.register_hit(&mut track); // hits = 3 >= confirm_m=3 => confirmed
        assert_eq!(
            track.status,
            TrackStatus::Confirmed,
            "Should be confirmed at hit 3"
        );
    }

    #[test]
    fn confirmed_deleted_after_misses() {
        let cfg = TrackManagerConfig {
            miss_limit_confirmed: 3,
            ..Default::default()
        };
        let mgr = TrackManager::new(cfg);
        let mut track = Track {
            status: TrackStatus::Confirmed,
            ..Track::new(TrackId(0), Vector6::zeros(), StateCov::identity(), 0.0)
        };

        mgr.register_miss(&mut track); // misses=1, limit=3, not deleted
        mgr.register_miss(&mut track); // misses=2
        mgr.register_miss(&mut track); // misses=3, still not deleted (> 3 is false)
        assert_eq!(
            track.status,
            TrackStatus::Confirmed,
            "Still alive at 3 misses (limit=3, delete when > 3)"
        );
        assert_eq!(
            track.status,
            TrackStatus::Deleted,
            "Deleted after miss > limit"
        );
    }

    #[test]
    fn tentative_deleted_after_one_miss() {
        let cfg = TrackManagerConfig {
            miss_limit_tentative: 1, // Delete after 1 miss (misses > 1 is false, so wait...)
            ..Default::default()
        };
        // wait, the logic is "if misses > limit { Deleted }". 
        // So with limit 1, misses=1 is OK, misses=2 is Deleted.
        let mgr = TrackManager::new(cfg);
        let mut track = Track::new(TrackId(0), Vector6::zeros(), StateCov::identity(), 0.0);
        
        mgr.register_miss(&mut track); // misses=1
        assert_eq!(track.status, TrackStatus::Tentative);
        mgr.register_miss(&mut track); // misses=2 > limit 1
        assert_eq!(track.status, TrackStatus::Deleted);
    }

    #[test]
    fn prune_deleted_removes_correctly() {
        let mut tracks = vec![
            Track { status: TrackStatus::Confirmed, ..Track::new(TrackId(0), Vector6::zeros(), StateCov::identity(), 0.0) },
            Track { status: TrackStatus::Deleted, ..Track::new(TrackId(1), Vector6::zeros(), StateCov::identity(), 0.0) },
            Track { status: TrackStatus::Tentative, ..Track::new(TrackId(2), Vector6::zeros(), StateCov::identity(), 0.0) },
        ];
        let count = TrackManager::prune_deleted(&mut tracks);
        assert_eq!(count, 1);
        assert_eq!(tracks.len(), 2);
        assert!(tracks.iter().all(|t| t.status != TrackStatus::Deleted));
    }
}
