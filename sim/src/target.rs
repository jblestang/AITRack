//! Target trajectory models and state propagation.
//!
//! Each target has a 6-DOF true state [px,py,pz,vx,vy,vz] and a `MotionSpec`
//! describing how it moves. The simulator steps each target forward in time.

use serde::{Deserialize, Serialize};

/// Describes target motion between waypoints/events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MotionSpec {
    /// Constant velocity: no acceleration. State propagates as CV.
    ConstantVelocity,
    /// Constant-turn-rate on XY plane. `omega` = yaw rate (rad/s).
    ConstantTurn { omega: f64 },
    /// Constant acceleration model. `ax, ay, az` in m/s².
    ConstantAccel { ax: f64, ay: f64, az: f64 },
    /// Waypoint tracker: target teleports velocity to head toward next waypoint.
    Waypoints {
        /// List of (t, x, y, z) waypoints
        waypoints: Vec<[f64; 4]>,
        speed: f64,
    },
    /// Segmented: switch motion model at given sim times.
    /// `segments` is sorted by time ascending: [(t_start, MotionSpec), ...].
    /// The active spec is the last one whose t_start <= current_t.
    Segmented {
        segments: Vec<(f64, Box<MotionSpec>)>,
    },
}

/// A simulated target with ground-truth state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Target {
    /// Unique target ID (used for metrics)
    pub id: u64,
    /// True state [px, py, pz, vx, vy, vz]
    pub state: [f64; 6],
    /// Motion model for this target
    pub motion: MotionSpec,
    /// Optional: target disappears after this time
    pub disappear_at: Option<f64>,
    /// Optional: target appears after this time (no measurements before)
    pub appear_at: Option<f64>,
    /// History of past states (for visualization of true trajectory)
    #[serde(skip)]
    pub history: std::collections::VecDeque<[f64; 6]>,
}

impl Target {
    /// Propagate true state by `dt` seconds according to motion spec.
    pub fn step(&mut self, t: f64, dt: f64) {
        // Record current state before stepping
        self.history.push_front(self.state);
        // Keep bounded history (e.g. 500 steps = 50s at 10Hz)
        if self.history.len() > 500 {
            self.history.pop_back();
        }

        let s = &mut self.state;
        match &self.motion.clone() {
            MotionSpec::ConstantVelocity => {
                s[0] += s[3] * dt;
                s[1] += s[4] * dt;
                s[2] += s[5] * dt;
            }
            MotionSpec::ConstantTurn { omega } => {
                let v = (s[3] * s[3] + s[4] * s[4]).sqrt();
                let heading = s[4].atan2(s[3]);
                let new_heading = heading + omega * dt;
                s[0] += v * heading.cos() * dt;
                s[1] += v * heading.sin() * dt;
                s[3] = v * new_heading.cos();
                s[4] = v * new_heading.sin();
            }
            MotionSpec::ConstantAccel { ax, ay, az } => {
                s[0] += s[3] * dt + 0.5 * ax * dt * dt;
                s[1] += s[4] * dt + 0.5 * ay * dt * dt;
                s[2] += s[5] * dt + 0.5 * az * dt * dt;
                s[3] += ax * dt;
                s[4] += ay * dt;
                s[5] += az * dt;
            }
            MotionSpec::Waypoints { waypoints, speed } => {
                // Find current target waypoint
                let target_wp = waypoints.iter().find(|wp| wp[0] >= t);
                if let Some(wp) = target_wp {
                    let dx = wp[1] - s[0];
                    let dy = wp[2] - s[1];
                    let dz = wp[3] - s[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist > 1.0 {
                        s[3] = speed * dx / dist;
                        s[4] = speed * dy / dist;
                        s[5] = speed * dz / dist;
                    } else {
                        s[3] = 0.0;
                        s[4] = 0.0;
                        s[5] = 0.0;
                    }
                }
                s[0] += s[3] * dt;
                s[1] += s[4] * dt;
                s[2] += s[5] * dt;
            }
            MotionSpec::Segmented { segments } => {
                // Find last segment whose start time <= t
                let active = segments.iter().filter(|(t_start, _)| *t_start <= t).last();
                if let Some((_, spec)) = active {
                    // Temporarily apply this spec for one step, prevent history duplication
                    let mut tmp = Target {
                        id: 0,
                        state: *s,
                        motion: *spec.clone(),
                        appear_at: None,
                        disappear_at: None,
                        history: std::collections::VecDeque::new(),
                    };
                    tmp.step(t, dt);
                    *s = tmp.state;
                } else {
                    // Before first segment: CV
                    s[0] += s[3] * dt;
                    s[1] += s[4] * dt;
                    s[2] += s[5] * dt;
                }
            }
        }
    }

    /// True if target is active at time `t`.
    pub fn is_active(&self, t: f64) -> bool {
        if let Some(appear) = self.appear_at {
            if t < appear {
                return false;
            }
        }
        if let Some(disappear) = self.disappear_at {
            if t >= disappear {
                return false;
            }
        }
        true
    }

    /// 2D position
    pub fn pos_2d(&self) -> (f64, f64) {
        (self.state[0], self.state[1])
    }
}
