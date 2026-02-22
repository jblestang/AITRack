//! `sim` — Scenario simulator: target trajectories, radar measurements, replay.

pub mod radar_sim;
pub mod replay;
pub mod scenarios;
pub mod target;

pub use radar_sim::RadarSimulator;
pub use replay::{load_replay, save_replay, ReplayLog};
pub use scenarios::{Scenario, ScenarioKind};
pub use target::{MotionSpec, Target};
