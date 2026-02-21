//! `sim` — Scenario simulator: target trajectories, radar measurements, replay.

pub mod replay;
pub mod scenarios;
pub mod radar_sim;
pub mod target;

pub use scenarios::{Scenario, ScenarioKind};
pub use target::{Target, MotionSpec};
pub use radar_sim::RadarSimulator;
pub use replay::{ReplayLog, load_replay, save_replay};
