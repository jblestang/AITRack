//! `tracker_core` — Core multi-target tracking algorithms.
//!
//! # Module layout
//! - [`types`]        — Fundamental types (IDs, state vectors, measurements)
//! - [`track`]        — Track struct and status management
//! - [`kf`]           — Kalman filter (predict / update)
//! - [`gating`]       — Mahalanobis gating
//! - [`association`]  — Bipartite graph, connected components, Hungarian solver
//! - [`track_manager`]— Birth / confirmation / deletion logic
//! - [`pipeline`]     — Full tracking pipeline orchestrator
//! - [`bias`]         — Bias estimation stub (Phase C)
//! - [`metrics`]      — RMSE, ID-switch, precision/recall

pub mod association;
pub mod bias;
pub mod gating;
pub mod imm;
pub mod kf;
pub mod metrics;
pub mod pipeline;
pub mod track;
pub mod track_manager;
pub mod types;

pub use pipeline::{Pipeline, PipelineConfig, PipelineDebugData, PipelineOutput};
pub use track::{Track, TrackStatus};
pub use types::{
    Measurement, MeasurementId, MeasurementValue, SensorId, StateCov, StateVec, TrackId,
};
