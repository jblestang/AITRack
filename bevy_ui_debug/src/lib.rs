//! `bevy_ui_debug` — Bevy ECS debug frontend for the tracker.
//!
//! # Feature gate
//! This crate is only useful in interactive mode. The `debug_ui` feature
//! enables Bevy rendering. In batch/CLI mode, `sim` + `tracker_core` are used
//! directly without Bevy.
//!
//! # Architecture
//! - **Resources**: `SimState`, `TrackerAppState`, `RenderSettings`
//! - **Events**: `StepEvent`, `ResetEvent`, `PipelineOutputEvent`
//! - **Systems** (in order): `advance_simulation`, `run_tracker`,
//!   `render_*`, `ui_*`

pub mod app;
pub mod render;
pub mod resources;
pub mod systems;
pub mod ui;

#[cfg(feature = "debug_ui")]
pub use app::run_debug_app;
