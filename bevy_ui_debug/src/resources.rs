//! Bevy resources shared across systems.

use bevy::prelude::*;
use sim::scenarios::Scenario;
use sim::radar_sim::RadarSimulator;
use tracker_core::pipeline::{Pipeline, PipelineConfig, PipelineOutput};
use tracker_core::types::{RadarBatch, SensorId, TrackId};

// ---------------------------------------------------------------------------
// Playback control
// ---------------------------------------------------------------------------

/// Current playback mode of the simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Resource)]
pub enum PlayMode {
    #[default]
    Paused,
    Playing,
    /// Advance exactly one simulation tick then pause
    StepOnce,
}

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

/// Global simulation state: scenario + clock.
#[derive(Resource)]
pub struct SimState {
    pub scenario: Scenario,
    pub sim_time: f64,
    pub tick: u64,
    pub play_mode: PlayMode,
    pub speed_multiplier: f32,
    /// Radar simulator (owns RNG)
    pub radar_sim: RadarSimulator,
    /// Rolling measurement-ID counter
    pub meas_id_counter: u64,
    /// All batches generated so far (for replay display)
    pub batch_history: Vec<RadarBatch>,
    /// All ground-truth frames for playback metric computation
    pub gt_history: Vec<sim::replay::GroundTruthFrame>,
}

impl SimState {
    pub fn new(scenario: Scenario, seed: u64) -> Self {
        let radar_sim = RadarSimulator::new(scenario.radars.clone(), seed);
        Self {
            radar_sim,
            scenario,
            sim_time: 0.0,
            tick: 0,
            play_mode: PlayMode::Paused,
            speed_multiplier: 1.0,
            meas_id_counter: 0,
            batch_history: Vec::new(),
            gt_history: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tracker state
// ---------------------------------------------------------------------------

/// Tracker state: pipeline + last output.
#[derive(Resource)]
pub struct TrackerAppState {
    pub pipeline: Pipeline,
    pub last_output: Option<PipelineOutput>,
    pub selected_track: Option<TrackId>,
    pub selected_sensor: Option<SensorId>,
    /// All measurements in the current tick (for rendering)
    pub current_measurements: Vec<tracker_core::types::Measurement>,
    /// Pipeline timing history (rolling window for display)
    pub timing_history: Vec<u64>,
}

impl TrackerAppState {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            pipeline: Pipeline::new(config),
            last_output: None,
            selected_track: None,
            selected_sensor: None,
            current_measurements: Vec::new(),
            timing_history: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering settings
// ---------------------------------------------------------------------------

/// Rendering parameters (adjustable from UI).
#[derive(Resource)]
pub struct RenderSettings {
    /// Scale from simulation meters to screen pixels
    pub world_to_screen_scale: f32,
    /// Show gating ellipses
    pub show_gates: bool,
    /// Show associations (edges track ↔ measurement)
    pub show_associations: bool,
    /// Show history trails
    pub show_trails: bool,
    /// Show components (Phase B: color by component)
    pub show_components: bool,
    /// Show only confirmed tracks (hide tentative)
    pub confirmed_only: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            world_to_screen_scale: 0.004,
            show_gates: true,
            show_associations: true,
            show_trails: true,
            show_components: false,
            confirmed_only: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Emitted to request a simulation step.
#[derive(Event, Default)]
pub struct StepEvent;

/// Emitted to reset the entire simulation.
#[derive(Event, Default)]
pub struct ResetEvent;
