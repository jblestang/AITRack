//! Bevy resources shared across systems.

use bevy::prelude::*;
use sim::radar_sim::RadarSimulator;
use sim::scenarios::Scenario;
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
    /// Flags if currently extracting frames for a movie
    pub is_recording: bool,
    pub recording_frame: u64,
    /// Whether the app should automatically encode and exit at the end of the scenario
    pub auto_record: bool,
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
            is_recording: false,
            recording_frame: 0,
            auto_record: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tracker state
// ---------------------------------------------------------------------------

/// Per-track statistical metrics accumulated during the simulation run.
#[derive(Clone, Debug, Default)]
pub struct TrackMetrics {
    /// Associated ground truth target ID (closest target when confirmed)
    pub target_id: Option<u64>,
    /// Time the track was born (sim time, s)
    pub start_time: f64,
    /// Last time the track was updated (sim time, s)
    pub end_time: f64,
    /// Cumulative sum of squared position errors
    pub sum_sq_err: f64,
    /// Number of frames this track was evaluated
    pub count: u64,
}

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
    /// Accumulated evaluation metrics for all tracks ever confirmed
    pub all_track_metrics: std::collections::HashMap<TrackId, TrackMetrics>,
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
            all_track_metrics: std::collections::HashMap::new(),
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
    /// Show simulated ground truth tracks
    pub show_ground_truth: bool,
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
            show_ground_truth: true,
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
