use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use sim::scenarios::{Scenario, ScenarioKind};
use tracker_core::pipeline::PipelineConfig;

use crate::{
    render::{
        render_associations_system, render_gates_system, render_ground_truth_system,
        render_measurements_system, render_radars_system, render_tracks_system,
    },
    resources::{RenderSettings, ResetEvent, SimState, StepEvent, TrackerAppState},
    systems::{
        advance_simulation_system, camera_control_system, keyboard_control_system,
        recording_system, reset_system,
    },
    ui::{ui_control_panel, ui_cost_overlay, ui_evaluation_panel, ui_timeline_panel, ui_track_inspector},
};

/// Main entry point for the interactive debug UI.
pub fn run_debug_app(kind: ScenarioKind, seed: u64, auto_record: bool) {
    let scenario = Scenario::build(kind.clone(), seed);
    let mut sim_state = SimState::new(scenario, seed);

    if auto_record {
        sim_state.is_recording = true;
        sim_state.play_mode = crate::resources::PlayMode::Playing;
        sim_state.speed_multiplier = 1.0;
        let _ = std::fs::remove_dir_all("video_frames");
        std::fs::create_dir_all("video_frames").unwrap();
        tracing::info!("Auto-recording enabled. Started capturing frames to video_frames/");
    }

    let pipeline_cfg = PipelineConfig::default();
    let tracker_state = TrackerAppState::new(pipeline_cfg);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "AITRack — Multi-Sensor Tracker Debug UI".into(),
                resolution: (1400., 900.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .insert_resource(sim_state)
        .insert_resource(tracker_state)
        .insert_resource(RenderSettings::default())
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.08)))
        .add_event::<StepEvent>()
        .add_event::<ResetEvent>()
        .add_systems(Startup, setup_camera)
        // Simulation + gizmo rendering (no egui dependency)
        .add_systems(
            Update,
            (
                keyboard_control_system,
                camera_control_system,
                reset_system,
                advance_simulation_system,
                render_radars_system,
                render_measurements_system,
                render_tracks_system,
                render_ground_truth_system,
                render_gates_system,
                render_associations_system,
                recording_system,
            )
                .chain(),
        )
        // egui UI panels — must run AFTER EguiSet::InitContexts
        .add_systems(
            Update,
            (
                ui_control_panel,
                ui_timeline_panel,
                ui_track_inspector,
                ui_cost_overlay,
                ui_evaluation_panel,
            )
                .chain(),
        )
        .run();
}

/// Set up a 2D orthographic camera centred at the origin.
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
}
