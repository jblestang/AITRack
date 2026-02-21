use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use sim::scenarios::{Scenario, ScenarioKind};
use tracker_core::{
    kf::CvKfConfig,
    pipeline::PipelineConfig,
    track_manager::TrackManagerConfig,
};

use crate::{
    render::{
        render_associations_system, render_gates_system, render_measurements_system,
        render_radars_system, render_tracks_system,
    },
    resources::{RenderSettings, ResetEvent, SimState, StepEvent, TrackerAppState},
    systems::{advance_simulation_system, keyboard_control_system, reset_system},
    ui::{ui_control_panel, ui_cost_overlay, ui_timeline_panel, ui_track_inspector},
};

/// Main entry point for the interactive debug UI.
pub fn run_debug_app(kind: ScenarioKind, seed: u64) {
    let scenario = Scenario::build(kind.clone(), seed);
    let sim_state = SimState::new(scenario, seed);

    // Choose pipeline config based on scenario
    let pipeline_cfg = match kind {
        ScenarioKind::Fighter => PipelineConfig {
            use_imm: true,
            // Deep wider gate (20.0 ≈ 99.99% for 2-DOF) to absorb high-G innovations
            gate_threshold: 20.0,
            // Slow CV has moderate noise; fast CV and CT have high noise
            kf_config: CvKfConfig { process_noise_std: 3.0 },
            imm_sigma_fast: 150.0,
            imm_ct_sigma_p: 200.0,
            imm_ct_sigma_v: 100.0,
            // Let confirmed tracks survive up to 15 consecutive missed scans
            track_manager_config: TrackManagerConfig {
                miss_limit_confirmed: 15,
                miss_limit_tentative: 2,
                confirm_m: 3,
                ..Default::default()
            },
            ..Default::default()
        },
        _ => PipelineConfig {
            // Slightly looser gate for noisy scenarios
            gate_threshold: 13.8,
            track_manager_config: TrackManagerConfig {
                miss_limit_confirmed: 7,
                ..Default::default()
            },
            ..Default::default()
        },
    };
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
                reset_system,
                advance_simulation_system,
                render_radars_system,
                render_measurements_system,
                render_tracks_system,
                render_gates_system,
                render_associations_system,
            )
                .chain(),
        )
        // egui UI panels — must run AFTER EguiSet::InitContexts
        .add_systems(
            Update,
            (ui_control_panel, ui_timeline_panel, ui_track_inspector, ui_cost_overlay)
                .chain(),
        )
        .run();
}

/// Set up a 2D orthographic camera centred at the origin.
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
}
