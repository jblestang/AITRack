//! Simulation advance and tracker pipeline systems.

use crate::resources::{PlayMode, ResetEvent, SimState, StepEvent, TrackerAppState};
use bevy::prelude::*;
use sim::replay::GroundTruthFrame;
use tracker_core::types::Measurement;

/// Keyboard input system: Space=play/pause, Period=step, R=reset, =/- = speed.
pub fn keyboard_control_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimState>,
    mut step_events: EventWriter<StepEvent>,
    mut reset_events: EventWriter<ResetEvent>,
) {
    if keys.just_pressed(KeyCode::Space) {
        sim_state.play_mode = match sim_state.play_mode {
            PlayMode::Playing => PlayMode::Paused,
            _ => PlayMode::Playing,
        };
    }
    if keys.just_pressed(KeyCode::Period) {
        sim_state.play_mode = PlayMode::StepOnce;
        step_events.send(StepEvent);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        reset_events.send(ResetEvent);
    }
    if keys.just_pressed(KeyCode::Equal) {
        sim_state.speed_multiplier = (sim_state.speed_multiplier * 2.0).min(64.0);
    }
    if keys.just_pressed(KeyCode::Minus) {
        sim_state.speed_multiplier = (sim_state.speed_multiplier / 2.0).max(0.125);
    }
}

/// Advance simulation time and generate radar batches when playing.
/// One Bevy frame can advance multiple sim ticks (speed_multiplier).
pub fn advance_simulation_system(
    time: Res<Time>,
    mut sim_state: ResMut<SimState>,
    mut tracker_state: ResMut<TrackerAppState>,
) {
    let mode = sim_state.play_mode;
    if mode == PlayMode::Paused {
        return;
    }

    let dt = sim_state.scenario.sim_dt;
    // How many ticks to advance this frame
    let ticks = if mode == PlayMode::StepOnce {
        sim_state.play_mode = PlayMode::Paused;
        1
    } else {
        // Time-scaled: try to advance sim_time proportionally to wall time
        let elapsed = time.delta_secs_f64() * sim_state.speed_multiplier as f64;
        (elapsed / dt).max(1.0) as usize
    };

    for _ in 0..ticks {
        if sim_state.sim_time >= sim_state.scenario.duration {
            sim_state.play_mode = PlayMode::Paused;
            break;
        }

        // Step targets
        let t = sim_state.sim_time;
        for target in &mut sim_state.scenario.targets {
            target.step(t, dt);
        }

        sim_state.sim_time += dt;
        sim_state.tick += 1;

        // Record ground truth
        let gt_frame = GroundTruthFrame {
            time: sim_state.sim_time,
            targets: sim_state
                .scenario
                .targets
                .iter()
                .filter(|t| t.is_active(sim_state.sim_time))
                .map(|t| sim::replay::TargetState {
                    id: t.id,
                    state: t.state,
                })
                .collect(),
        };
        sim_state.gt_history.push(gt_frame);

        // Generate radar batches.
        // Copy scalar fields first to avoid multi-borrow on sim_state (ResMut is opaque).
        let targets_snapshot = sim_state.scenario.targets.clone();
        let batch_time = sim_state.sim_time;
        let mut meas_id = sim_state.meas_id_counter;
        let new_batches =
            sim_state
                .radar_sim
                .generate_batches(&targets_snapshot, batch_time, &mut meas_id);
        sim_state.meas_id_counter = meas_id;

        // Process each batch through the tracker
        for batch in new_batches {
            sim_state.batch_history.push(batch.clone());
            let output = tracker_state.pipeline.process_batch(&batch);

            // Collect measurements for rendering
            tracker_state.current_measurements = batch.measurements.clone();

            tracker_state.timing_history.push(output.total_time_us);
            if tracker_state.timing_history.len() > 100 {
                tracker_state.timing_history.remove(0);
            }

            // Accumulate metrics for all confirmed tracks
            use tracker_core::track::TrackStatus;
            for track in &output.tracks {
                if track.status == TrackStatus::Confirmed {
                    // Greedy association: find closest active target
                    let mut min_dist_sq = f64::MAX;
                    let mut best_target = None;
                    for target in &sim_state.scenario.targets {
                        if target.is_active(sim_state.sim_time) {
                            let dx = track.state[0] - target.state[0];
                            let dy = track.state[1] - target.state[1];
                            let dist_sq = dx * dx + dy * dy;
                            if dist_sq < min_dist_sq {
                                min_dist_sq = dist_sq;
                                best_target = Some(target.id);
                            }
                        }
                    }

                    let metrics = tracker_state.all_track_metrics.entry(track.id).or_insert_with(|| {
                        crate::resources::TrackMetrics {
                            target_id: best_target,
                            start_time: track.born_at,
                            end_time: sim_state.sim_time,
                            sum_sq_err: 0.0,
                            count: 0,
                        }
                    });

                    metrics.end_time = sim_state.sim_time;
                    if best_target.is_some() {
                        metrics.sum_sq_err += min_dist_sq;
                        metrics.count += 1;
                        if metrics.target_id.is_none() {
                            metrics.target_id = best_target;
                        }
                    }
                }
            }

            tracker_state.last_output = Some(output);
        }
    }
}

/// Handle reset events: restart the scenario from scratch.
pub fn reset_system(
    mut reset_events: EventReader<ResetEvent>,
    mut sim_state: ResMut<SimState>,
    mut tracker_state: ResMut<TrackerAppState>,
) {
    for _ in reset_events.read() {
        let scenario = sim_state.scenario.clone();
        let seed = scenario.seed;
        *sim_state = SimState::new(scenario, seed);
        tracker_state.pipeline.reset();
        tracker_state.last_output = None;
        tracker_state.current_measurements.clear();
        tracker_state.timing_history.clear();
        tracker_state.all_track_metrics.clear();
        tracing::info!("Simulation reset");
    }
}
