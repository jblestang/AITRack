//! egui-based UI panels: control bar, timeline, track inspector, metrics.

use crate::resources::{PlayMode, RenderSettings, ResetEvent, SimState, TrackerAppState};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use tracker_core::track::TrackStatus;


/// Control panel: Play/Pause/Step buttons + speed + scenario info.
pub fn ui_control_panel(
    mut contexts: EguiContexts,
    mut sim_state: ResMut<SimState>,
    mut reset_events: EventWriter<ResetEvent>,
    mut render: ResMut<RenderSettings>,
) {
    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };
    egui::TopBottomPanel::top("control_panel").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("🎯 AITRack");
            ui.separator();

            // Play/Pause button
            let play_label = match sim_state.play_mode {
                PlayMode::Playing => "⏸ Pause",
                _ => "▶ Play",
            };
            if ui.button(play_label).clicked() {
                sim_state.play_mode = match sim_state.play_mode {
                    PlayMode::Playing => PlayMode::Paused,
                    _ => PlayMode::Playing,
                };
            }

            // Step button
            if ui.button("⏭ Step").clicked() {
                sim_state.play_mode = PlayMode::StepOnce;
            }

            // Reset button
            if ui.button("🔄 Reset").clicked() {
                reset_events.send(ResetEvent);
            }

            ui.separator();

            let record_label = if sim_state.is_recording {
                "⏹ Stop & Encode"
            } else {
                "🎥 Record Movie"
            };
            if ui.button(record_label).clicked() {
                sim_state.is_recording = !sim_state.is_recording;
                if sim_state.is_recording {
                    sim_state.recording_frame = 0;
                    sim_state.play_mode = PlayMode::Playing;
                    sim_state.speed_multiplier = 1.0;
                    let _ = std::fs::remove_dir_all("video_frames");
                    std::fs::create_dir_all("video_frames").unwrap();
                    tracing::info!("Started recording frames to video_frames/");
                } else {
                    sim_state.play_mode = PlayMode::Paused;
                    tracing::info!("Encoding video_frames/ to output.mp4...");
                    std::thread::spawn(|| {
                        let status = std::process::Command::new("ffmpeg")
                            .args(&[
                                "-y",
                                "-framerate",
                                "20",
                                "-i",
                                "video_frames/frame_%05d.bmp",
                                "-c:v",
                                "libx264",
                                "-pix_fmt",
                                "yuv420p",
                                "output.mp4",
                            ])
                            .status();
                        match status {
                            Ok(s) if s.success() => {
                                tracing::info!("Successfully encoded output.mp4!");
                                let _ = std::fs::remove_dir_all("video_frames");
                            }
                            _ => tracing::error!("Failed to encode video (is ffmpeg installed?)"),
                        }
                    });
                }
            }

            ui.separator();

            // Speed multiplier
            ui.label("Speed:");
            let mut speed = sim_state.speed_multiplier;
            ui.add(
                egui::Slider::new(&mut speed, 0.125..=32.0)
                    .logarithmic(true)
                    .text("×"),
            );
            sim_state.speed_multiplier = speed;

            ui.separator();

            // Sim info
            ui.label(format!(
                "t={:.1}s  tick={}  tracks={}",
                sim_state.sim_time,
                sim_state.tick,
                0, // filled below
            ));
        });

        // Second row: render toggles
        ui.horizontal(|ui| {
            ui.checkbox(&mut render.show_ground_truth, "Ground Truth");
            ui.checkbox(&mut render.show_gates, "Gates");
            ui.checkbox(&mut render.show_associations, "Assoc.");
            ui.checkbox(&mut render.show_trails, "Trails");
            ui.checkbox(&mut render.show_components, "Components");
            ui.checkbox(&mut render.confirmed_only, "Confirmed only");
            ui.separator();
            ui.label("Scale:");
            ui.add(
                egui::Slider::new(&mut render.world_to_screen_scale, 0.0001..=0.02)
                    .logarithmic(true),
            );
        });
    });
}

/// Timeline slider panel (bottom).
pub fn ui_timeline_panel(
    mut contexts: EguiContexts,
    sim_state: Res<SimState>,
    tracker_state: Res<TrackerAppState>,
) {
    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };
    egui::TopBottomPanel::bottom("timeline").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Progress
            let progress = (sim_state.sim_time / sim_state.scenario.duration) as f32;
            let bar = egui::ProgressBar::new(progress).text(format!(
                "{:.1}s / {:.1}s",
                sim_state.sim_time, sim_state.scenario.duration
            ));
            ui.add(bar);

            ui.separator();

            // Track counts
            if let Some(output) = &tracker_state.last_output {
                let confirmed = output
                    .tracks
                    .iter()
                    .filter(|t| t.status == TrackStatus::Confirmed)
                    .count();
                let tentative = output
                    .tracks
                    .iter()
                    .filter(|t| t.status == TrackStatus::Tentative)
                    .count();
                ui.label(format!(
                    "✅ {confirmed} confirmed  🟡 {tentative} tentative"
                ));

                ui.separator();

                // Timing
                if !tracker_state.timing_history.is_empty() {
                    let avg: u64 = tracker_state.timing_history.iter().sum::<u64>()
                        / tracker_state.timing_history.len() as u64;
                    ui.label(format!("⏱ {avg}µs/batch"));
                }

                // Debug timing breakdown
                let d = &output.debug;
                ui.label(format!(
                    "pred={} gate={} assn={} upd={} mgmt={}µs",
                    d.timing_predict_us,
                    d.timing_gate_us,
                    d.timing_assign_us,
                    d.timing_update_us,
                    d.timing_manage_us,
                ));
            }
        });
    });
}

/// Track inspector panel (right side).
pub fn ui_track_inspector(
    mut contexts: EguiContexts,
    tracker_state: Res<TrackerAppState>,
    sim_state: Res<SimState>,
) {
    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };
    egui::SidePanel::right("inspector")
        .min_width(240.0)
        .show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();

            let output = match &tracker_state.last_output {
                Some(o) => o,
                None => {
                    ui.label("No output yet. Press ▶ Play.");
                    return;
                }
            };

            // Track list
            ui.collapsing(format!("Tracks ({})", output.tracks.len()), |ui| {
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for track in &output.tracks {
                            let status_icon = match track.status {
                                TrackStatus::Confirmed => "✅",
                                TrackStatus::Tentative => "🟡",
                                TrackStatus::Deleted => "❌",
                            };
                            ui.collapsing(
                                format!(
                                    "{status_icon} {} — hits={} miss={}",
                                    track.id, track.total_hits, track.misses
                                ),
                                |ui| {
                                    ui.label(format!(
                                        "pos:  ({:.0}, {:.0}, {:.0}) m",
                                        track.state[0], track.state[1], track.state[2]
                                    ));
                                    ui.label(format!(
                                        "vel:  ({:.1}, {:.1}, {:.1}) m/s",
                                        track.state[3], track.state[4], track.state[5]
                                    ));
                                    ui.label(format!(
                                        "P_diag: ({:.1}, {:.1}, {:.1})",
                                        track.cov[(0, 0)].sqrt(),
                                        track.cov[(1, 1)].sqrt(),
                                        track.cov[(2, 2)].sqrt()
                                    ));
                                    ui.label(format!(
                                        "age: {:.1}s",
                                        sim_state.sim_time - track.born_at
                                    ));
                                },
                            );
                        }
                    });
            });

            ui.separator();

            // Radar list
            ui.collapsing(
                format!("Radars ({})", sim_state.scenario.radars.len()),
                |ui| {
                    for radar in &sim_state.scenario.radars {
                        ui.collapsing(format!("{}", radar.id), |ui| {
                            ui.label(format!(
                                "pos: ({:.0}, {:.0})",
                                radar.params.position[0], radar.params.position[1]
                            ));
                            ui.label(format!("rate: {:.1} Hz", radar.params.refresh_rate));
                            ui.label(format!("P_D: {:.2}", radar.params.p_detection));
                            ui.label(format!("λ_clutter: {:.2e}", radar.params.lambda_clutter));
                            ui.label(format!(
                                "σ_range: {:.0}m  σ_az: {:.4}rad",
                                radar.params.range_noise_std, radar.params.azimuth_noise_std
                            ));
                            ui.label(format!(
                                "Injected bias (True): dx={:.0}m dy={:.0}m dθ={:.3}rad dt0={:.3}s",
                                radar.bias.dx, radar.bias.dy, radar.bias.dtheta, radar.bias.dt0
                            ));
                            
                            // Visualize the Tracker's Bias Compensation (Phase C)
                            if let Some(est) = output.debug.sensor_biases.get(&radar.id) {
                                ui.label(egui::RichText::new(format!(
                                    "Applied Bias: dx={:.0}m dy={:.0}m br={:.1}m ba={:.3}rad dθ={:.3}rad dt0={:.3}s",
                                    est.spatial.dx, est.spatial.dy, est.spatial.br, est.spatial.ba, est.spatial.dtheta, est.temporal.dt0
                                )).color(egui::Color32::from_rgb(100, 200, 255)));
                            } else {
                                ui.label(egui::RichText::new("Estimated bias: (no data yet)").color(egui::Color32::DARK_GRAY));
                            }
                        });
                    }
                },
            );

            ui.separator();

            // Innovation residuals
            ui.collapsing("Innovations", |ui| {
                for (track_id, innov) in &output.debug.innovations {
                    ui.label(format!(
                        "{}: [{:.1}]",
                        track_id,
                        innov
                            .iter()
                            .map(|v| format!("{v:.1}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                if output.debug.innovations.is_empty() {
                    ui.label("(none)");
                }
            });

            // Connected components
            ui.collapsing(
                format!("Components ({})", output.debug.components.len()),
                |ui| {
                    for (i, comp) in output.debug.components.iter().enumerate() {
                        ui.label(format!("C{i}: {} tracks", comp.len()));
                    }
                    if output.debug.components.is_empty() {
                        ui.label("(none — all singleton)");
                    }
                },
            );

            // Keyboard shortcuts legend
            ui.separator();
            ui.label(egui::RichText::new("Shortcuts").small().italics());
            ui.label(egui::RichText::new("Space=play/pause  .=step  R=reset  +/-=speed").small());
        });
}

/// Keyboard shortcuts help overlay (top-right corner, translucent).
pub fn ui_keyboard_help(mut contexts: EguiContexts) {
    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };
    egui::Area::new("help_area".into())
        .anchor(egui::Align2::LEFT_CENTER, [10.0, 0.0])
        .show(ctx, |_ui| {
            // intentionally empty — shortcuts are in inspector panel
        });
}

/// Overlay to render Hungarian assignment costs next to association lines.
pub fn ui_cost_overlay(
    mut contexts: EguiContexts,
    tracker_state: Res<TrackerAppState>,
    render: Res<RenderSettings>,
    window_query: Query<&Window, bevy::prelude::With<bevy::window::PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
) {
    if !render.show_associations || !render.show_components {
        return;
    }

    let output = match &tracker_state.last_output {
        Some(o) => o,
        None => return,
    };

    let Ok(window) = window_query.get_single() else {
        return;
    };
    let Ok((_camera, _camera_transform)) = camera_query.get_single() else {
        return;
    };

    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };

    egui::Area::new("cost_overlay".into())
        .fixed_pos(egui::pos2(0.0, 0.0))
        .interactable(false)
        .show(ctx, |ui| {
            let screen_size = window.resolution.physical_size();
            // Allocate full-screen rect to paint text anywhere without layout clips
            ui.allocate_rect(
                egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(screen_size.x as f32, screen_size.y as f32),
                ),
                egui::Sense::hover(),
            );
            let painter = ui.painter();

            use std::collections::HashMap;
            use tracker_core::types::MeasurementId;
            let meas_map: HashMap<MeasurementId, (f64, f64)> = tracker_state
                .current_measurements
                .iter()
                .map(|m| {
                    let c = m.to_cartesian_2d();
                    (m.id, (c[0], c[1]))
                })
                .collect();

            let track_map: HashMap<_, (f64, f64)> = output
                .tracks
                .iter()
                .map(|t| (t.id, t.position_2d()))
                .collect();

            let cost_map: HashMap<(tracker_core::types::TrackId, MeasurementId), f64> = output
                .debug
                .gate_edges
                .iter()
                .map(|(t, m, c)| ((*t, *m), *c))
                .collect();

            // Only show costs if there aren't too many, to avoid screen clutter / perf drop
            if output.debug.assignments.len() > 100 {
                painter.text(
                    egui::pos2(10.0, 80.0),
                    egui::Align2::LEFT_TOP,
                    format!("Cost overlay hidden (>100 assignments)"),
                    egui::FontId::proportional(14.0),
                    egui::Color32::from_rgb(255, 100, 100),
                );
                return;
            }

            for (track_id, meas_id) in &output.debug.assignments {
                if let (Some(&(tx, ty)), Some(&(mx, my))) =
                    (track_map.get(track_id), meas_map.get(meas_id))
                {
                    // Convert world coordinates to viewport (screen) coordinates.
                    let tp = egui::pos2(
                        window.width() / 2.0 + (tx as f32 * render.world_to_screen_scale),
                        window.height() / 2.0 - (ty as f32 * render.world_to_screen_scale),
                    );
                    let mp = egui::pos2(
                        window.width() / 2.0 + (mx as f32 * render.world_to_screen_scale),
                        window.height() / 2.0 - (my as f32 * render.world_to_screen_scale),
                    );

                    let midpoint = egui::pos2((tp.x + mp.x) / 2.0, (tp.y + mp.y) / 2.0);
                    let cost = cost_map.get(&(*track_id, *meas_id)).copied().unwrap_or(0.0);

                    painter.text(
                        midpoint,
                        egui::Align2::CENTER_CENTER,
                        format!("{:.1}", cost),
                        egui::FontId::proportional(12.0),
                        egui::Color32::from_rgb(255, 255, 100),
                    );
                }
            }
        });
}

/// Statistics UI shown after the simulation has finished (or paused at the end).
pub fn ui_evaluation_panel(
    mut contexts: EguiContexts,
    sim_state: Res<SimState>,
    tracker_state: Res<TrackerAppState>,
) {
    if sim_state.sim_time < sim_state.scenario.duration {
        return; // Only show at the end of the simulation
    }

    let ctx = match contexts.try_ctx_mut() {
        Some(c) => c,
        None => return,
    };

    let mut open = true;
    egui::Window::new("Analysis & Statistics")
        .open(&mut open)
        .resizable(true)
        .default_width(600.0)
        .show(ctx, |ui| {
            ui.heading("Track Evaluation Diagnostics");
            ui.label("Per-track RMSE and Lifespan against Ground Truth Targets.");
            ui.separator();

            // Calculate total target durations
            use std::collections::HashMap;
            let mut target_lifespans = HashMap::new();
            for target in &sim_state.scenario.targets {
                let start = target.appear_at.unwrap_or(0.0);
                let end = target.disappear_at.unwrap_or(sim_state.scenario.duration);
                target_lifespans.insert(target.id, end - start);
            }

            egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                ui.style_mut().spacing.item_spacing = egui::vec2(10.0, 5.0);
                
                // Table header
                egui::Grid::new("eval_grid")
                    .striped(true)
                    .num_columns(5)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Track ID").strong());
                        ui.label(egui::RichText::new("Target ID").strong());
                        ui.label(egui::RichText::new("RMSE (2D)").strong());
                        ui.label(egui::RichText::new("Track Life").strong());
                        ui.label(egui::RichText::new("Continuity %").strong());
                        ui.end_row();

                        let tracks: Vec<_> = tracker_state.all_track_metrics.iter().collect();

                        // Pre-compute statistics for sorting
                        let mut track_data: Vec<_> = tracks.into_iter().map(|(track_id, metrics)| {
                            let track_life = (metrics.end_time - metrics.start_time).max(0.0);
                            let rmse = if metrics.count > 0 {
                                (metrics.sum_sq_err / metrics.count as f64).sqrt()
                            } else {
                                f64::NAN
                            };
                            let (target_id_str, continuity) = match metrics.target_id {
                                Some(tid) => {
                                    let tgt_life = target_lifespans.get(&tid).copied().unwrap_or(sim_state.scenario.duration);
                                    let cont = (track_life / tgt_life.max(0.001)) * 100.0;
                                    (format!("Target {}", tid), cont)
                                }
                                None => ("None".to_string(), 0.0),
                            };
                            (track_id, metrics, track_life, rmse, target_id_str, continuity)
                        }).collect();

                        // Sort by continuity descending
                        track_data.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));

                        let mut avg_rmse = 0.0;
                        let mut avg_continuity = 0.0;
                        let mut valid_stats_count = 0;

                        for (track_id, metrics, track_life, rmse, target_id_str, continuity) in track_data {
                            if metrics.count > 0 {
                                avg_rmse += rmse;
                                avg_continuity += continuity;
                                valid_stats_count += 1;
                            }

                            ui.label(format!("T{}", track_id.0));
                            ui.label(target_id_str);
                            if rmse.is_nan() {
                                ui.label("-");
                            } else {
                                ui.label(format!("{:.2} m", rmse));
                            }
                            ui.label(format!("{:.1}s", track_life));
                            ui.label(format!("{:.1}%", continuity.min(100.0)));
                            ui.end_row();
                        }

                        if valid_stats_count > 0 {
                            avg_rmse /= valid_stats_count as f64;
                            avg_continuity /= valid_stats_count as f64;
                        }

                        // Summary row
                        ui.separator(); ui.separator(); ui.separator(); ui.separator(); ui.separator(); ui.end_row();
                        ui.label(egui::RichText::new("AVERAGES").strong());
                        ui.label("-");
                        ui.label(egui::RichText::new(format!("{:.2} m", avg_rmse)).strong().color(egui::Color32::LIGHT_GREEN));
                        ui.label("-");
                        ui.label(egui::RichText::new(format!("{:.1}%", avg_continuity.min(100.0))).strong().color(egui::Color32::LIGHT_GREEN));
                        ui.end_row();
                    });
            });
        });
}
