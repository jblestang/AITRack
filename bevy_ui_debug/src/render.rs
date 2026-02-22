//! Gizmo-based 2D rendering of tracks, measurements, gates, and trails.

use crate::resources::{RenderSettings, SimState, TrackerAppState};
use bevy::prelude::*;
use tracker_core::track::TrackStatus;
use tracker_core::types::SensorId;

// Radar colors (up to 8 radars)
const SENSOR_COLORS: [Color; 8] = [
    Color::srgb(0.2, 0.6, 1.0),  // blue
    Color::srgb(1.0, 0.35, 0.2), // red-orange
    Color::srgb(0.2, 1.0, 0.4),  // green
    Color::srgb(1.0, 0.9, 0.1),  // yellow
    Color::srgb(0.9, 0.2, 1.0),  // purple
    Color::srgb(0.1, 0.9, 0.9),  // cyan
    Color::srgb(1.0, 0.5, 0.0),  // orange
    Color::srgb(1.0, 1.0, 1.0),  // white
];

fn sensor_color(id: &SensorId) -> Color {
    SENSOR_COLORS[id.0 as usize % SENSOR_COLORS.len()]
}

/// World position (meters) → Bevy screen position (pixels)
fn world_to_screen(x: f64, y: f64, scale: f32) -> Vec2 {
    Vec2::new(x as f32 * scale, y as f32 * scale)
}

/// Render raw measurements as small circles, colored by sensor.
pub fn render_measurements_system(
    mut gizmos: Gizmos,
    tracker_state: Res<TrackerAppState>,
    render: Res<RenderSettings>,
) {
    for meas in &tracker_state.current_measurements {
        let color = sensor_color(&meas.sensor_id);
        let cart = meas.to_cartesian_2d();
        let pos = world_to_screen(cart[0], cart[1], render.world_to_screen_scale);
        gizmos.circle_2d(pos, 3.0, color);
    }
}

/// Render track positions, velocity arrows, labels, and trails.
pub fn render_tracks_system(
    mut gizmos: Gizmos,
    tracker_state: Res<TrackerAppState>,
    render: Res<RenderSettings>,
) {
    let output = match &tracker_state.last_output {
        Some(o) => o,
        None => return,
    };

    for track in &output.tracks {
        if render.confirmed_only && track.status != TrackStatus::Confirmed {
            continue;
        }

        let (tx, ty) = track.position_2d();
        let pos = world_to_screen(tx, ty, render.world_to_screen_scale);

        // Color by status or component
        let mut component_color = None;
        if render.show_components {
            for (i, comp) in output.debug.components.iter().enumerate() {
                if comp.contains(&track.id) {
                    // Generate a distinct color based on the component index
                    // Hue shift by golden angle approximation (137.5 degrees)
                    let hue = (i as f32 * 137.5) % 360.0;
                    component_color = Some(Color::hsl(hue, 0.8, 0.6));
                    break;
                }
            }
        }

        let color = component_color.unwrap_or_else(|| match track.status {
            TrackStatus::Confirmed => Color::srgb(0.0, 1.0, 0.3),
            TrackStatus::Tentative => Color::srgb(1.0, 0.8, 0.0),
            TrackStatus::Deleted => Color::srgb(0.5, 0.5, 0.5),
        });

        // Cross marker at track position
        let sz = 6.0;
        gizmos.line_2d(pos - Vec2::new(sz, 0.), pos + Vec2::new(sz, 0.), color);
        gizmos.line_2d(pos - Vec2::new(0., sz), pos + Vec2::new(0., sz), color);

        // Velocity arrow
        let (vx, vy) = track.velocity_2d();
        let vel_scale = 3.0; // seconds of prediction to show
        let arrow_end = world_to_screen(
            tx + vx * vel_scale,
            ty + vy * vel_scale,
            render.world_to_screen_scale,
        );
        gizmos.line_2d(pos, arrow_end, color.with_alpha(0.5));

        // Trail
        if render.show_trails {
            let v: Vec<Vec2> = track
                .history
                .iter()
                .map(|s| world_to_screen(s[0], s[1], render.world_to_screen_scale))
                .collect();
            for w in v.windows(2) {
                gizmos.line_2d(w[0], w[1], color.with_alpha(0.25));
            }
        }
    }
}

/// Render gating ellipses (approximated as circles for Phase A).
pub fn render_gates_system(
    mut gizmos: Gizmos,
    tracker_state: Res<TrackerAppState>,
    render: Res<RenderSettings>,
) {
    if !render.show_gates {
        return;
    }
    let output = match &tracker_state.last_output {
        Some(o) => o,
        None => return,
    };
    for ellipse in &output.debug.gate_ellipses {
        let center = world_to_screen(
            ellipse.center.0,
            ellipse.center.1,
            render.world_to_screen_scale,
        );
        let rx = ellipse.semi_x as f32 * render.world_to_screen_scale;
        let ry = ellipse.semi_y as f32 * render.world_to_screen_scale;
        // Draw as approximated circle using the larger semi-axis
        let r = rx.max(ry).max(5.0);
        gizmos.circle_2d(center, r, Color::srgba(0.5, 0.5, 1.0, 0.3));
    }
}

/// Render association edges (lines: track → measurement).
pub fn render_associations_system(
    mut gizmos: Gizmos,
    tracker_state: Res<TrackerAppState>,
    render: Res<RenderSettings>,
) {
    if !render.show_associations {
        return;
    }
    let output = match &tracker_state.last_output {
        Some(o) => o,
        None => return,
    };

    // Build measurement position lookup
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

    // Build track position lookup
    let track_map: HashMap<_, (f64, f64)> = output
        .tracks
        .iter()
        .map(|t| (t.id, t.position_2d()))
        .collect();

    for (track_id, meas_id) in &output.debug.assignments {
        if let (Some(&(tx, ty)), Some(&(mx, my))) = (track_map.get(track_id), meas_map.get(meas_id))
        {
            let tp = world_to_screen(tx, ty, render.world_to_screen_scale);
            let mp = world_to_screen(mx, my, render.world_to_screen_scale);
            gizmos.line_2d(tp, mp, Color::srgba(1.0, 1.0, 0.0, 0.6));
        }
    }
}

/// Render radar positions.
pub fn render_radars_system(
    mut gizmos: Gizmos,
    sim_state: Res<SimState>,
    render: Res<RenderSettings>,
) {
    for radar in &sim_state.scenario.radars {
        let pos = world_to_screen(
            radar.params.position[0],
            radar.params.position[1],
            render.world_to_screen_scale,
        );
        let color = sensor_color(&radar.id);
        gizmos.circle_2d(pos, 10.0, color);
        // Draw FoV indicator (stubbed as outer circle fraction)
        let r = radar.params.max_range as f32 * render.world_to_screen_scale;
        gizmos.circle_2d(pos, r.min(600.0), color.with_alpha(0.08));
    }
}
