# AITRack — Architecture Documentation

## Overview

AITRack is a **multi-sensor multi-target tracker** implemented in Rust using Bevy for the debug UI.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     Cargo Workspace: AITRack/                           │
├──────────────┬──────────────┬──────────┬─────────────────┬────────────┤
│tracker_core  │sensor_models │   sim    │ bevy_ui_debug   │    cli     │
│ (pure Rust)  │ (pure Rust)  │(pure Rust│  (Bevy app)    │  (binary)  │
└──────────────┴──────────────┴──────────┴─────────────────┴────────────┘
```

## Data Flow

```
sim::Scenario
    ↓ generates every N ms (async per radar)
sim::RadarBatch { sensor_id, timestamp, Vec<Measurement> }
    ↓ passed to
tracker_core::Pipeline::process_batch()
    ├── 1. Bias correction (Phase C: spatial + temporal)
    ├── 2. Predict all tracks to batch_time (KF predict)
    ├── 3. Mahalanobis gate check (track × meas pairs)
    ├── 4. Build sparse bipartite graph
    ├── 5. Partition into connected components (union-find O(αn))
    ├── 6. Hungarian algorithm per component (O(n³))
    ├── 7. KF update for matched pairs (Joseph form)
    ├── 8. Register hits/misses → M-of-N confirmation/deletion
    ├── 9. Birth tentative tracks for unmatched measurements
    └──10. Prune deleted tracks
         ↓ returns
PipelineOutput { tracks, debug_data, timings }
    ↓ consumed by
bevy_ui_debug   (render + inspect)
cli             (metrics + export)
```

## Module Structure

### `tracker_core`

| Module | Responsibility |
|--------|---------------|
| `types` | `StateVec`, `StateCov`, `TrackId`, `SensorId`, `Measurement`, `RadarBatch` |
| `track` | `Track` struct: state, cov, status, M-of-N counters, history ring-buffer |
| `kf` | `KalmanFilter` trait + `CvKalmanFilter` (CV model, Joseph form covariance update) |
| `gating` | Mahalanobis gate, χ² thresholds, `GatingEllipse` for UI |
| `association` | Bipartite graph, union-find component partition, O(n³) Hungarian solver |
| `track_manager` | Birth/confirm/delete lifecycle policy |
| `pipeline` | 10-step pipeline orchestrator with debug data collection |
| `bias` | Bias types + estimator stub (Phase C) |
| `metrics` | RMSE, precision, recall, ID-switch accumulation |

### `sensor_models`

| Module | Responsibility |
|--------|---------------|
| `radar` | `RadarParams`: position, FoV, P_D, λ_clutter, noise stds |
| `observation` | `CartesianXY` (linear H), `PolarObservation` (EKF Jacobian) |

### `sim`

| Module | Responsibility |
|--------|---------------|
| `target` | `Target` with CV/CTRV/CA/waypoint motion models |
| `radar_sim` | `RadarSimulator`: async batch generation, noise, clutter, bias injection |
| `scenarios` | 4 named scenarios: `Simple`, `DenseCrossing`, `Stress`, `BiasCalibration` |
| `replay` | JSON serialization of `ReplayLog` (batches + ground truth) |

### `bevy_ui_debug`

| Module | Responsibility |
|--------|---------------|
| `app` | Bevy `App` setup, plugin registration, system schedule |
| `resources` | `SimState`, `TrackerAppState`, `RenderSettings`, events |
| `systems` | Keyboard control, simulation advance, reset |
| `render` | Gizmo rendering: measurements, tracks, gates, associations, radars |
| `ui` | egui panels: control bar, timeline, track inspector, radar inspector |

## Key Algorithms

### Kalman Filter (CV model)
```
F = I₆ + dt·[[0₃ I₃]; [0₃ 0₃]]   # transition  
Q = q·block[[dt⁴/4·I₃, dt³/2·I₃]; [dt³/2·I₃, dt²·I₃]]  # process noise
ν = z − H·x̂                         # innovation
S = H·P·Hᵀ + R                      # innovation cov
K = P·Hᵀ·S⁻¹                        # Kalman gain
x' = x + K·ν                        # updated state
P' = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ    # Joseph form (symmetric)
```

### Mahalanobis Gating
```
d²(z, track) = νᵀ·S⁻¹·ν  < χ²(0.99, dim)
```
χ²(0.99, 2) = 9.21 for 2D cartesian measurements.

### Connected Components (union-find)
Path-halving union-find, O(α·n) amortised. Each edge (track_i, meas_j) unions nodes i and n_tracks + j. Components are solved independently.

### Hungarian Assignment
Jonker-Volgenant O(n³) on square cost matrix. Dummy assignments (cost = `dummy_cost`) handle missed detections and false alarms naturally.

### Track Management (M-of-N)
- Birth: tentative track on each unmatched measurement
- Confirm: `hits ≥ M` (default M=3)
- Delete confirmed: `misses > miss_limit_confirmed` (default 5)
- Delete tentative: `misses > 1` (fast pruning)

## Scenarios

| Name | Targets | Radars | Duration | Notes |
|------|---------|--------|----------|-------|
| `simple` | 5 | 2 | 120s | Straight + CTRV, low clutter |
| `dense_crossing` | 50 | 3 | 300s | All heading center, high clutter |
| `stress` | 500 | 4 | 60s | Random positions/velocities/motion |
| `bias_calibration` | 20 | 2 | 180s | dx=±200m, dθ=±0.02rad, dt0=±0.3s |

## Complexity Summary

| Step | Complexity |
|------|-----------|
| Predict all tracks | O(N_tracks) |
| Gate check | O(N_tracks × N_meas) |
| Component partition | O(E·α) where E = edges past gate |
| Hungarian per component | O(n³) per component |
| KF update | O(n_matched) |
| Total (sparse) | O(N·M + Σ_comp nᵢ³) |

For sparse association graphs (typical), Σnᵢ³ ≪ N³.

## Extension Guide

### Adding a new motion model
1. Add a variant to `MotionSpec` in `sim/src/target.rs`
2. Implement propagation in `Target::step()`
3. For tracker: add a new `KalmanFilter` impl in `tracker_core/src/kf.rs`
4. Wire into IMM in Phase D

### Adding a new radar type
1. Add fields to `RadarParams` in `sensor_models/src/radar.rs`
2. Implement `ObservationModel` trait in `sensor_models/src/observation.rs`
3. Update `RadarSimulator::generate_batches()` if measurement format changes

### Adding a new scenario
1. Add variant to `ScenarioKind` enum in `sim/src/scenarios.rs`
2. Implement the corresponding builder function
3. Add to `Scenario::build()` match

### Enabling JPDA on dense components (Phase D)
1. Detect component size > threshold in `pipeline.rs`
2. Route to `jpda_solve()` instead of `hungarian_solve()`
3. JPDA marginalises over all feasible assignments → soft updates
