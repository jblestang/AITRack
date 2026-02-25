//! `aittrack` CLI: batch evaluation, scenario runs, replay import/export.

use anyhow::Result;
use clap::{Parser, Subcommand};
use sim::radar_sim::RadarSimulator;
use sim::replay::{save_replay, GroundTruthFrame, ReplayLog, TargetState};
use sim::scenarios::{Scenario, ScenarioKind};
use std::path::PathBuf;
// use tracker_core::metrics::TrackingMetrics;
use tracker_core::pipeline::{Pipeline, PipelineConfig};
use tracker_core::types::RadarBatch;

#[derive(Parser)]
#[command(name = "aittrack", about = "Multi-sensor tracker CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a named scenario in batch mode and output metrics.
    RunScenario {
        #[arg(value_enum)]
        scenario: ScenarioKind,
        /// Random seed for reproducibility
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output metrics to a JSON file
        #[arg(long)]
        output: Option<PathBuf>,
        /// Also save the full replay log
        #[arg(long)]
        save_replay: Option<PathBuf>,
        /// Enable Joint Probabilistic Data Association (JPDA) instead of Hungarian.
        #[arg(long)]
        jpda: bool,
    },
    /// Load and replay a previously recorded scenario log.
    Replay {
        /// Path to replay JSON file
        input: PathBuf,
        /// Output metrics to a JSON file
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run all golden tests and compare results.
    VerifyGolden {
        /// Directory containing golden JSON files
        #[arg(long, default_value = "tests/golden")]
        dir: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::RunScenario {
            scenario,
            seed,
            output,
            save_replay: save_path,
            jpda,
        } => {
            run_scenario(scenario, seed, output.as_deref(), save_path.as_deref(), jpda)?;
        }
        Commands::Replay { input, output } => {
            run_replay(&input, output.as_deref())?;
        }
        Commands::VerifyGolden { dir } => {
            verify_golden(&dir)?;
        }
    }

    Ok(())
}

fn run_scenario(
    kind: ScenarioKind,
    seed: u64,
    output_path: Option<&std::path::Path>,
    replay_path: Option<&std::path::Path>,
    use_jpda: bool,
) -> Result<()> {
    let mut scenario = Scenario::build(kind.clone(), seed);
    let mut radar_sim = RadarSimulator::new(scenario.radars.clone(), seed);
    let mut config = PipelineConfig::default();
    config.use_jpda = use_jpda;
    let mut pipeline = Pipeline::new(config);

    let dt = scenario.sim_dt;
    let duration = scenario.duration;
    let mut sim_time = 0.0f64;
    let mut meas_id_counter = 0u64;
    let mut all_batches: Vec<RadarBatch> = Vec::new();
    let mut gt_frames: Vec<GroundTruthFrame> = Vec::new();

    // Track stats over time
    use std::collections::HashMap;
    #[derive(Default)]
    struct CliTrackStat {
        target_id: Option<u64>,
        start: f64,
        end: f64,
        count: u64,
    }
    let mut track_stats: HashMap<tracker_core::types::TrackId, CliTrackStat> = HashMap::new();

    println!(
        "Running scenario '{}' (seed={}, duration={:.0}s)...",
        scenario.name, seed, duration
    );

    let start = std::time::Instant::now();
    let mut total_batches = 0;

    while sim_time < duration {
        // Step targets
        for target in &mut scenario.targets {
            target.step(sim_time, dt);
        }
        sim_time += dt;

        // Record GT
        gt_frames.push(GroundTruthFrame {
            time: sim_time,
            targets: scenario
                .targets
                .iter()
                .filter(|t| t.is_active(sim_time))
                .map(|t| TargetState {
                    id: t.id,
                    state: t.state,
                })
                .collect(),
        });

        // Generate + process batches
        let batches = radar_sim.generate_batches(&scenario.targets, sim_time, &mut meas_id_counter);
        for batch in batches {
            total_batches += 1;
            all_batches.push(batch.clone());
            let out = pipeline.process_batch(&batch);
            
            // Capture online metrics
            let active_targets: Vec<_> = scenario.targets.iter().filter(|t| t.is_active(sim_time)).collect();
            for track in &out.tracks {
                if track.status == tracker_core::track::TrackStatus::Confirmed {
                    let mut min_dist_sq = f64::MAX;
                    let mut best_target = None;
                    for target in &active_targets {
                        let dx = track.state[0] - target.state[0];
                        let dy = track.state[1] - target.state[1];
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                            best_target = Some(target.id);
                        }
                    }
                    if min_dist_sq > 2000.0 * 2000.0 { best_target = None; }
                    
                    let stat = track_stats.entry(track.id).or_insert_with(|| CliTrackStat {
                        target_id: best_target,
                        start: track.born_at,
                        end: sim_time,
                        count: 0,
                    });
                    if min_dist_sq < 500.0 * 500.0 {
                        if stat.target_id.is_some() && stat.target_id != best_target {
                            println!("[Swap] t={:.2}, Track {} swapped to Target {:?}", sim_time, track.id.0, best_target);
                        }
                        if stat.target_id.is_none() {
                            stat.target_id = best_target;
                        }
                    }
                    stat.end = sim_time;
                    stat.count += 1;
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "Done: {} ticks, {} batches, {} tracks alive, elapsed={:.2}s",
        (duration / dt) as u64,
        total_batches,
        pipeline.tracks.len(),
        elapsed.as_secs_f64(),
    );
    println!(
        "Tracks: {} confirmed, {} tentative",
        pipeline
            .tracks
            .iter()
            .filter(|t| t.status == tracker_core::track::TrackStatus::Confirmed)
            .count(),
        pipeline
            .tracks
            .iter()
            .filter(|t| t.status == tracker_core::track::TrackStatus::Tentative)
            .count()
    );

    println!("--- Continuity Report ---");
    let mut stats: Vec<_> = track_stats.iter().collect();
    stats.sort_by_key(|(id, _)| **id);
    for (id, stat) in stats {
        let lifespan = stat.end - stat.start;
        let target_duration = duration; // approx for simplicity
        let cont_pct = (lifespan / target_duration) * 100.0;
        let tid_str = stat.target_id.map(|id| id.to_string()).unwrap_or_else(|| "?".to_string());
        println!("Track {} -> Target {} | Lifespan {:.1}s -> {:.1}s | {:.1}% Continuity", 
                 id, tid_str, stat.start, stat.end, cont_pct);
        if cont_pct < 99.0 {
            println!("  => Drop detected! Track died at {:.1}s", stat.end);
        }
    }
    println!("-------------------------");
    
    println!("--- Sensor Bias Estimates ---");
    let mut bias_json = HashMap::new();
    let mut sensor_ids: Vec<_> = pipeline.bias_estimator.sensor_states.keys().cloned().collect();
    sensor_ids.sort();
    for id in sensor_ids {
        if let Some(state) = pipeline.bias_estimator.sensor_states.get(&id) {
            println!("Sensor {}: dx={:.1}m, dy={:.1}m, dtheta={:.3}rad, dt0={:.3}s",
                id.0, state.spatial.dx, state.spatial.dy, state.spatial.dtheta, state.temporal.dt0);
            
            bias_json.insert(id.0.to_string(), serde_json::json!({
                "dx": state.spatial.dx,
                "dy": state.spatial.dy,
                "dtheta": state.spatial.dtheta,
                "dt0": state.temporal.dt0,
            }));
        }
    }
    println!("-------------------------");

    // Save replay if requested
    if let Some(rpath) = replay_path {
        let log = ReplayLog {
            scenario_name: scenario.name.clone(),
            seed,
            sim_dt: dt,
            duration,
            batches: all_batches,
            ground_truth: gt_frames,
        };
        save_replay(&log, rpath)?;
        println!("Replay saved to {}", rpath.display());
    }

    // Output metrics
    if let Some(opath) = output_path {
        let json = serde_json::json!({
            "scenario": scenario.name,
            "seed": seed,
            "elapsed_s": elapsed.as_secs_f64(),
            "total_batches": total_batches,
            "final_tracks": pipeline.tracks.len(),
            "sensor_biases": bias_json,
        });
        std::fs::write(opath, serde_json::to_string_pretty(&json)?)?;
        println!("Metrics saved to {}", opath.display());
    }

    Ok(())
}

fn run_replay(input: &std::path::Path, output_path: Option<&std::path::Path>) -> Result<()> {
    let log = sim::replay::load_replay(input)?;
    println!(
        "Replaying '{}' ({} batches)...",
        log.scenario_name,
        log.batches.len()
    );

    let mut pipeline = Pipeline::new(PipelineConfig::default());
    let start = std::time::Instant::now();

    for batch in &log.batches {
        pipeline.process_batch(batch);
    }

    let elapsed = start.elapsed();
    println!(
        "Replay done: {} tracks alive, elapsed={:.2}s",
        pipeline.tracks.len(),
        elapsed.as_secs_f64()
    );

    println!("--- Sensor Bias Estimates ---");
    let mut bias_json = std::collections::HashMap::new();
    let mut sensor_ids: Vec<_> = pipeline.bias_estimator.sensor_states.keys().cloned().collect();
    sensor_ids.sort();
    for id in sensor_ids {
        if let Some(state) = pipeline.bias_estimator.sensor_states.get(&id) {
            println!("Sensor {}: dx={:.1}m, dy={:.1}m, dtheta={:.3}rad, dt0={:.3}s",
                id.0, state.spatial.dx, state.spatial.dy, state.spatial.dtheta, state.temporal.dt0);
            
            bias_json.insert(id.0.to_string(), serde_json::json!({
                "dx": state.spatial.dx,
                "dy": state.spatial.dy,
                "dtheta": state.spatial.dtheta,
                "dt0": state.temporal.dt0,
            }));
        }
    }
    println!("-------------------------");

    if let Some(opath) = output_path {
        let json = serde_json::json!({
            "scenario": log.scenario_name,
            "seed": log.seed,
            "elapsed_s": elapsed.as_secs_f64(),
            "final_tracks": pipeline.tracks.len(),
            "sensor_biases": bias_json,
        });
        std::fs::write(opath, serde_json::to_string_pretty(&json)?)?;
    }

    Ok(())
}

fn verify_golden(dir: &std::path::Path) -> Result<()> {
    use std::fs;
    let entries = fs::read_dir(dir)?;
    let mut failures = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "json") {
            let content = fs::read_to_string(&path)?;
            let baseline: serde_json::Value = serde_json::from_str(&content)?;
            
            let scenario_name = baseline["scenario"].as_str().unwrap();
            let seed = baseline["seed"].as_u64().unwrap();
            
            println!("Verifying Golden: {} (seed={})...", scenario_name, seed);
            
            // Map string name back to ScenarioKind
            let kind = match scenario_name {
                "Simple" => ScenarioKind::Simple,
                "Dense Crossing" => ScenarioKind::DenseCrossing,
                "Stress" => ScenarioKind::Stress,
                "Bias Calibration" => ScenarioKind::BiasCalibration,
                _ => continue,
            };

            // Run scenario and capture results
            let mut scenario = Scenario::build(kind, seed);
            let mut radar_sim = RadarSimulator::new(scenario.radars.clone(), seed);
            let mut pipeline = Pipeline::new(PipelineConfig::default());
            
            let mut sim_time = 0.0;
            let mut meas_id_counter = 0;
            while sim_time < scenario.duration {
                for target in &mut scenario.targets { target.step(sim_time, scenario.sim_dt); }
                sim_time += scenario.sim_dt;
                let batches = radar_sim.generate_batches(&scenario.targets, sim_time, &mut meas_id_counter);
                for batch in batches { pipeline.process_batch(&batch); }
            }
            
            let actual_tracks = pipeline.tracks.len();
            let expected_tracks = baseline["final_tracks"].as_u64().unwrap() as usize;
            
            if actual_tracks == expected_tracks {
                println!("  [PASS] final_tracks matched ({}).", actual_tracks);
            } else {
                println!("  [FAIL] final_tracks mismatch! Expected {}, got {}.", expected_tracks, actual_tracks);
                failures += 1;
            }
        }
    }

    if failures > 0 {
        anyhow::bail!("{} golden tests failed!", failures);
    } else {
        println!("All golden tests passed.");
    }

    Ok(())
}
