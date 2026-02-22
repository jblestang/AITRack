//! `aittrack` CLI: batch evaluation, scenario runs, replay import/export.

use anyhow::Result;
use clap::{Parser, Subcommand};
use sim::radar_sim::RadarSimulator;
use sim::replay::{save_replay, GroundTruthFrame, ReplayLog, TargetState};
use sim::scenarios::{Scenario, ScenarioKind};
use std::path::PathBuf;
use tracker_core::metrics::TrackingMetrics;
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
    },
    /// Load and replay a previously recorded scenario log.
    Replay {
        /// Path to replay JSON file
        input: PathBuf,
        /// Output metrics to a JSON file
        #[arg(long)]
        output: Option<PathBuf>,
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
        } => {
            run_scenario(scenario, seed, output.as_deref(), save_path.as_deref())?;
        }
        Commands::Replay { input, output } => {
            run_replay(&input, output.as_deref())?;
        }
    }

    Ok(())
}

fn run_scenario(
    kind: ScenarioKind,
    seed: u64,
    output_path: Option<&std::path::Path>,
    replay_path: Option<&std::path::Path>,
) -> Result<()> {
    let mut scenario = Scenario::build(kind.clone(), seed);
    let mut radar_sim = RadarSimulator::new(scenario.radars.clone(), seed);
    let mut pipeline = Pipeline::new(PipelineConfig::default());

    let dt = scenario.sim_dt;
    let duration = scenario.duration;
    let mut sim_time = 0.0f64;
    let mut meas_id_counter = 0u64;
    let mut all_batches: Vec<RadarBatch> = Vec::new();
    let mut gt_frames: Vec<GroundTruthFrame> = Vec::new();

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
            pipeline.process_batch(&batch);
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
            .count(),
    );

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

    if let Some(opath) = output_path {
        let json = serde_json::json!({
            "scenario": log.scenario_name,
            "seed": log.seed,
            "elapsed_s": elapsed.as_secs_f64(),
            "final_tracks": pipeline.tracks.len(),
        });
        std::fs::write(opath, serde_json::to_string_pretty(&json)?)?;
    }

    Ok(())
}
