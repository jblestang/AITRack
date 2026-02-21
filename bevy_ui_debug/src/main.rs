//! bevy_ui_debug binary entry point.
//!
//! Usage:
//!   cargo run --release --package bevy_ui_debug -- --scenario simple --seed 42
//!   cargo run --release --package bevy_ui_debug -- --scenario dense-crossing
//!   cargo run --release --package bevy_ui_debug -- --scenario stress
//!   cargo run --release --package bevy_ui_debug -- --scenario bias-calibration

use clap::Parser;
use sim::scenarios::ScenarioKind;

#[derive(Parser)]
#[command(name = "bevy_ui_debug", about = "AITRack interactive debug UI")]
struct Args {
    /// Scenario to run
    #[arg(long, value_enum, default_value = "simple")]
    scenario: ScenarioKind,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    bevy_ui_debug::run_debug_app(args.scenario, args.seed);
}
