//! Replay: serialize/deserialize simulation logs for offline analysis & UI replay.

use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tracker_core::types::RadarBatch;

/// A full recorded simulation log.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayLog {
    pub scenario_name: String,
    pub seed: u64,
    pub sim_dt: f64,
    pub duration: f64,
    /// All radar batches in chronological order
    pub batches: Vec<RadarBatch>,
    /// Ground-truth target states, sampled every `sim_dt`
    pub ground_truth: Vec<GroundTruthFrame>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundTruthFrame {
    pub time: f64,
    pub targets: Vec<TargetState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetState {
    pub id: u64,
    pub state: [f64; 6],
}

/// Save a replay log to a JSON file.
pub fn save_replay(log: &ReplayLog, path: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, log)?;
    Ok(())
}

/// Load a replay log from a JSON file.
pub fn load_replay(path: &Path) -> anyhow::Result<ReplayLog> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let log: ReplayLog = serde_json::from_reader(reader)?;
    Ok(log)
}
