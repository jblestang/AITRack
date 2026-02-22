//! Fundamental types used across the entire workspace.

use nalgebra::{DMatrix, DVector, Matrix6, Vector6};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Scalar type: use f64 throughout for numerical precision in Kalman filter.
// ---------------------------------------------------------------------------

/// 6-DOF state vector: [px, py, pz, vx, vy, vz]
pub type StateVec = Vector6<f64>;

/// 6×6 state covariance matrix
pub type StateCov = Matrix6<f64>;

/// Generic dynamic-size vector (used for measurement innovation)
pub type DVec = DVector<f64>;

/// Generic dynamic-size matrix (used for H, R, S)
pub type DMat = DMatrix<f64>;

// ---------------------------------------------------------------------------
// Identifier types — newtype wrappers so IDs are never confused at compile time
// ---------------------------------------------------------------------------

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct TrackId(pub u64);

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct SensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MeasurementId(pub u64);

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

impl fmt::Display for SensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// A single measurement returned by a radar.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurement {
    /// Unique measurement identifier within a batch
    pub id: MeasurementId,
    /// Which radar produced this measurement
    pub sensor_id: SensorId,
    /// Measurement timestamp in simulation seconds
    pub timestamp: f64,
    /// Observation value (already bias-corrected if correction is enabled)
    pub value: MeasurementValue,
    /// Measurement noise covariance R (in the observation space)
    pub noise_cov: Vec<f64>, // row-major, dim × dim; use DMat in pipeline
}

impl Measurement {
    /// Return the measurement as a 2D cartesian vector [x, y].
    /// For 3D measurements the z component is dropped if not available.
    pub fn to_cartesian_2d(&self) -> DVector<f64> {
        match &self.value {
            MeasurementValue::Cartesian2D { x, y } => DVector::from_vec(vec![*x, *y]),
            MeasurementValue::Cartesian3D { x, y, .. } => DVector::from_vec(vec![*x, *y]),
            MeasurementValue::Polar2D { range, azimuth } => {
                let x = range * azimuth.cos();
                let y = range * azimuth.sin();
                DVector::from_vec(vec![x, y])
            }
        }
    }

    /// Dimension of the observation vector
    pub fn dim(&self) -> usize {
        match &self.value {
            MeasurementValue::Cartesian2D { .. } => 2,
            MeasurementValue::Polar2D { .. } => 2,
            MeasurementValue::Cartesian3D { .. } => 3,
        }
    }

    /// Return noise covariance as a DMatrix
    pub fn noise_cov_matrix(&self) -> DMatrix<f64> {
        let d = self.dim();
        DMatrix::from_row_slice(d, d, &self.noise_cov)
    }
}

/// The actual observation value carried by a [`Measurement`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MeasurementValue {
    /// Pre-converted 2D cartesian coordinates (meters)
    Cartesian2D { x: f64, y: f64 },
    /// Pre-converted 3D cartesian coordinates (meters)
    Cartesian3D { x: f64, y: f64, z: f64 },
    /// Polar coordinates from radar (meters, radians)
    Polar2D { range: f64, azimuth: f64 },
}

// ---------------------------------------------------------------------------
// RadarBatch — a timestamped batch of measurements from one sensor
// ---------------------------------------------------------------------------

/// A batch of measurements from one radar at a given time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadarBatch {
    pub sensor_id: SensorId,
    /// Time the batch was generated / emitted by the sensor (simulation clock)
    pub sensor_time: f64,
    /// Time the batch arrived at the tracker (may differ due to latency)
    pub arrival_time: f64,
    pub measurements: Vec<Measurement>,
}
