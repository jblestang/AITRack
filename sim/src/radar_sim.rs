//! Radar measurement simulator.
//!
//! Generates asynchronous radar batches with:
//! - Gaussian position/angle noise
//! - Miss probability (1 - P_D)
//! - Poisson clutter (false alarms)
//! - Configurable spatial/temporal bias injection (for Phase C scenarios)

use crate::target::Target;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use sensor_models::radar::RadarParams;
use serde::{Deserialize, Serialize};
use tracker_core::types::{Measurement, MeasurementId, MeasurementValue, RadarBatch, SensorId};

/// Known bias parameters injected by the simulator (ground truth for Phase C).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InjectedBias {
    /// Spatial: offset x (meters)
    pub dx: f64,
    /// Spatial: offset y (meters)
    pub dy: f64,
    /// Spatial: rotation (radians)
    pub dtheta: f64,
    /// Temporal: clock offset (seconds) — measurements appear earlier/later
    pub dt0: f64,
}

/// One configured radar in the simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimRadar {
    pub id: SensorId,
    pub params: RadarParams,
    /// Injected bias (known to the simulator, unknown to the tracker)
    pub bias: InjectedBias,
    /// Next scheduled scan time
    pub next_scan_time: f64,
}

impl SimRadar {
    pub fn new(id: u32, params: RadarParams, bias: InjectedBias) -> Self {
        let _dt = 1.0 / params.refresh_rate;
        Self {
            id: SensorId(id),
            next_scan_time: 0.0,
            params,
            bias,
        }
    }

    /// Check if this radar should fire at the current simulation time.
    pub fn should_scan(&self, t: f64) -> bool {
        t >= self.next_scan_time
    }

    /// Advance the schedule by one scan interval.
    pub fn advance_schedule(&mut self) {
        self.next_scan_time += 1.0 / self.params.refresh_rate;
    }
}

/// Generates radar measurement batches from a set of targets.
pub struct RadarSimulator {
    pub radars: Vec<SimRadar>,
    rng: ChaCha8Rng,
}

impl RadarSimulator {
    pub fn new(radars: Vec<SimRadar>, seed: u64) -> Self {
        Self {
            radars,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Generate all batches that should fire at or before `sim_time`.
    /// Returns (batches, ground_truth_at_fire_time).
    pub fn generate_batches(
        &mut self,
        targets: &[Target],
        sim_time: f64,
        meas_id_start: &mut u64,
    ) -> Vec<RadarBatch> {
        let mut batches = Vec::new();

        for radar in &mut self.radars {
            if !radar.should_scan(sim_time) {
                continue;
            }
            let scan_time = radar.next_scan_time;
            radar.advance_schedule();

            let mut measurements = Vec::new();

            // True detections
            for target in targets {
                if !target.is_active(scan_time) {
                    continue;
                }

                // Miss detection?
                if self.rng.gen::<f64>() > radar.params.p_detection {
                    continue;
                }

                let tx = target.state[0];
                let ty = target.state[1];
                let rpos = radar.params.position;

                let dx = tx - rpos[0];
                let dy = ty - rpos[1];
                let range = (dx * dx + dy * dy).sqrt();
                let azimuth = dy.atan2(dx);

                // Check range and FoV
                if range > radar.params.max_range {
                    continue;
                }
                let az_local = (azimuth - radar.params.heading + std::f64::consts::PI)
                    .rem_euclid(2.0 * std::f64::consts::PI)
                    - std::f64::consts::PI;
                if az_local.abs() > radar.params.fov_half {
                    continue;
                }

                // Add noise
                let noisy_range =
                    range + self.rng.gen::<f64>() * radar.params.range_noise_std * 2.0
                        - radar.params.range_noise_std;
                let noisy_az = azimuth
                    + self.rng.gen::<f64>() * radar.params.azimuth_noise_std * 2.0
                        - radar.params.azimuth_noise_std;

                // Convert to cartesian (with bias injection)
                let (mx, my) = if radar.params.output_cartesian {
                    let xraw = rpos[0] + noisy_range * noisy_az.cos();
                    let yraw = rpos[1] + noisy_range * noisy_az.sin();
                    // Apply injected bias (rotation then translation)
                    let cos_t = radar.bias.dtheta.cos();
                    let sin_t = radar.bias.dtheta.sin();
                    (
                        cos_t * xraw - sin_t * yraw + radar.bias.dx,
                        sin_t * xraw + cos_t * yraw + radar.bias.dy,
                    )
                } else {
                    (noisy_range, noisy_az)
                };

                // Temporal bias: shift the reported timestamp
                let reported_time = scan_time + radar.bias.dt0;

                let sigma_xy = range * radar.params.azimuth_noise_std + radar.params.range_noise_std;
                let sigma_sq = sigma_xy * sigma_xy;

                let id = MeasurementId(*meas_id_start);
                *meas_id_start += 1;

                let value = if radar.params.output_cartesian {
                    MeasurementValue::Cartesian2D { x: mx, y: my }
                } else {
                    MeasurementValue::Polar2D { range: mx, azimuth: my }
                };

                measurements.push(Measurement {
                    id,
                    sensor_id: radar.id,
                    timestamp: reported_time,
                    value,
                    noise_cov: vec![sigma_sq, 0.0, 0.0, sigma_sq],
                });
            }

            // Clutter (Poisson)
            // lambda_clutter = mean number of false alarms per scan
            // Use Poisson approximation: sample from Poisson(lambda)
            // Simple approximation: n_clutter = floor(-lambda * ln(U)) is Poisson but
            // here we just use a capped uniform for speed in Phase A.
            let lambda = radar.params.lambda_clutter;
            // Draw Poisson sample: sum of geometric ≈ use simple inversion for small lambda
            let n_clutter = if lambda <= 0.0 {
                0usize
            } else {
                // Approximate Poisson by drawing N until product of U < e^{-lambda}
                let mut n = 0usize;
                let threshold = (-lambda).exp();
                let mut prod = self.rng.gen::<f64>();
                while prod > threshold && n < 50 {
                    prod *= self.rng.gen::<f64>();
                    n += 1;
                }
                n
            };
            for _ in 0..n_clutter {
                let clutter_range = radar.params.max_range * self.rng.gen::<f64>().sqrt();
                let clutter_az = self.rng.gen::<f64>() * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
                let rpos = radar.params.position;
                let x = rpos[0] + clutter_range * clutter_az.cos() + radar.bias.dx;
                let y = rpos[1] + clutter_range * clutter_az.sin() + radar.bias.dy;
                let sigma_sq = (radar.params.range_noise_std * 2.0).powi(2);
                let id = MeasurementId(*meas_id_start);
                *meas_id_start += 1;
                measurements.push(Measurement {
                    id,
                    sensor_id: radar.id,
                    timestamp: scan_time,
                    value: MeasurementValue::Cartesian2D { x, y },
                    noise_cov: vec![sigma_sq, 0.0, 0.0, sigma_sq],
                });
            }

            batches.push(RadarBatch {
                sensor_id: radar.id,
                sensor_time: scan_time,
                arrival_time: sim_time,
                measurements,
            });
        }

        batches
    }
}
