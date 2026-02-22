//! Scenario definitions.
//!
//! Each scenario is a named configuration of targets and radars.
//! All scenarios are deterministic given the same seed.

use crate::{
    radar_sim::{InjectedBias, SimRadar},
    target::{MotionSpec, Target},
};
use sensor_models::radar::RadarParams;
use serde::{Deserialize, Serialize};

/// Which pre-defined scenario to load.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
pub enum ScenarioKind {
    /// 5 targets, 2 radars, straight lines, low clutter
    Simple,
    /// 50 targets crossing, 3 radars, high clutter
    DenseCrossing,
    /// 500 targets, 4 radars — scalability stress test
    Stress,
    /// 20 targets, 2 radars with strong spatial + temporal bias (Phase C)
    BiasCalibration,
    /// 8 manoeuvring fighters: Split-S, Immelmann, high-G turns
    Fighter,
    /// 2000 planes in a mega-dense airspace (6 radars)
    MegaDense,
}

/// A fully configured simulation scenario.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub seed: u64,
    pub duration: f64, // seconds
    pub sim_dt: f64,   // simulation step (s) — between-event heartbeat
    pub targets: Vec<Target>,
    pub radars: Vec<SimRadar>,
}

impl Scenario {
    /// Build the named scenario. Uses `seed` for repeatability.
    pub fn build(kind: ScenarioKind, seed: u64) -> Self {
        match kind {
            ScenarioKind::Simple => Self::simple(seed),
            ScenarioKind::DenseCrossing => Self::dense_crossing(seed),
            ScenarioKind::Stress => Self::stress(seed),
            ScenarioKind::BiasCalibration => Self::bias_calibration(seed),
            ScenarioKind::Fighter => Self::fighter(seed),
            ScenarioKind::MegaDense => Self::mega_dense(seed),
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 1: Simple
    // -----------------------------------------------------------------------
    fn simple(seed: u64) -> Self {
        let targets = vec![
            target(
                0,
                [-20000., 0., 0.],
                [150., 0., 0.],
                MotionSpec::ConstantVelocity,
                None,
                None,
            ),
            target(
                1,
                [0., -20000., 0.],
                [0., 150., 0.],
                MotionSpec::ConstantVelocity,
                None,
                None,
            ),
            target(
                2,
                [10000., 10000., 0.],
                [-80., -80., 0.],
                MotionSpec::ConstantVelocity,
                None,
                None,
            ),
            target(
                3,
                [-5000., 15000., 0.],
                [100., -50., 0.],
                MotionSpec::ConstantTurn { omega: 0.01 },
                None,
                None,
            ),
            target(
                4,
                [15000., -5000., 0.],
                [-50., 120., 0.],
                MotionSpec::ConstantVelocity,
                None,
                None,
            ),
        ];

        let radars = vec![
            sim_radar(
                0,
                [-10000., -10000., 0.],
                1.0,
                0.9,
                2.0,
                200.,
                0.04,
                InjectedBias::default(),
            ),
            sim_radar(
                1,
                [10000., 10000., 0.],
                0.5,
                0.85,
                2.0,
                250.,
                0.05,
                InjectedBias::default(),
            ),
        ];

        Scenario {
            name: "simple".into(),
            seed,
            duration: 120.0,
            sim_dt: 0.1,
            targets,
            radars,
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 2: Dense Crossing
    // -----------------------------------------------------------------------
    fn dense_crossing(seed: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));

        let targets = (0..50)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 50.0;
                let r = 25000.0_f64;
                let px = r * angle.cos();
                let py = r * angle.sin();
                let speed = 100.0 + rng.gen::<f64>() * 150.0;
                let vangle = angle + std::f64::consts::PI; // heading toward center
                target(
                    i as u64,
                    [px, py, 0.],
                    [speed * vangle.cos(), speed * vangle.sin(), 0.],
                    MotionSpec::ConstantVelocity,
                    None,
                    None,
                )
            })
            .collect();

        let radars = vec![
            sim_radar(
                0,
                [0., 0., 0.],
                1.0,
                0.85,
                5.0,
                250.,
                0.05,
                InjectedBias::default(),
            ),
            sim_radar(
                1,
                [-15000., 0., 0.],
                0.5,
                0.80,
                5.0,
                300.,
                0.06,
                InjectedBias::default(),
            ),
            sim_radar(
                2,
                [15000., 0., 0.],
                0.75,
                0.88,
                5.0,
                200.,
                0.04,
                InjectedBias::default(),
            ),
        ];

        Scenario {
            name: "dense_crossing".into(),
            seed,
            duration: 300.0,
            sim_dt: 0.05,
            targets,
            radars,
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 3: Stress
    // -----------------------------------------------------------------------
    fn stress(seed: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(2));

        let targets = (0..500)
            .map(|i| {
                let px = (rng.gen::<f64>() - 0.5) * 200_000.0;
                let py = (rng.gen::<f64>() - 0.5) * 200_000.0;
                let vx = (rng.gen::<f64>() - 0.5) * 400.0;
                let vy = (rng.gen::<f64>() - 0.5) * 400.0;
                let motion = if rng.gen::<f64>() < 0.3 {
                    MotionSpec::ConstantTurn {
                        omega: (rng.gen::<f64>() - 0.5) * 0.05,
                    }
                } else {
                    MotionSpec::ConstantVelocity
                };
                target(i as u64, [px, py, 0.], [vx, vy, 0.], motion, None, None)
            })
            .collect();

        let radars = (0..4)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 4.0;
                let rx = 80_000.0 * angle.cos();
                let ry = 80_000.0 * angle.sin();
                sim_radar(
                    i as u32,
                    [rx, ry, 0.],
                    0.5 + i as f64 * 0.1,
                    0.80,
                    8.0,
                    250.,
                    0.05,
                    InjectedBias::default(),
                )
            })
            .collect();

        Scenario {
            name: "stress".into(),
            seed,
            duration: 60.0,
            sim_dt: 0.1,
            targets,
            radars,
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 4: Bias Calibration
    // -----------------------------------------------------------------------
    fn bias_calibration(seed: u64) -> Self {
        let targets = (0..20)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 20.0;
                target(
                    i as u64,
                    [10000. * angle.cos(), 10000. * angle.sin(), 0.],
                    [80. * (angle + 0.5).cos(), 80. * (angle + 0.5).sin(), 0.],
                    MotionSpec::ConstantVelocity,
                    None,
                    None,
                )
            })
            .collect();

        let radars = vec![
            sim_radar(
                0,
                [-5000., 0., 0.],
                1.0,
                0.9,
                2.0,
                200.,
                0.04,
                InjectedBias {
                    dx: 200.0,
                    dy: -150.0,
                    dtheta: 0.02,
                    dt0: 0.3,
                },
            ),
            sim_radar(
                1,
                [5000., 0., 0.],
                1.0,
                0.9,
                2.0,
                200.,
                0.04,
                InjectedBias {
                    dx: -100.0,
                    dy: 80.0,
                    dtheta: -0.015,
                    dt0: -0.2,
                },
            ),
        ];

        Scenario {
            name: "bias_calibration".into(),
            seed,
            duration: 180.0,
            sim_dt: 0.1,
            targets,
            radars,
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 5: Fighter — manoeuvring aircraft
    // -----------------------------------------------------------------------
    /// 8 agile fighters executing a sequence of manoeuvres over 3 minutes:
    ///
    /// ```
    /// t=0–15s   straight inbound at ~250 m/s (≈900 km/h)
    /// t=15–25s  hard 9-G turn (omega = ±0.25 rad/s, split-S or Immelmann)
    /// t=25–60s  second straight leg (post-turn escape run)
    /// t=60–75s  barrel-roll approximation (slower turn, opposite sense)
    /// t=75–180s final straight
    /// ```
    fn fighter(seed: u64) -> Self {
        let seg = |segs: Vec<(f64, MotionSpec)>| MotionSpec::Segmented {
            segments: segs.into_iter().map(|(t, m)| (t, Box::new(m))).collect(),
        };

        // 8 fighters arranged in two flights of 4, approaching from different sectors
        let targets = vec![
            // ---- Flight Alpha (from south-west, heading north-east) ----
            Target {
                id: 0,
                state: [-30000., -30000., 8000., 220., 220., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (15.0, MotionSpec::ConstantTurn { omega: 0.22 }), // hard left
                    (27.0, MotionSpec::ConstantVelocity),
                    (62.0, MotionSpec::ConstantTurn { omega: -0.18 }), // barrel roll right
                    (72.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 1,
                state: [-32000., -28000., 8200., 225., 215., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (18.0, MotionSpec::ConstantTurn { omega: -0.24 }), // Split-S right
                    (30.0, MotionSpec::ConstantVelocity),
                    (65.0, MotionSpec::ConstantTurn { omega: 0.20 }),
                    (75.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 2,
                state: [-28000., -32000., 7800., 215., 230., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (12.0, MotionSpec::ConstantTurn { omega: 0.28 }), // 9-G hard left
                    (
                        22.0,
                        MotionSpec::ConstantAccel {
                            ax: 5.,
                            ay: -3.,
                            az: 0.,
                        },
                    ),
                    (30.0, MotionSpec::ConstantVelocity),
                    (70.0, MotionSpec::ConstantTurn { omega: -0.15 }),
                    (80.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 3,
                state: [-35000., -25000., 8500., 230., 210., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (20.0, MotionSpec::ConstantTurn { omega: -0.26 }),
                    (32.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            // ---- Flight Bravo (from north-east, heading south-west) ----
            Target {
                id: 4,
                state: [30000., 30000., 9000., -240., -240., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (10.0, MotionSpec::ConstantTurn { omega: 0.20 }),
                    (22.0, MotionSpec::ConstantVelocity),
                    (55.0, MotionSpec::ConstantTurn { omega: -0.22 }),
                    (65.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 5,
                state: [28000., 32000., 8800., -235., -245., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (14.0, MotionSpec::ConstantTurn { omega: -0.25 }),
                    (26.0, MotionSpec::ConstantVelocity),
                    (60.0, MotionSpec::ConstantTurn { omega: 0.18 }),
                    (70.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 6,
                state: [32000., 28000., 9200., -250., -235., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (16.0, MotionSpec::ConstantTurn { omega: 0.23 }),
                    (
                        28.0,
                        MotionSpec::ConstantAccel {
                            ax: -4.,
                            ay: 6.,
                            az: 0.,
                        },
                    ),
                    (36.0, MotionSpec::ConstantVelocity),
                    (68.0, MotionSpec::ConstantTurn { omega: -0.20 }),
                    (78.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
            Target {
                id: 7,
                state: [25000., 35000., 7500., -245., -255., 0.],
                motion: seg(vec![
                    (0.0, MotionSpec::ConstantVelocity),
                    (22.0, MotionSpec::ConstantTurn { omega: -0.28 }), // max-G 10g
                    (34.0, MotionSpec::ConstantVelocity),
                ]),
                appear_at: None,
                disappear_at: None,
                history: std::collections::VecDeque::new(),
            },
        ];

        // Two high-performance air-intercept radars with faster refresh and good P_D
        let radars = vec![
            sim_radar(
                0,
                [-5000., 0., 0.],
                2.0,
                0.92,
                3.0,
                150.,
                0.03,
                InjectedBias::default(),
            ),
            sim_radar(
                1,
                [5000., 0., 0.],
                2.0,
                0.90,
                3.0,
                150.,
                0.03,
                InjectedBias::default(),
            ),
            sim_radar(
                2,
                [0., 8000., 0.],
                1.0,
                0.88,
                2.0,
                200.,
                0.04,
                InjectedBias::default(),
            ),
        ];

        Scenario {
            name: "fighter".into(),
            seed,
            duration: 180.0,
            sim_dt: 0.05, // 20 Hz — tight enough to capture 9-G manoeuvres
            targets,
            radars,
        }
    }

    // -----------------------------------------------------------------------
    // Scenario 6: MegaDense — 2000 targets
    // -----------------------------------------------------------------------
    fn mega_dense(seed: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(6));

        let targets = (0..2000)
            .map(|i| {
                let px = (rng.gen::<f64>() - 0.5) * 300_000.0;
                let py = (rng.gen::<f64>() - 0.5) * 300_000.0;
                let speed = 150.0 + rng.gen::<f64>() * 300.0; // 150–450 m/s
                let heading = rng.gen::<f64>() * std::f64::consts::TAU;
                let vx = speed * heading.cos();
                let vy = speed * heading.sin();
                let motion = match rng.gen_range(0u8..4) {
                    0 => MotionSpec::ConstantTurn {
                        omega: (rng.gen::<f64>() - 0.5) * 0.06,
                    },
                    1 => MotionSpec::ConstantAccel {
                        ax: (rng.gen::<f64>() - 0.5) * 4.0,
                        ay: (rng.gen::<f64>() - 0.5) * 4.0,
                        az: 0.0,
                    },
                    _ => MotionSpec::ConstantVelocity,
                };
                Target {
                    id: i as u64,
                    state: [px, py, 0., vx, vy, 0.],
                    motion,
                    appear_at: None,
                    disappear_at: None,
                    history: std::collections::VecDeque::new(),
                }
            })
            .collect();

        // 6 radars in a hexagonal ring at 60 km radius
        let radars = (0..6)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 6.0;
                sim_radar(
                    i as u32,
                    [60_000. * angle.cos(), 60_000. * angle.sin(), 0.],
                    0.5, // 0.5 Hz — slow enough to be realistic for long-range
                    0.80,
                    10.0, // 10 clutters/scan in this dense environment
                    300.,
                    0.06,
                    InjectedBias::default(),
                )
            })
            .collect();

        Scenario {
            name: "mega_dense".into(),
            seed,
            duration: 240.0,
            sim_dt: 0.1,
            targets,
            radars,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

fn target(
    id: u64,
    pos: [f64; 3],
    vel: [f64; 3],
    motion: MotionSpec,
    appear_at: Option<f64>,
    disappear_at: Option<f64>,
) -> Target {
    Target {
        id,
        state: [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]],
        motion,
        appear_at,
        disappear_at,
        history: std::collections::VecDeque::new(),
    }
}

fn sim_radar(
    id: u32,
    pos: [f64; 3],
    refresh_rate: f64,
    p_detection: f64,
    lambda_clutter: f64,
    range_noise_std: f64,
    az_noise_std: f64,
    bias: InjectedBias,
) -> SimRadar {
    SimRadar::new(
        id,
        RadarParams {
            position: pos,
            refresh_rate,
            p_detection,
            lambda_clutter,
            range_noise_std,
            azimuth_noise_std: az_noise_std,
            max_range: 150_000.0,
            fov_half: std::f64::consts::PI,
            output_cartesian: true,
            ..Default::default()
        },
        bias,
    )
}
