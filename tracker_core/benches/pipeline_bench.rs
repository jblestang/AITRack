use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tracker_core::pipeline::{Pipeline, PipelineConfig};
use tracker_core::types::{Measurement, MeasurementId, MeasurementValue, RadarBatch, SensorId};

fn make_batch(n: usize, t: f64) -> RadarBatch {
    let measurements = (0..n)
        .map(|i| {
            let angle = i as f64 * std::f64::consts::TAU / n as f64;
            let r = 10000.0_f64;
            Measurement {
                id: MeasurementId(i as u64),
                sensor_id: SensorId(0),
                timestamp: t,
                value: MeasurementValue::Cartesian2D {
                    x: r * angle.cos(),
                    y: r * angle.sin(),
                },
                noise_cov: vec![2500.0, 0.0, 0.0, 2500.0],
            }
        })
        .collect();
    RadarBatch {
        sensor_id: SensorId(0),
        sensor_time: t,
        arrival_time: t,
        measurements,
    }
}

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");

    for n in [50, 500, 1000, 2000] {
        group.bench_function(format!("{n}_targets"), |b| {
            b.iter(|| {
                let mut pipeline = Pipeline::new(PipelineConfig::default());
                // Warm up with one batch to create tracks
                pipeline.process_batch(&make_batch(n, 0.0));
                // Measure full batch processing with established tracks
                let batch = make_batch(n, 1.0);
                black_box(pipeline.process_batch(&batch));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
