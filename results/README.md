# Inference Pipeline Benchmark Results

## Objective

Measure per-stage latency and variability for the YOLO26n inference pipeline using the inference backend and Triton server.

## Environment

- Model: `yolo26n`
- Server: `Triton Inference Server`
- CPU affinity: CPU 0 (`taskset -c 0`)
- Build configuration: Release
- Clock: `std::chrono::steady_clock`

See each run's `metadata.txt` for system and compiler details.

## Workload

- Configured images: 1,000
- Warm-up: 10
- Measured iterations: 100
- Recorded samples per stage: 100

The precise relationship between image count, iterations, and stage samples must be documented by the benchmark implementation.

## Reproduction

1. Build the project in Release mode.
2. Start the required Triton server with the correct model configuration.
3. Verify that the server is ready and the model is loaded.
4. Use the same input dataset and preprocessing configuration.
5. Run the benchmark using `scripts/run_benchmark.sh`.
6. Retain the raw output, environment metadata, Git revision, and any configuration needed to reproduce the run.

## Metrics

- Total duration: sum of recorded stage durations.
- Average: mean duration per recorded sample.
- Minimum and maximum: observed extremes.
- Variance and standard deviation: measures of latency variability.
- P50, P90, P95, and P99: latency percentiles.

All latency values are reported in milliseconds.

## Interpretation

Results characterize the specific hardware, software, workload, model configuration, and run conditions recorded for each experiment.

Stage durations should not be summed to infer end-to-end latency unless their timing intervals are known to be disjoint. Nested or overlapping stages can cause their summed duration to exceed the pipeline duration.

## Limitations

A single run is not sufficient to establish general performance. Repeat experiments under controlled conditions and compare distributions across independent runs before making performance claims.
