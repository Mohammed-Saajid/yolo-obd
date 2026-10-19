# YOLOv26n Real-Time Inference Backend

A C++17 real-time object detection pipeline for NVIDIA GPUs. Frames are captured from a V4L2 camera, preprocessed on the GPU, sent to NVIDIA Triton through CUDA shared memory, and postprocessed into detections.

The current runtime path is designed for low-latency camera inference rather than an OpenCV display loop. The benchmark executable measures the same production path used by `InferenceBackend`.

## Pipeline

```text
V4L2 camera
    -> MMAP capture buffers
    -> CUDA-mapped camera memory
    -> GPU YUYV to normalized RGB planar preprocessing
    -> 3 x 640 x 640 FP32 CUDA tensor
    -> Triton CUDA shared memory
    -> YOLOv26n inference
    -> detection postprocessing
```

The camera can capture at a configurable resolution, while the CUDA preprocessing kernel samples that frame into the model's fixed `640 x 640` input tensor. This prevents host copies and avoids allocating an output tensor sized for the camera frame instead of the model input.

## Key Capabilities

- Native V4L2 capture with MMAP buffers.
- CUDA registration and zero-copy access to camera buffers.
- GPU YUYV preprocessing into normalized planar FP32 RGB.
- CUDA stream-based preprocessing and inference synchronization.
- Triton gRPC inference with CUDA shared memory registration.
- YOLOv26n output decoding into bounding-box detections.
- Signal-aware shutdown and explicit CUDA resource cleanup.
- Stage-level latency statistics with totals, averages, extrema, variance, standard deviation, and percentiles.
- Reproducible camera benchmarks with warmup exclusion and CSV export.

## Requirements

- Linux with a V4L2 camera, for example `/dev/video0`.
- NVIDIA GPU and CUDA Toolkit.
- NVIDIA Container Toolkit and Docker for Triton.
- OpenCV 5 development libraries.
- CMake 3.20 or newer.
- C++17 compiler.
- Triton C++ client libraries in `third_party/triton`.

The configured model expects:

| Tensor | Type | Shape |
| --- | --- | --- |
| `images` | FP32 | `1 x 3 x 640 x 640` |
| `output0` | FP32 | `1 x 300 x 6` |

## Build

Configure and build the debug or release tree:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j"$(nproc)"
```

For a release build:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j"$(nproc)"
```

The main targets are:

- `InferenceBackend`: live camera inference.
- `timing_benchmark`: production-path camera benchmark.
- `stage_stats_tests`: statistics unit tests, when enabled by the configured build.

## Start Triton

The model repository is under `model_repository/triton_serve`. Start Triton in a separate terminal:

```bash
./scripts/start_triton.sh
```

The server exposes gRPC on `localhost:8001` by default. Confirm that the model is ready before starting the client.

## Run Live Inference

Run until interrupted:

```bash
./build/release/InferenceBackend
```

Run a bounded smoke test:

```bash
./build/debug/InferenceBackend \
    --device /dev/video0 \
    --url localhost:8001 \
    --model yolo26n \
    --width 1280 \
    --height 720 \
    --frames 100
```

Available options:

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `/dev/video0` | V4L2 camera device |
| `--url` | `localhost:8001` | Triton gRPC endpoint |
| `--model` | `yolo26n` | Triton model name |
| `--width` | `1280` | Requested camera width |
| `--height` | `720` | Requested camera height |
| `--frames` | `0` | Number of frames; `0` runs until Ctrl+C |

## Benchmark the Current Pipeline

`timing_benchmark` uses the same native V4L2, CUDA preprocessing, Triton, and postprocessing path as the live executable. Warmup frames are excluded. Only successfully captured and inferred frames are counted in the measured statistics.

```bash
./build/release/timing_benchmark \
    --device /dev/video0 \
    --warmup 20 \
    --iterations 1000 \
    --output benchmark.csv
```

The benchmark CSV uses this schema:

```text
stage,samples,total_ms,average_ms,min_ms,max_ms,stddev_ms,p50_ms,p90_ms,p95_ms,p99_ms
```

Current stages are:

- `capture_and_preprocess`: V4L2 buffer acquisition and CUDA preprocessing launch.
- `triton_inference`: Triton request and response handling.
- `postprocess`: YOLO output decoding and coordinate scaling.
- `inference_pipeline`: Complete inference operation, including Triton and postprocessing.
- `total_ms`: Complete measured frame interval.

To create timestamped benchmark artifacts with metadata:

```bash
./scripts/run_client_benchmark.sh
```

This stores `benchmark.csv`, console output, metadata, and the source revision under `results/runs/<run-number>/<timestamp>/`. The run number can be supplied as the first argument:

```bash
./scripts/run_client_benchmark.sh 3
```

To average all CSV files in the numbered result directories:

```bash
python3 scripts/write_average.py
```

The aggregation script supports both root-level CSV files and timestamped nested benchmark directories.

## Latest Recorded Baseline

The latest recorded run-3 summaries contain three 1,000-frame benchmark runs at `1280 x 720`, with 10 warmup frames per run. Values below are averages across those three runs; latency is measured in milliseconds.

| Stage | Samples/run | Total (ms) | Average (ms) | Min (ms) | Max (ms) | Stddev (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `total_ms` | 1000 | 16978.499 | 16.978 | 13.309 | 40.821 | 1.332 | 16.807 | 17.569 | 18.150 | 21.247 |
| `capture_and_preprocess` | 1000 | 28.150 | 0.028 | 0.017 | 0.299 | 0.025 | 0.023 | 0.033 | 0.049 | 0.147 |
| `inference_pipeline` | 1000 | 16949.377 | 16.949 | 13.287 | 40.792 | 1.330 | 16.780 | 17.539 | 18.122 | 21.216 |
| `triton_inference` | 1000 | 16571.877 | 16.572 | 12.953 | 40.225 | 1.311 | 16.405 | 17.153 | 17.721 | 20.716 |
| `postprocess` | 1000 | 0.899 | 0.001 | 0.000 | 0.111 | 0.005 | 0.001 | 0.001 | 0.001 | 0.001 |

The mean end-to-end latency of 16.978 ms corresponds to approximately 58.9 measured frames per second. Throughput depends on camera mode, GPU, Triton configuration, server load, and whether the client is CPU- or scheduler-limited.

These values are an engineering baseline, not a universal performance guarantee. For comparisons, keep the camera resolution, warmup count, measured iteration count, model configuration, Triton version, GPU, and CPU affinity consistent.

## Repository Layout

```text
.
├── include/                 Public C++ interfaces
├── src/                     Live client, V4L2 camera, CUDA preprocessing, statistics
├── benchmarks/              Production-path timing benchmark
├── scripts/                 Triton startup, benchmark, dataset, and export helpers
├── model_repository/        Triton model configuration and model files
├── third_party/triton/      Triton C++ client SDK and libraries
├── tests/                   Unit tests
└── results/runs/            Timestamped benchmark artifacts and averages
```

## Notes

- The current camera path expects a capture format compatible with the CUDA YUYV preprocessing kernel. The default integrated camera exposes YUYV at `1280 x 720`.
- `/dev/video1` may expose metadata rather than video frames; use a video-capture node such as `/dev/video0`.
- The client requires Triton to be live before CUDA shared memory registration can succeed.
- The benchmark intentionally does not include display or rendering time. Those stages belonged to the earlier OpenCV experiment and are not part of the current production path.

## License

This project is open-source and available under the MIT License.
