# YOLOv26n Real-Time Inference Backend

A high-performance C++ inference client for real-time object detection using YOLOv26n and NVIDIA Triton Inference Server. This project implements a complete edge-ready pipeline from raw webcam frames to visualized bounding box detections.

## Features

- **High-Performance C++ Backend:** Efficiently handles frame preprocessing, gRPC communication, and post-processing.
- **YOLOv26n Optimized:** Designed specifically for the YOLOv26n architecture.
- **Triton gRPC Client:** Uses gRPC protocol for low-latency inference on Triton Inference Server.
- **OpenCV Integration:** Seamlessly integrates with OpenCV for hardware-accelerated video capture and visualization.
- **Edge Ready:** Modular architecture suitable for deployment on edge devices and high-load production servers.

## Tech Stack

- **Language:** C++17
- **Inference Server:** NVIDIA Triton Inference Server 24.05
- **Computer Vision:** OpenCV
- **Communication:** gRPC
- **Build System:** CMake

## Project Structure

```text
.
├── include/                # Header files (inference_client.hpp)
├── src/                    # Implementation (main.cpp, inference_client.cpp)
├── model_repository/       # Triton model repository configuration
│   └── triton_serve/
│       └── yolo26n/       # yolo26n model configuration and versioning
├── scripts/                # Helper scripts for server management and conversion
├── third_party/            # Triton C++ Client SDK and libraries
└── CMakeLists.txt          # Build configuration
```

## Getting Started

### Prerequisites

- **OS:** Linux (Ubuntu recommended)
- **NVIDIA GPU:** Required for GPU-accelerated inference
- **NVIDIA Container Toolkit:** For running Triton in Docker
- **OpenCV:** Installed on your system (`sudo apt install libopencv-dev`)
- **Docker:** Required for starting the inference server

### 1. Setup the Triton Inference Server

Ensure your yolo26n model weights are placed in [model_repository/triton_serve/yolo26n/1/](model_repository/triton_serve/yolo26n/1/). 

Start the server using the provided script:

```bash
chmod +x scripts/start_triton.sh
./scripts/start_triton.sh
```

### 2. Build the C++ Backend

The project uses CMake for building. Ensure you have the `third_party/triton` SDK available in the root folder.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Run the Application

The application will attempt to connect to the Triton server at `localhost:8001` (gRPC) and open your default webcam (`/dev/video0`).

```bash
./InferenceBackend
```

## Configuration

- **Triton URL:** Adjustable in [src/main.cpp](src/main.cpp) (Default: `localhost:8001`).
- **Model Name:** Set to `yolo26n` in [src/main.cpp](src/main.cpp).
- **Camera Device:** Index can be changed in the `cv::VideoCapture` initialization in [src/main.cpp](src/main.cpp).

## Performance Benchmarks

The backend reports per-stage timing statistics when the application exits. The
reported stages include camera capture, preprocessing, Triton inference,
postprocessing, rendering, display, and the complete inference pipeline. Each
stage includes its sample count, average, minimum, and maximum duration in
milliseconds.

The following benchmarks were recorded using `perf_analyzer` on an NVIDIA Release 24.05 environment with a batch size of 1.

| Metric | Value |
| :--- | :--- |
| **Throughput** | 28.11 infer/sec |
| **Avg Latency** | 34.39 ms |
| **p50 Latency** | 32.72 ms |
| **p90 Latency** | 48.77 ms |
| **p95 Latency** | 55.00 ms |
| **p99 Latency** | 63.67 ms |

### Server-Side Statistics (per request)
- **Compute Infer:** 21.31 ms
- **Compute Input:** 1.42 ms
- **Compute Output:** 0.52 ms
- **Queue Time:** 0.32 ms
- **Overhead:** 1.91 ms

### Reproducible Stage Benchmark

The `timing_benchmark` executable runs the production preprocessing, Triton
inference, and postprocessing path without opening a camera or display window.
Images are discovered in sorted filename order. Warmup iterations are excluded
from the reported statistics, and measurements include image loading and the
complete inference pipeline.

With Triton running and the dataset downloaded, run:

```bash
cmake -S . -B build
cmake --build build --target timing_benchmark -j$(nproc)
./build/timing_benchmark \
	--input data/coco_1k_hf/images \
	--warmup 20 \
	--iterations 500 \
	--output benchmark-results.csv
```

Use `--max-images N` to limit the deterministic image set. The CSV contains
sample count, total, average, minimum, maximum, standard deviation, and p50,
p90, p95, and p99 latency for each stage. Set `--url` and `--model` when
benchmarking a different Triton endpoint or model.

## License

This project is open-source and available under the MIT License.
