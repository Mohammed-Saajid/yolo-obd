#include "inference_client.hpp"
#include "native_v4l2_camera.hpp"
#include "scoped_timer.hpp"
#include "stage_stats.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr std::size_t kInputChannels = 3;
constexpr std::size_t kInputSize = 640 * 640;

std::atomic<bool> stop_requested{false};

void handle_signal(int) noexcept {
	stop_requested.store(true, std::memory_order_relaxed);
}

struct Options {
	std::string device = "/dev/video0";
	std::string triton_url = "localhost:8001";
	std::string model_name = "yolo26n";
	int width = kDefaultWidth;
	int height = kDefaultHeight;
	std::size_t frames = 0;
};

void print_usage(const char* program) {
	std::cout
		<< "Usage: " << program << " [options]\n\n"
		<< "Options:\n"
		<< "  --device PATH      V4L2 device (default: /dev/video0)\n"
		<< "  --url ADDRESS      Triton gRPC address (default: localhost:8001)\n"
		<< "  --model NAME       Triton model name (default: yolo26n)\n"
		<< "  --width N          Camera width (default: 1280)\n"
		<< "  --height N         Camera height (default: 720)\n"
		<< "  --frames N         Stop after N frames; 0 means until Ctrl+C\n"
		<< "  --help             Show this help\n";
}

std::string require_value(int& index, int argc, char** argv, const char* option) {
	if (index + 1 >= argc) {
		throw std::invalid_argument(std::string(option) + " requires a value");
	}
	return argv[++index];
}

std::size_t parse_count(const std::string& value, const char* option) {
	std::size_t position = 0;
	const unsigned long long parsed = std::stoull(value, &position);
	if (position != value.size() || parsed > std::numeric_limits<std::size_t>::max()) {
		throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
	}
	return static_cast<std::size_t>(parsed);
}

int parse_dimension(const std::string& value, const char* option) {
	const std::size_t parsed = parse_count(value, option);
	if (parsed == 0 || parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
		throw std::invalid_argument(std::string(option) + " requires a positive integer");
	}
	return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string argument = argv[index];
		if (argument == "--help") {
			print_usage(argv[0]);
			std::exit(0);
		}
		if (argument == "--device") {
			options.device = require_value(index, argc, argv, "--device");
		} else if (argument == "--url") {
			options.triton_url = require_value(index, argc, argv, "--url");
		} else if (argument == "--model") {
			options.model_name = require_value(index, argc, argv, "--model");
		} else if (argument == "--width") {
			options.width = parse_dimension(require_value(index, argc, argv, "--width"), "--width");
		} else if (argument == "--height") {
			options.height = parse_dimension(require_value(index, argc, argv, "--height"), "--height");
		} else if (argument == "--frames") {
			options.frames = parse_count(require_value(index, argc, argv, "--frames"), "--frames");
		} else {
			throw std::invalid_argument("Unknown option: " + argument);
		}
	}
	return options;
}

void check_cuda(cudaError_t error, const char* operation) {
	if (error != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
	}
}

} // namespace

int main(int argc, char** argv) {
	try {
		const Options options = parse_options(argc, argv);
		std::signal(SIGINT, handle_signal);
		std::signal(SIGTERM, handle_signal);

		StageStatsMap statistics;
		const ScopedTimer::Callback record_timing =
			[&statistics](const std::string& stage, double elapsed_ms) {
				statistics[stage].record(elapsed_ms);
			};

		const std::size_t tensor_bytes =
			kInputChannels * kInputSize * sizeof(float);
		float* device_tensor = nullptr;
		check_cuda(cudaMalloc(&device_tensor, tensor_bytes), "cudaMalloc");

		try {
			NativeV4L2Camera camera(
				options.device,
				static_cast<std::uint32_t>(options.width),
				static_cast<std::uint32_t>(options.height));
			if (!camera.init()) {
				throw std::runtime_error("Unable to initialize camera: " + options.device);
			}
			if (!camera.start()) {
				throw std::runtime_error("Unable to start camera: " + options.device);
			}

			TritonInferenceClient client(
				options.triton_url,
				options.model_name,
				device_tensor,
				tensor_bytes,
				record_timing);
			if (!client.isServerLive()) {
				throw std::runtime_error("Triton server is not live at " + options.triton_url);
			}

			cudaStream_t stream = nullptr;
			check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

			std::cout << "Running " << options.model_name << " on " << options.device
					  << ". Press Ctrl+C to stop.\n";
			std::size_t frame_count = 0;
			while (!stop_requested.load(std::memory_order_relaxed) &&
				   (options.frames == 0 || frame_count < options.frames)) {
				{
					ScopedTimer timer("capture_and_preprocess", record_timing);
					if (!camera.getLatestFrameToGPU(device_tensor, stream)) {
						continue;
					}
				}

				std::vector<Detection> detections;
				{
					ScopedTimer timer("inference_pipeline", record_timing);
					detections = client.infer(
						device_tensor, options.width, options.height, stream);
				}

				++frame_count;
				std::cout << "Frame " << frame_count << ": "
						  << detections.size() << " detection(s)\n";
			}

			check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
			check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
			std::cout << "Processed " << frame_count << " frame(s).\n";
		} catch (...) {
			cudaFree(device_tensor);
			throw;
		}

		check_cuda(cudaFree(device_tensor), "cudaFree");
		print_summary(statistics);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "Inference backend failed: " << error.what() << '\n';
		return 1;
	}
}
