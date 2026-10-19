#include "inference_client.hpp"
#include "native_v4l2_camera.hpp"
#include "scoped_timer.hpp"
#include "stage_stats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr std::size_t kInputElements = 3 * 640 * 640;

std::atomic<bool> stop_requested{false};

void handle_signal(int) noexcept {
	stop_requested.store(true, std::memory_order_relaxed);
}

struct Options {
	std::string device = "/dev/video0";
	std::size_t warmup_iterations = 10;
	std::size_t measurement_iterations = 100;
	int width = kDefaultWidth;
	int height = kDefaultHeight;
	std::filesystem::path output_csv;
	std::string triton_url = "localhost:8001";
	std::string model_name = "yolo26n";
};

void print_usage(const char* program) {
	std::cout
		<< "Usage: " << program << " [options]\n\n"
		<< "Options:\n"
		<< "  --device PATH      V4L2 device (default: /dev/video0)\n"
		<< "  --warmup N         Warmup frames (default: 10)\n"
		<< "  --iterations N     Measured frames (default: 100)\n"
		<< "  --width N          Camera width (default: 1280)\n"
		<< "  --height N         Camera height (default: 720)\n"
		<< "  --url ADDRESS      Triton gRPC address (default: localhost:8001)\n"
		<< "  --model NAME       Triton model name (default: yolo26n)\n"
		<< "  --output FILE      Also write results as CSV\n"
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
		} else if (argument == "--warmup") {
			options.warmup_iterations = parse_count(
				require_value(index, argc, argv, "--warmup"), "--warmup");
		} else if (argument == "--iterations") {
			options.measurement_iterations = parse_count(
				require_value(index, argc, argv, "--iterations"), "--iterations");
		} else if (argument == "--width") {
			options.width = parse_dimension(
				require_value(index, argc, argv, "--width"), "--width");
		} else if (argument == "--height") {
			options.height = parse_dimension(
				require_value(index, argc, argv, "--height"), "--height");
		} else if (argument == "--url") {
			options.triton_url = require_value(index, argc, argv, "--url");
		} else if (argument == "--model") {
			options.model_name = require_value(index, argc, argv, "--model");
		} else if (argument == "--output") {
			options.output_csv = require_value(index, argc, argv, "--output");
		} else {
			throw std::invalid_argument("Unknown option: " + argument);
		}
	}
	if (options.measurement_iterations == 0) {
		throw std::invalid_argument("--iterations must be greater than zero");
	}
	return options;
}

void check_cuda(cudaError_t error, const char* operation) {
	if (error != cudaSuccess) {
		throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
	}
}

void write_csv(const std::filesystem::path& path, const StageStatsMap& statistics) {
	std::vector<std::string> stages;
	for (const auto& [stage, unused] : statistics) {
		static_cast<void>(unused);
		stages.push_back(stage);
	}
	std::sort(stages.begin(), stages.end());

	std::ofstream output(path);
	if (!output) {
		throw std::runtime_error("Unable to write CSV output: " + path.string());
	}
	output << "stage,samples,total_ms,average_ms,min_ms,max_ms,stddev_ms,p50_ms,p90_ms,p95_ms,p99_ms\n";
	output << std::fixed << std::setprecision(6);
	for (const auto& stage : stages) {
		const StageStats& stats = statistics.at(stage);
		output << stage << ',' << stats.samples() << ',' << stats.total_ms() << ',';
		const auto write_value = [&output](const std::optional<double>& value) {
			if (value.has_value()) {
				output << *value;
			}
			output << ',';
		};
		write_value(stats.average_ms());
		write_value(stats.min_ms());
		write_value(stats.max_ms());
		write_value(stats.stddev_ms());
		write_value(stats.percentile_ms(50.0));
		write_value(stats.percentile_ms(90.0));
		write_value(stats.percentile_ms(95.0));
		const auto p99 = stats.percentile_ms(99.0);
		if (p99.has_value()) {
			output << *p99;
		}
		output << '\n';
	}
}

} // namespace

int main(int argc, char** argv) {
	try {
		const Options options = parse_options(argc, argv);
		std::signal(SIGINT, handle_signal);
		std::signal(SIGTERM, handle_signal);

		StageStatsMap statistics;
		bool recording = false;
		bool capture_completed = false;
		const ScopedTimer::Callback record_timing =
			[&statistics, &recording, &capture_completed](const std::string& stage, double elapsed_ms) {
				if (recording && capture_completed) {
					statistics[stage].record(elapsed_ms);
				}
			};

		const std::size_t tensor_bytes = kInputElements * sizeof(float);
		float* device_tensor = nullptr;
		check_cuda(cudaMalloc(&device_tensor, tensor_bytes), "cudaMalloc");

		try {
			NativeV4L2Camera camera(
				options.device,
				static_cast<std::uint32_t>(options.width),
				static_cast<std::uint32_t>(options.height));
			if (!camera.init() || !camera.start()) {
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
			std::cout << "Benchmark configuration\n"
					  << "  device:      " << options.device << '\n'
					  << "  resolution:  " << options.width << 'x' << options.height << '\n'
					  << "  model:       " << options.model_name << '\n'
					  << "  server:      " << options.triton_url << '\n'
					  << "  warmup:      " << options.warmup_iterations << '\n'
					  << "  iterations:  " << options.measurement_iterations << '\n';

			const auto run_iteration = [&](bool collect_statistics) {
				bool captured = false;
				capture_completed = false;
				recording = collect_statistics;
				{
					ScopedTimer total_timer("total_ms", record_timing);
					{
						ScopedTimer timer("capture_and_preprocess", record_timing);
						captured = camera.getLatestFrameToGPU(device_tensor, stream);
						capture_completed = captured;
					}
					if (captured) {
						ScopedTimer timer("inference_pipeline", record_timing);
						static_cast<void>(client.infer(
							device_tensor, options.width, options.height, stream));
					}
				}
				recording = false;
				return captured;
			};

			std::size_t warmup_completed = 0;
			while (warmup_completed < options.warmup_iterations &&
				   !stop_requested.load(std::memory_order_relaxed)) {
				if (run_iteration(false)) {
					++warmup_completed;
				}
			}

			recording = true;
			std::size_t measured = 0;
			while (measured < options.measurement_iterations &&
				   !stop_requested.load(std::memory_order_relaxed)) {
				if (run_iteration(true)) {
					++measured;
				}
			}
			recording = false;

			check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
			check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
			if (measured != options.measurement_iterations) {
				throw std::runtime_error("Benchmark interrupted before all iterations completed");
			}
		} catch (...) {
			cudaFree(device_tensor);
			throw;
		}

		check_cuda(cudaFree(device_tensor), "cudaFree");
		print_summary(statistics);
		if (!options.output_csv.empty()) {
			write_csv(options.output_csv, statistics);
			std::cout << "\nCSV written to " << options.output_csv << '\n';
		}
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "Benchmark failed: " << error.what() << '\n';
		return 1;
	}
}
