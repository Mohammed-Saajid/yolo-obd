#include "inference_client.hpp"
#include "scoped_timer.hpp"
#include "stage_stats.hpp"
#include "image_reader.cpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
	std::filesystem::path input_directory = "data/coco_1k_hf/images";
	std::size_t warmup_iterations = 10;
	std::size_t measurement_iterations = 100;
	std::size_t max_images = 0;
	std::filesystem::path output_csv;
	std::string triton_url = "localhost:8001";
	std::string model_name = "yolo26n";
};

void print_usage(const char* program) {
	std::cout
		<< "Usage: " << program << " [options]\n\n"
		<< "Options:\n"
		<< "  --input DIR       Directory of benchmark images\n"
		<< "                    (default: data/coco_1k_hf/images)\n"
		<< "  --warmup N        Warmup iterations, excluded from results (default: 10)\n"
		<< "  --iterations N    Measured iterations (default: 100)\n"
		<< "  --max-images N    Use at most N images; 0 means all (default: 0)\n"
		<< "  --url ADDRESS     Triton gRPC address (default: localhost:8001)\n"
		<< "  --model NAME      Triton model name (default: yolo26n)\n"
		<< "  --output FILE     Also write results as CSV\n"
		<< "  --help            Show this help\n";
}

std::size_t parse_count(const std::string& value, const char* option) {
	std::size_t position = 0;
	const auto count = std::stoull(value, &position);
	if (position != value.size()) {
		throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
	}
	return static_cast<std::size_t>(count);
}

Options parse_options(int argc, char** argv) {
	Options options;
	for (int index = 1; index < argc; ++index) {
		const std::string argument = argv[index];
		const auto require_value = [&](const char* option) -> std::string {
			if (index + 1 >= argc) {
				throw std::invalid_argument(std::string(option) + " requires a value");
			}
			return argv[++index];
		};

		if (argument == "--help") {
			print_usage(argv[0]);
			std::exit(0);
		} else if (argument == "--input") {
			options.input_directory = require_value("--input");
		} else if (argument == "--warmup") {
			options.warmup_iterations = parse_count(require_value("--warmup"), "--warmup");
		} else if (argument == "--iterations") {
			options.measurement_iterations = parse_count(require_value("--iterations"), "--iterations");
		} else if (argument == "--max-images") {
			options.max_images = parse_count(require_value("--max-images"), "--max-images");
		} else if (argument == "--url") {
			options.triton_url = require_value("--url");
		} else if (argument == "--model") {
			options.model_name = require_value("--model");
		} else if (argument == "--output") {
			options.output_csv = require_value("--output");
		} else {
			throw std::invalid_argument("Unknown option: " + argument);
		}
	}

	if (options.measurement_iterations == 0) {
		throw std::invalid_argument("--iterations must be greater than zero");
	}
	return options;
}

void write_csv(const std::filesystem::path& path, const StageStatsMap& statistics) {
	std::vector<std::string> stages;
	stages.reserve(statistics.size());
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
		const auto write_value = [&](const std::optional<double>& value) {
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

}  // namespace

int main(int argc, char** argv) {
	try {
		const Options options = parse_options(argc, argv);
		DirectoryImageReader image_reader(options.input_directory);
		auto images = image_reader.scan_image_paths();
		if (options.max_images != 0 && images.size() > options.max_images) {
			images.resize(options.max_images);
		}
		if (images.empty()) {
			throw std::runtime_error("No supported images found in: " + options.input_directory.string());
		}
		StageStatsMap statistics;
		bool recording = false;
		const ScopedTimer::Callback record_timing =
			[&statistics, &recording](const std::string& stage, double elapsed_ms) {
				if (recording) {
					statistics[stage].record(elapsed_ms);
				}
			};

		TritonInferenceClient client(options.triton_url, options.model_name, record_timing);
		if (!client.isServerLive()) {
			throw std::runtime_error("Triton server is not live at " + options.triton_url);
		}

		const auto run_iteration = [&](const std::filesystem::path& image_path) {
		DirectoryImageReader::ImageRecord image_record;
			{
				ScopedTimer timer("image_load", record_timing);
				image_record = DirectoryImageReader::read_single(image_path);
			}
			if (!image_record.is_valid) {
				throw std::runtime_error("Unable to decode image: " + image_path.string());
			}
			{
				ScopedTimer timer("inference_pipeline", record_timing);
				static_cast<void>(client.infer(image_record.image));
			}
		};

		std::cout << "Benchmark configuration\n"
				  << "  model:       " << options.model_name << '\n'
				  << "  server:      " << options.triton_url << '\n'
				  << "  images:      " << images.size() << '\n'
				  << "  warmup:      " << options.warmup_iterations << '\n'
				  << "  iterations:  " << options.measurement_iterations << '\n';

		for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
			run_iteration(images[iteration % images.size()]);
		}

		recording = true;
		for (std::size_t iteration = 0; iteration < options.measurement_iterations; ++iteration) {
			run_iteration(images[iteration % images.size()]);
		}
		recording = false;

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
