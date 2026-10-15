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
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace
{

	struct Options
	{
		std::filesystem::path input_directory = "data/coco_1k_hf/images";
		std::size_t warmup_iterations = 10;
		std::size_t measurement_iterations = 100;
		std::size_t max_images = 0;
		int camera_index = 0;
		bool use_camera = false;
		std::filesystem::path output_csv;
		std::string triton_url = "localhost:8001";
		std::string model_name = "yolo26n";
	};

	void print_usage(const char *program)
	{
		std::cout
			<< "Usage: " << program << " [options]\n\n"
			<< "Options:\n"
			<< "  --input DIR       Directory of benchmark images\n"
			<< "                    (default: data/coco_1k_hf/images)\n"
			<< "  --warmup N        Warmup iterations, excluded from results (default: 10)\n"
			<< "  --iterations N    Measured iterations (default: 100)\n"
			<< "  --max-images N    Use at most N images; 0 means all (default: 0)\n"
			<< "  --camera INDEX    Benchmark a realtime OpenCV camera (default: 0)\n"
			<< "  --url ADDRESS     Triton gRPC address (default: localhost:8001)\n"
			<< "  --model NAME      Triton model name (default: yolo26n)\n"
			<< "  --output FILE     Also write results as CSV\n"
			<< "  --help            Show this help\n";
	}

	std::size_t parse_count(const std::string &value, const char *option)
	{
		std::size_t position = 0;
		const auto count = std::stoull(value, &position);
		if (position != value.size())
		{
			throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
		}
		return static_cast<std::size_t>(count);
	}

	Options parse_options(int argc, char **argv)
	{
		Options options;
		for (int index = 1; index < argc; ++index)
		{
			const std::string argument = argv[index];
			const auto require_value = [&](const char *option) -> std::string
			{
				if (index + 1 >= argc)
				{
					throw std::invalid_argument(std::string(option) + " requires a value");
				}
				return argv[++index];
			};

			if (argument == "--help")
			{
				print_usage(argv[0]);
				std::exit(0);
			}
			else if (argument == "--input")
			{
				options.input_directory = require_value("--input");
			}
			else if (argument == "--warmup")
			{
				options.warmup_iterations = parse_count(require_value("--warmup"), "--warmup");
			}
			else if (argument == "--iterations")
			{
				options.measurement_iterations = parse_count(require_value("--iterations"), "--iterations");
			}
			else if (argument == "--max-images")
			{
				options.max_images = parse_count(require_value("--max-images"), "--max-images");
			}
			else if (argument == "--camera")
			{
				const auto value = require_value("--camera");
				const auto camera_index = parse_count(value, "--camera");
				if (camera_index > static_cast<std::size_t>(std::numeric_limits<int>::max()))
				{
					throw std::invalid_argument("--camera index is too large");
				}
				options.camera_index = static_cast<int>(camera_index);
				options.use_camera = true;
			}
			else if (argument == "--url")
			{
				options.triton_url = require_value("--url");
			}
			else if (argument == "--model")
			{
				options.model_name = require_value("--model");
			}
			else if (argument == "--output")
			{
				options.output_csv = require_value("--output");
			}
			else
			{
				throw std::invalid_argument("Unknown option: " + argument);
			}
		}

		if (options.measurement_iterations == 0)
		{
			throw std::invalid_argument("--iterations must be greater than zero");
		}
		return options;
	}

	void write_csv(const std::filesystem::path &path, const StageStatsMap &statistics)
	{
		std::vector<std::string> stages;
		stages.reserve(statistics.size());
		for (const auto &[stage, unused] : statistics)
		{
			static_cast<void>(unused);
			stages.push_back(stage);
		}
		std::sort(stages.begin(), stages.end());

		std::ofstream output(path);
		if (!output)
		{
			throw std::runtime_error("Unable to write CSV output: " + path.string());
		}
		output << "stage,samples,total_ms,average_ms,min_ms,max_ms,stddev_ms,p50_ms,p90_ms,p95_ms,p99_ms\n";
		output << std::fixed << std::setprecision(6);
		for (const auto &stage : stages)
		{
			const StageStats &stats = statistics.at(stage);
			output << stage << ',' << stats.samples() << ',' << stats.total_ms() << ',';
			const auto write_value = [&](const std::optional<double> &value)
			{
				if (value.has_value())
				{
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
			if (p99.has_value())
			{
				output << *p99;
			}
			output << '\n';
		}
	}

	void output_handler(
		std::filesystem::path output_csv,
		const StageStatsMap &statistics)
	{
		if (!output_csv.empty())
		{
			write_csv(output_csv, statistics);
			std::cout << "\nCSV written to " << output_csv << '\n';
		}
	}

	StageStatsMap run_benchmark_from_images(
		const Options options)
	{
		DirectoryImageReader image_reader(options.input_directory);
		auto images = image_reader.scan_image_paths();

		if (options.max_images != 0 && images.size() > options.max_images)
		{
			images.resize(options.max_images);
		}

		if (images.empty())
		{
			throw std::runtime_error("No Supported images found in:" + options.input_directory.string());
		}

		StageStatsMap statistics;
		bool recording = false;
		const ScopedTimer::Callback record_timing =
			[&statistics, &recording](const std::string &stage, double elapsed_ms)
		{
			if (recording)
			{
				statistics[stage].record(elapsed_ms);
			}
		};
		TritonInferenceClient client(options.triton_url, options.model_name, record_timing);
		if (!client.isServerLive())
		{
			throw std::runtime_error("Triton server is not live at " + options.triton_url);
		}

		const auto run_iteration = [&](const std::filesystem::path &image_path)
		{
			ScopedTimer timer("Total time", record_timing);
			DirectoryImageReader::ImageRecord image_record;
			{
				ScopedTimer timer("image_load", record_timing);
				image_record = DirectoryImageReader::read_single(image_path);
			}
			if (!image_record.is_valid)
			{
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

		for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration)
		{
			run_iteration(images[iteration % images.size()]);
		}

		recording = true;
		for (std::size_t iteration = 0; iteration < options.measurement_iterations; ++iteration)
		{
			run_iteration(images[iteration % images.size()]);
		}
		recording = false;
		output_handler(options.output_csv, statistics);
		return statistics;
	}

	StageStatsMap run_benchmark_from_camera(const Options &options)
	{
		StageStatsMap statistics;
		bool recording = false;
		const ScopedTimer::Callback record_timing =
			[&statistics, &recording](const std::string &stage, double elapsed_ms)
		{
			if (recording)
			{
				statistics[stage].record(elapsed_ms);
			}
		};
		TritonInferenceClient client(options.triton_url, options.model_name, record_timing);
		if (!client.isServerLive())
		{
			throw std::runtime_error("Triton server is not live at " + options.triton_url);
		}

		cv::VideoCapture cap(options.camera_index);
		if (!cap.isOpened())
		{
			throw std::runtime_error("Unable to open camera " + std::to_string(options.camera_index));
		}

		std::cout << "Benchmark configuration\n"
				  << "  model:       " << options.model_name << '\n'
				  << "  server:      " << options.triton_url << '\n'
				  << "  camera:      " << options.camera_index << '\n'
				  << "  warmup:      " << options.warmup_iterations << '\n'
				  << "  iterations:  " << options.measurement_iterations << '\n'
				  << "Webcam active. Press 'q' to exit.\n";

		cv::Mat frame;
		bool should_stop = false;
		const auto run_iteration = [&]()
		{
			{
				ScopedTimer timer("capture", record_timing);
				if (!cap.read(frame) || frame.empty())
				{
					throw std::runtime_error("Unable to capture a non-empty frame");
				}
			}

			std::vector<Detection> detections;
			{
				ScopedTimer timer("inference_pipeline", record_timing);
				detections = client.infer(frame);
			}

			{
				ScopedTimer timer("render", record_timing);
				for (const auto &det : detections)
				{
					cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
					cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
					cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);

					std::string label = "Class " + std::to_string(det.class_id) + " : " +
										std::to_string(det.confidence).substr(0, 4);
					int base_line;
					cv::Size label_size = cv::getTextSize(
						label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
					cv::Point text_origin(pt1.x, std::max(pt1.y - 5, label_size.height));
					cv::rectangle(
						frame,
						text_origin + cv::Point(0, base_line),
						text_origin + cv::Point(label_size.width, -label_size.height),
						cv::Scalar(0, 255, 0),
						cv::FILLED);
					cv::putText(
						frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX,
						0.5, cv::Scalar(0, 0, 0), 1);
				}
			}

			{
				ScopedTimer timer("display", record_timing);
				cv::imshow("Edge Inference - Raw Feed", frame);
				should_stop = cv::waitKey(1) == 'q';
			}
		};

		for (std::size_t iteration = 0;
			 iteration < options.warmup_iterations && !should_stop;
			 ++iteration)
		{
			run_iteration();
		}

		recording = true;
		for (std::size_t iteration = 0;
			 iteration < options.measurement_iterations && !should_stop;
			 ++iteration)
		{
			ScopedTimer timer("Total time", record_timing);
			run_iteration();
		}
		recording = false;

		cap.release();
		cv::destroyAllWindows();
		output_handler(options.output_csv, statistics);
		return statistics;
	}

} // namespace

int main(int argc, char **argv)
{
	try
	{
		const Options options = parse_options(argc, argv);
		StageStatsMap statistics = options.use_camera
									   ? run_benchmark_from_camera(options)
									   : run_benchmark_from_images(options);
		print_summary(statistics);

		return 0;
	}
	catch (const std::exception &error)
	{
		std::cerr << "Benchmark failed: " << error.what() << '\n';
		return 1;
	}
}
