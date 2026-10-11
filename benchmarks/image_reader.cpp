#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <future>
#include <system_error>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class DirectoryImageReader {
public:
    struct ImageRecord {
        std::string filename;
        fs::path filepath;
        cv::Mat image;
        bool is_valid{false};
    };

    explicit DirectoryImageReader(fs::path directory_path)
        : dir_path_(std::move(directory_path)) {}

    // Collects all valid image paths from directory without loading pixels into RAM
    [[nodiscard]] std::vector<fs::path> scan_image_paths() const {
        std::vector<fs::path> image_paths;

        std::error_code ec;
        if (!fs::exists(dir_path_, ec) || !fs::is_directory(dir_path_, ec)) {
            std::cerr << "[ERROR] Invalid directory path: " << dir_path_ << "\n";
            return image_paths;
        }

        for (const auto& entry : fs::directory_iterator(dir_path_, fs::directory_options::skip_permission_denied, ec)) {
            if (entry.is_regular_file(ec) && is_image_extension(entry.path())) {
                image_paths.push_back(entry.path());
            }
        }

        // Sort pathnames to ensure deterministic ordering across OS/filesystems
        std::sort(image_paths.begin(), image_paths.end());
        return image_paths;
    }

    // Read a single image synchronously
    [[nodiscard]] static ImageRecord read_single(const fs::path& path, cv::ImreadModes mode = cv::IMREAD_COLOR) {
        ImageRecord record;
        record.filepath = path;
        record.filename = path.filename().string();
        
        // cv::imread allocates memory for Mat data array internally
        record.image = cv::imread(path.string(), mode);
        record.is_valid = !record.image.empty();

        if (!record.is_valid) {
            std::cerr << "[WARNING] Failed to decode image: " << path << "\n";
        }

        return record;
    }

    // Read images concurrently using worker threads
    [[nodiscard]] std::vector<ImageRecord> read_batch_parallel(
        const std::vector<fs::path>& paths,
        size_t start_idx,
        size_t batch_size,
        cv::ImreadModes mode = cv::IMREAD_COLOR) const 
    {
        const size_t actual_batch_size = std::min(batch_size, paths.size() - start_idx);
        std::vector<ImageRecord> batch(actual_batch_size);
        std::vector<std::future<void>> futures;
        futures.reserve(actual_batch_size);

        for (size_t i = 0; i < actual_batch_size; ++i) {
            const auto& path = paths[start_idx + i];
            futures.push_back(std::async(std::launch::async, [&batch, i, path, mode]() {
                batch[i] = read_single(path, mode);
            }));
        }

        for (auto& fut : futures) {
            fut.wait();
        }

        return batch;
    }

private:
    fs::path dir_path_;

    static bool is_image_extension(const fs::path& path) {
        static const std::vector<std::string> valid_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"
        };
        
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        return std::find(valid_extensions.begin(), valid_extensions.end(), ext) != valid_extensions.end();
    }
};

