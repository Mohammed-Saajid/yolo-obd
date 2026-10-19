#pragma once

#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <vector>
// Forward-declaration to avoid including heavy OpenCV headers in the public interface
namespace cv {
    class Mat;
}

struct V4L2Buffer;

class NativeV4L2Camera {
public:
    NativeV4L2Camera(const std::string& device, std::uint32_t width, std::uint32_t height);
    ~NativeV4L2Camera();

    // Disable copy constructors for RAII safety
    NativeV4L2Camera(const NativeV4L2Camera&) = delete;
    NativeV4L2Camera& operator=(const NativeV4L2Camera&) = delete;

    bool init();
    bool start();
    void stop();
    bool getLatestFrame(cv::Mat& out_mat);
    bool getLatestFrameToGPU(float* d_output_tensor_vram, cudaStream_t stream);

private:
    void captureLoop();
    void closeDevice();

    std::string dev_name_;
    std::uint32_t width_;
    std::uint32_t height_;
    int fd_{-1};
    
    std::atomic<bool> is_running_{false};
    std::thread worker_thread_;

    std::vector<V4L2Buffer> buffers_;
    size_t buffer_count_{0};
    int active_buffer_index_{-1};
    void* cuda_pinned_buffer_{nullptr};
    size_t latest_frame_size_{0};
    std::mutex frame_mutex_;
    unsigned int pixel_format_;

    // Atomic index pointing to the latest ready buffer
    std::atomic<int> shared_buffer_idx_{-1};
    std::atomic<size_t> shared_frame_bytes_{0};

    // Thread-local buffer indices tracking state
    int producer_idx_{-1};
    int consumer_idx_{-1};
};