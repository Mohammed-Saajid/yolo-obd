#include "native_v4l2_camera.hpp"
#include "yuyv_preprocess.h"
#include <iostream>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <stdio.h>

struct V4L2Buffer {
    void* start{nullptr};
    size_t length{0};
    void* cuda_mapped_ptr{nullptr};
};

NativeV4L2Camera::NativeV4L2Camera(const std::string& device, std::uint32_t width, std::uint32_t height)
    : dev_name_(device), width_(width), height_(height) {}

NativeV4L2Camera::~NativeV4L2Camera() {
    stop();
    closeDevice();
}

bool NativeV4L2Camera::init() {
    fd_ = open(dev_name_.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
        perror("V4L2: Failed to open video device");
        return false;
    }

    // Query device capabilities
    v4l2_capability cap{};
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0 || !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        perror("V4L2: Device does not support single-planar video capture");
        return false;
    }

    // Set video format (Prefer uncompressed YUYV for GPU/CUDA pipelines, fallback to MJPEG if needed)
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; 
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        // Fallback attempt with MJPEG if camera rejects raw YUYV
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            perror("V4L2: Failed to set video format (YUYV and MJPEG rejected)");
            return false;
        }
    }

    pixel_format_ = fmt.fmt.pix.pixelformat;

    // Request kernel ring buffers (MMAP mode)
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        perror("V4L2: Insufficient buffer memory allocated");
        return false;
    }

    // Map kernel buffers to user space and register directly with CUDA (Zero-Copy)
    buffers_.resize(req.count);
    for (size_t i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<std::uint32_t>(i);

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("V4L2: Failed to query buffer");
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, 
                                 MAP_SHARED, fd_, buf.m.offset);

        if (buffers_[i].start == MAP_FAILED) {
            perror("V4L2: mmap failed");
            return false;
        }

        // Zero-copy import: Register mapped host buffer into CUDA virtual memory address space
        cudaError_t err = cudaHostRegister(buffers_[i].start, buffers_[i].length, cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: Failed to register V4L2 buffer with cudaHostRegister: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Get device pointer accessible by GPU kernels over PCIe
        err = cudaHostGetDevicePointer(&buffers_[i].cuda_mapped_ptr, buffers_[i].start, 0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: Failed to retrieve mapped device pointer: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
    }

    return true;
}

bool NativeV4L2Camera::start() {
    // Queue all buffers to start recording
    for (size_t i = 0; i < buffers_.size(); ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = static_cast<std::uint32_t>(i);
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            perror("V4L2: Failed to queue buffer");
            return false;
        }
    }

    // Start video streaming on capture device
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        perror("V4L2: Failed to start streaming (VIDIOC_STREAMON)");
        return false;
    }

    is_running_ = true;
    worker_thread_ = std::thread(&NativeV4L2Camera::captureLoop, this);
    return true;
}

bool NativeV4L2Camera::getLatestFrame(cv::Mat& out_mat) {
    // Atomically exchange consumer slot with the shared slot
    int latest_idx = shared_buffer_idx_.exchange(-1, std::memory_order_acq_rel);

    // No new frame arrived since last call
    if (latest_idx < 0) {
        if (consumer_idx_ < 0) return false; // Never received any frame yet
        // Keep using active consumer_idx_ if we allow reading current frame
    } else {
        // Re-queue the previous consumer buffer back to V4L2 driver
        if (consumer_idx_ >= 0 && consumer_idx_ != latest_idx) {
            v4l2_buffer old_buf{};
            old_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            old_buf.memory = V4L2_MEMORY_MMAP;
            old_buf.index = static_cast<uint32_t>(consumer_idx_);
            ioctl(fd_, VIDIOC_QBUF, &old_buf);
        }
        consumer_idx_ = latest_idx;
    }

    size_t frame_bytes = shared_frame_bytes_.load(std::memory_order_acquire);
    void* current_buf = buffers_[consumer_idx_].start;

    // Decode/Process directly without holding any mutex
    if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
        cv::Mat raw_yuyv(height_, width_, CV_8UC2, current_buf);
        cv::cvtColor(raw_yuyv, out_mat, cv::COLOR_YUV2BGR_YUYV);
    } else if (pixel_format_ == V4L2_PIX_FMT_MJPEG) {
        cv::Mat raw_mjpeg(1, static_cast<int>(frame_bytes), CV_8UC1, current_buf);
        out_mat = cv::imdecode(raw_mjpeg, cv::IMREAD_COLOR);
    }

    return !out_mat.empty();
}

bool NativeV4L2Camera::getLatestFrameToGPU(float* d_output_tensor_vram, cudaStream_t stream) {
    // Atomically retrieve current ready buffer index
    int latest_idx = shared_buffer_idx_.exchange(-1, std::memory_order_acq_rel);

    if (latest_idx >= 0) {
        // Return previous buffer slot to V4L2 driver
        if (consumer_idx_ >= 0 && consumer_idx_ != latest_idx) {
            v4l2_buffer old_buf{};
            old_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            old_buf.memory = V4L2_MEMORY_MMAP;
            old_buf.index = static_cast<uint32_t>(consumer_idx_);
            ioctl(fd_, VIDIOC_QBUF, &old_buf);
        }
        consumer_idx_ = latest_idx;
    }

    if (consumer_idx_ < 0) return false;

    // Retrieve GPU mapped pointer for zero-copy PCIe reading
    void* d_src_yuyv = buffers_[consumer_idx_].cuda_mapped_ptr;

    // Launch CUDA preprocessing kernel asynchronously
    launch_yuyv_preprocess(
        d_src_yuyv,
        d_output_tensor_vram,
        static_cast<int>(width_),
        static_cast<int>(height_),
        640,
        640,
        stream);

    return true;
}

void NativeV4L2Camera::captureLoop() {
    while (is_running_) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        timeval tv{1, 0};
        if (select(fd_ + 1, &fds, NULL, NULL, &tv) <= 0) continue;

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // Dequeue frame written by hardware
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;
            break;
        }

        // New frame slot ready from hardware
        producer_idx_ = static_cast<int>(buf.index);
        shared_frame_bytes_.store(buf.bytesused, std::memory_order_release);

        // Atomically publish producer_idx and retrieve the old shared slot
        int old_shared = shared_buffer_idx_.exchange(producer_idx_, std::memory_order_acq_rel);

        // If an old buffer was returned and it's not currently held by the consumer,
        // re-queue it back to the V4L2 kernel ring
        if (old_shared >= 0 && old_shared != consumer_idx_) {
            v4l2_buffer old_buf{};
            old_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            old_buf.memory = V4L2_MEMORY_MMAP;
            old_buf.index = static_cast<uint32_t>(old_shared);
            ioctl(fd_, VIDIOC_QBUF, &old_buf);
        }
    }
}

void NativeV4L2Camera::stop() {
    if (!is_running_) return;

    is_running_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
}

void NativeV4L2Camera::closeDevice() {
    for (auto& b : buffers_) {
        if (b.start && b.start != MAP_FAILED) {
            cudaHostUnregister(b.start);
            munmap(b.start, b.length);
            b.start = nullptr;
        }
    }
    buffers_.clear();

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}
