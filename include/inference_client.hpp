#pragma once

#include "grpc_client.h"
#include "scoped_timer.hpp"

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <functional>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TritonInferenceClient {
public:
    TritonInferenceClient(
        const std::string& url,
        const std::string& model_name,
        float* d_input_tensor_vram,
        size_t tensor_byte_size,
        ScopedTimer::Callback timer_callback = nullptr
    );

    ~TritonInferenceClient();

    bool isServerLive();

    // Runs zero-copy inference directly using the GPU preprocessed input tensor
    std::vector<Detection> infer(
        float* d_input_tensor_vram,
        int original_width,
        int original_height,
        cudaStream_t stream = nullptr
    );

private:
    std::vector<Detection> postprocess(const float* raw_output, int original_width, int original_height);

    std::string model_name_;
    std::string shm_region_name_{"input_tensor_cuda_shm"};
    ScopedTimer::Callback timer_callback_;
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client_;
    
    float* d_input_tensor_vram_{nullptr};
    size_t tensor_byte_size_{0};
};