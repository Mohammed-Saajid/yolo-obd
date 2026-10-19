#include "inference_client.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <utility>

TritonInferenceClient::TritonInferenceClient(
    const std::string& url,
    const std::string& model_name,
    float* d_input_tensor_vram,
    size_t tensor_byte_size,
    ScopedTimer::Callback timer_callback
)
    : model_name_(model_name),
      d_input_tensor_vram_(d_input_tensor_vram),
      tensor_byte_size_(tensor_byte_size),
      timer_callback_(std::move(timer_callback)) {

    // 1. Initialize Triton gRPC Client Handle
    triton::client::Error err = triton::client::InferenceServerGrpcClient::Create(&client_, url);
    if (!err.IsOk()) {
        std::cerr << "Failed to create Triton client: " << err.Message() << std::endl;
        throw std::runtime_error("Triton Client Initialization Failed");
    }

    // 2. Unregister stale CUDA shared memory region if left over from prior runs
    client_->UnregisterCudaSharedMemory(shm_region_name_);

    // 3. Register the CUDA VRAM pointer as shared memory in Triton (Zero-Copy)
    cudaIpcMemHandle_t cuda_handle{};
    const cudaError_t cuda_err = cudaIpcGetMemHandle(&cuda_handle, d_input_tensor_vram_);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Failed to create CUDA IPC memory handle: "
                  << cudaGetErrorString(cuda_err) << std::endl;
        throw std::runtime_error("CUDA IPC Memory Handle Creation Failed");
    }

    err = client_->RegisterCudaSharedMemory(
        shm_region_name_,
        cuda_handle,
        0, // Device ID (NVIDIA GPU 0)
        tensor_byte_size_
    );

    if (!err.IsOk()) {
        std::cerr << "Failed to register CUDA Shared Memory with Triton: " << err.Message() << std::endl;
        throw std::runtime_error("CUDA Shared Memory Registration Failed");
    }
}

TritonInferenceClient::~TritonInferenceClient() {
    if (client_) {
        // Clean up registered shared memory region on server shutdown
        client_->UnregisterCudaSharedMemory(shm_region_name_);
    }
}

bool TritonInferenceClient::isServerLive() {
    bool live = false;
    triton::client::Error err = client_->IsServerLive(&live);
    if (!err.IsOk()) {
        std::cerr << "Error communicating with server: " << err.Message() << std::endl;
        return false;
    }
    return live;
}

std::vector<Detection> TritonInferenceClient::infer(
    float* d_input_tensor_vram,
    int original_width,
    int original_height,
    cudaStream_t stream
) {
    // Synchronize CUDA stream to ensure the GPU preprocessing kernel has completed
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    // 1. Create InferInput binding to registered CUDA Shared Memory
    std::vector<int64_t> shape = {1, 3, 640, 640};
    triton::client::InferInput* input{nullptr};
    triton::client::Error err = triton::client::InferInput::Create(&input, "images", shape, "FP32");
    if (!err.IsOk()) throw std::runtime_error("Failed to create input binding: " + err.Message());

    err = input->SetSharedMemory(shm_region_name_, tensor_byte_size_);
    if (!err.IsOk()) {
        delete input;
        throw std::runtime_error("Failed to bind CUDA Shared Memory: " + err.Message());
    }

    // 2. Request output tensor
    triton::client::InferRequestedOutput* output{nullptr};
    err = triton::client::InferRequestedOutput::Create(&output, "output0");
    if (!err.IsOk()) {
        delete input;
        throw std::runtime_error("Failed to create output descriptor: " + err.Message());
    }

    triton::client::InferOptions options(model_name_);
    std::vector<triton::client::InferInput*> inputs = {input};
    std::vector<const triton::client::InferRequestedOutput*> outputs = {output};

    // 3. Issue zero-copy inference request to Triton
    triton::client::InferResult* result{nullptr};
    {
        ScopedTimer timer("triton_inference", timer_callback_);
        err = client_->Infer(&result, options, inputs, outputs);
    }

    delete input;
    delete output;

    if (!err.IsOk()) {
        if (result) delete result;
        throw std::runtime_error("Triton zero-copy inference failed: " + err.Message());
    }

    // 4. Retrieve raw model output predictions
    const float* raw_output{nullptr};
    size_t output_byte_size{0};
    err = result->RawData("output0", reinterpret_cast<const uint8_t**>(&raw_output), &output_byte_size);
    if (!err.IsOk()) {
        delete result;
        throw std::runtime_error("Failed to parse output data: " + err.Message());
    }

    // 5. Run postprocessing directly on predicted boxes
    std::vector<Detection> detections;
    {
        ScopedTimer timer("postprocess", timer_callback_);
        detections = postprocess(raw_output, original_width, original_height);
    }

    delete result;
    return detections;
}

std::vector<Detection> TritonInferenceClient::postprocess(
    const float* raw_output,
    int original_width,
    int original_height
) {
    std::vector<Detection> detections;
    const int num_boxes = 300;
    const int elements_per_box = 6;
    const float confidence_threshold = 0.50f;

    const float scale_x = static_cast<float>(original_width) / 640.0f;
    const float scale_y = static_cast<float>(original_height) / 640.0f;

    for (int i = 0; i < num_boxes; ++i) {
        int index = i * elements_per_box;
        float confidence = raw_output[index + 4];

        if (confidence >= confidence_threshold) {
            Detection det;
            det.x1 = raw_output[index + 0] * scale_x;
            det.y1 = raw_output[index + 1] * scale_y;
            det.x2 = raw_output[index + 2] * scale_x;
            det.y2 = raw_output[index + 3] * scale_y;
            det.confidence = confidence;
            det.class_id = static_cast<int>(raw_output[index + 5]);
            detections.push_back(det);
        }
    }
    return detections;
}