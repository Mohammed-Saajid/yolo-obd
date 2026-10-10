#include "inference_client.hpp"
#include "grpc_client.h"
#include <iostream>
#include <stdexcept>
#include <utility>

TritonInferenceClient::TritonInferenceClient(
    const std::string& url,
    const std::string& model_name,
    ScopedTimer::Callback timer_callback
)
    : model_name_(model_name),
      timer_callback_(std::move(timer_callback)) {
    
    // Modern Triton API (v2.41+) requires passing the address of the unique_ptr directly
    triton::client::Error err = triton::client::InferenceServerGrpcClient::Create(&client_, url);
    
    if (!err.IsOk()) {
        std::cerr << "Failed to create Triton client: " << err.Message() << std::endl;
        throw std::runtime_error("Triton Client Initialization Failed");
    }
}

TritonInferenceClient::~TritonInferenceClient() = default;

bool TritonInferenceClient::isServerLive() {
    bool live = false;
    triton::client::Error err = client_->IsServerLive(&live);
    
    if (!err.IsOk()) {
        std::cerr << "Error communicating with server: " << err.Message() << std::endl;
        return false;
    }
    return live;
}

std::vector<float> TritonInferenceClient::preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb, float_img;
    cv::resize(frame, resized, cv::Size(640, 640));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    std::vector<float> input_tensor(3 * 640 * 640);
    std::vector<cv::Mat> chw_channels;
    for (int i = 0; i < 3; ++i) {
        chw_channels.push_back(cv::Mat(640, 640, CV_32FC1, input_tensor.data() + i * 640 * 640));
    }
    cv::split(float_img, chw_channels);
    return input_tensor;
}

std::vector<Detection> TritonInferenceClient::infer(const cv::Mat& frame) {
    std::vector<float> input_data;
    {
        ScopedTimer timer("preprocess", timer_callback_);
        input_data = preprocess(frame);
    }
    std::vector<int64_t> shape = {1, 3, 640, 640};
    
    triton::client::InferInput* input;
    triton::client::Error err = triton::client::InferInput::Create(&input, "images", shape, "FP32");
    if (!err.IsOk()) throw std::runtime_error("Failed to create input: " + err.Message());
    
    size_t byte_size = input_data.size() * sizeof(float);
    err = input->AppendRaw(reinterpret_cast<const uint8_t*>(input_data.data()), byte_size);
    if (!err.IsOk()) throw std::runtime_error("Failed to set input data: " + err.Message());

    triton::client::InferRequestedOutput* output;
    err = triton::client::InferRequestedOutput::Create(&output, "output0");
    if (!err.IsOk()) throw std::runtime_error("Failed to create output: " + err.Message());

    triton::client::InferOptions options(model_name_);
    std::vector<triton::client::InferInput*> inputs = {input};
    std::vector<const triton::client::InferRequestedOutput*> outputs = {output};
    
    triton::client::InferResult* result;
    {
        ScopedTimer timer("triton_inference", timer_callback_);
        err = client_->Infer(&result, options, inputs, outputs);
    }
    if (!err.IsOk()) throw std::runtime_error("Inference failed: " + err.Message());

    const float* raw_output;
    size_t output_byte_size;
    err = result->RawData("output0", reinterpret_cast<const uint8_t**>(&raw_output), &output_byte_size);
    if (!err.IsOk()) throw std::runtime_error("Failed to read output: " + err.Message());

    std::vector<Detection> detections;
    {
        ScopedTimer timer("postprocess", timer_callback_);
        detections = postprocess(raw_output, frame.cols, frame.rows);
    }
    
    delete input;
    delete output;
    delete result;

    return detections;
}

std::vector<Detection> TritonInferenceClient::postprocess(const float* raw_output, int original_width, int original_height) {
    std::vector<Detection> detections;
    int num_boxes = 300;
    int elements_per_box = 6;
    float confidence_threshold = 0.50f;

    float scale_x = static_cast<float>(original_width) / 640.0f;
    float scale_y = static_cast<float>(original_height) / 640.0f;

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