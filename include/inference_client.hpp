#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declaration to hide gRPC includes from the rest of the app
namespace triton { namespace client { class InferenceServerGrpcClient; } }

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TritonInferenceClient {
public:
    TritonInferenceClient(const std::string& url, const std::string& model_name);
    ~TritonInferenceClient();

    bool isServerLive();
    std::vector<Detection> infer(const cv::Mat& frame);

private:
    std::unique_ptr<triton::client::InferenceServerGrpcClient> client_;
    std::string model_name_;
    
    std::vector<float> preprocess(const cv::Mat& frame);
    std::vector<Detection> postprocess(const float* raw_output, int original_width, int original_height);
};