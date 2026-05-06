#include <iostream>
#include <opencv2/opencv.hpp>
#include "inference_client.hpp"

int main() {
    try {
        std::cout << "Initializing Edge Inference Pipeline..." << std::endl;

        // 1. Initialize Network Client
        std::string triton_url = "localhost:8001";
        TritonInferenceClient client(triton_url, "yolo26n");

        if (client.isServerLive()) {
            std::cout << "SUCCESS: Triton Server is LIVE." << std::endl;
        } else {
            std::cerr << "FATAL: Triton Server is unreachable. Please start Docker." << std::endl;
            return 1;
        }

        // 2. Initialize Hardware
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Fatal Error: Cannot open webcam." << std::endl;
            return 1;
        }

        std::cout << "Webcam active. Press 'q' to exit." << std::endl;
        cv::Mat frame;
        
        // 3. The Real-Time Execution Loop
        while (true) {
            cap.read(frame);
            if (frame.empty()) {
                std::cerr << "Warning: Blank frame grabbed." << std::endl;
                break;
            }

            // === CORE PIPELINE ===
            
            // Execute Inference
            std::vector<Detection> detections = client.infer(frame);
            
            // Draw Bounding Boxes
            for (const auto& det : detections) {
                // Draw Rectangle
                cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
                cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
                cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2); // Green box, thickness 2
                
                // Draw Label and Confidence
                std::string label = "Class " + std::to_string(det.class_id) + " : " + std::to_string(det.confidence).substr(0, 4);
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::Point text_origin(pt1.x, std::max(pt1.y - 5, labelSize.height));
                
                // Background for text visibility
                cv::rectangle(frame, text_origin + cv::Point(0, baseLine), 
                              text_origin + cv::Point(labelSize.width, -labelSize.height), 
                              cv::Scalar(0, 255, 0), cv::FILLED);
                
                // Text
                cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }


            cv::imshow("Edge Inference - Raw Feed", frame);
            
            // Wait 1ms for UI, break if 'q' is pressed
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();
        std::cout << "Pipeline terminated successfully." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
}