#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <opencv2/opencv.hpp>
#include "inference_client.hpp"
#include "scoped_timer.hpp"
#include "stage_stats.hpp"



int main() {
    try {
        std::cout << "Initializing Edge Inference Pipeline..." << std::endl;
        StageStatsMap stage_stats = StageStatsMap();
        
        const ScopedTimer::Callback record_timing =
            [&stage_stats](const std::string& stage, double elapsed_ms) {
                stage_stats[stage].record(elapsed_ms);
            };

        // Initialize Network Client
        const std::string triton_url = "localhost:8001";
        TritonInferenceClient client(triton_url, "yolo26n", record_timing);

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
            {
                ScopedTimer timer("capture", record_timing);
                if (!cap.read(frame) || frame.empty()) {
                    std::cerr << "Warning: Blank frame grabbed." << std::endl;
                    break;
                }
            }

            std::vector<Detection> detections;
            {
                ScopedTimer timer("inference_pipeline", record_timing);
                detections = client.infer(frame);
            }

            {
                ScopedTimer timer("render", record_timing);
                for (const auto& det : detections) {
                    cv::Point pt1(static_cast<int>(det.x1), static_cast<int>(det.y1));
                    cv::Point pt2(static_cast<int>(det.x2), static_cast<int>(det.y2));
                    cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);

                    std::string label = "Class " + std::to_string(det.class_id) + " : " +
                        std::to_string(det.confidence).substr(0, 4);
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(
                        label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine
                    );
                    cv::Point text_origin(pt1.x, std::max(pt1.y - 5, labelSize.height));
                    cv::rectangle(
                        frame,
                        text_origin + cv::Point(0, baseLine),
                        text_origin + cv::Point(labelSize.width, -labelSize.height),
                        cv::Scalar(0, 255, 0),
                        cv::FILLED
                    );
                    cv::putText(
                        frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 0), 1
                    );
                }
            }

            int key;
            {
                ScopedTimer timer("display", record_timing);
                cv::imshow("Edge Inference - Raw Feed", frame);
                key = cv::waitKey(1);
            }

            if (key == 'q') {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

       print_summary(stage_stats);
        

        std::cout << "Pipeline terminated successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}