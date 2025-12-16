// Test for VideoCapture component - Single-threaded video display
#include "capture/video_capture.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Video Capture Component Test..." << std::endl;

    // Initialize VideoCapture with default camera (ID 0)
    // Resolution: 640x480
    VideoCapture videoCapture(0, cv::CAP_ANY, 640, 480);

    std::cout << "\nInitializing camera..." << std::endl;
    if (!videoCapture.initialize()) {
        std::cerr << "ERROR: Failed to initialize camera" << std::endl;
        return -1;
    }

    // Create window for display
    cv::namedWindow("Video Capture Test", cv::WINDOW_AUTOSIZE);

    // Statistics tracking
    int frameCount = 0;
    int failedFrames = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto lastFPSUpdate = startTime;

    std::cout << "\nStarting video capture loop..." << std::endl;
    std::cout << "Press ESC to exit\n" << std::endl;

    cv::Mat frame;

    // Main capture loop
    while (true) {
        // Capture frame
        bool success = videoCapture.captureFrame(frame);

        if (!success || frame.empty()) {
            std::cerr << "WARNING: Failed to capture frame" << std::endl;
            failedFrames++;

            // If too many consecutive failures, break
            if (failedFrames > 30) {
                std::cerr << "ERROR: Too many failed frames, exiting" << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Reset failed frame counter on success
        failedFrames = 0;
        frameCount++;

        // Calculate and display FPS every second
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastFPSUpdate
        ).count();

        if (elapsed >= 1000) {
            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - startTime
            ).count();

            double fps = frameCount / static_cast<double>(totalElapsed);

            std::cout << "FPS: " << std::fixed << std::setprecision(2) << fps
                      << " | Total Frames: " << frameCount << std::endl;

            lastFPSUpdate = currentTime;
        }

        // Add frame info overlay
        std::string frameInfo = "Frame: " + std::to_string(frameCount);
        cv::putText(frame, frameInfo, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Display frame
        cv::imshow("Video Capture Test", frame);

        // Check for ESC key (27)
        int key = cv::waitKey(1);
        if (key == 27) {
            std::cout << "\nESC pressed - exiting..." << std::endl;
            break;
        }
    }

    // Cleanup
    videoCapture.release();
    cv::destroyAllWindows();

    // Print final statistics
    auto endTime = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(
        endTime - startTime
    ).count();

    double avgFPS = (totalTime > 0) ? frameCount / static_cast<double>(totalTime) : 0;

    std::cout << "TEST COMPLETE" << std::endl;
    std::cout << "Total Frames Captured: " << frameCount << std::endl;
    std::cout << "Total Time: " << totalTime << " seconds" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avgFPS << std::endl;

    return 0;
}
