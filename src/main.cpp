#include <iostream>
#include <opencv2/opencv.hpp>
#include "capture/video_capture.hpp"
#include "core/frame_buffer.hpp"

int main() {
    std::cout << "=== RealTimeCV Webcam Test ===" << std::endl;

    // Create frame buffer (max 10 frames)
    FrameBuffer frameBuffer(10);

    // Create video capture (device 0, default API, 640x480)
    VideoCapture capture(0, cv::CAP_ANY, 640, 480);

    // Initialize camera
    if (!capture.initialize()) {
        std::cerr << "Failed to initialize camera!" << std::endl;
        return -1;
    }

    // Start capture thread
    capture.start(&frameBuffer);

    std::cout << "Press 'q' to quit" << std::endl;

    // Main display loop
    while (true) {
        // Get frame from buffer (with timeout)
        auto frameOpt = frameBuffer.popWithTimeout(1000);  // 1 second timeout

        if (!frameOpt.has_value()) {
            std::cerr << "No frame received (timeout)" << std::endl;
            continue;
        }

        Frame frame = frameOpt.value();

        if (!frame.isValid()) {
            std::cerr << "Invalid frame received" << std::endl;
            continue;
        }

        // Display the frame
        cv::imshow("RealTimeCV - Webcam Test", frame.data);

        // Check for quit key
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {  // 'q' or ESC
            break;
        }
    }

    // Cleanup
    std::cout << "\nShutting down..." << std::endl;
    capture.stop();
    cv::destroyAllWindows();

    std::cout << "Total frames captured: " << capture.getFrameCount() << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}