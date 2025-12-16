// Test for VideoCapture + FaceDetector - Two-threaded pipeline
#include "capture/video_capture.hpp"
#include "detection/face_detector.hpp"
#include "core/frame_buffer.hpp"
#include "core/types.hpp"
#include "core/env_loader.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>

#define ENV_PATH "../.env"

// Global control flag
std::atomic<bool> isRunning(true);

// Statistics
std::atomic<int> capturedFrames(0);
std::atomic<int> processedFrames(0);
std::atomic<int> droppedFrames(0);

// Capture thread function
void captureThreadFunc(VideoCapture* videoCapture, FrameBuffer* buffer, int& frameIdCounter) {
    std::cout << "[Capture Thread] Started" << std::endl;

    cv::Mat frame;
    int localCaptured = 0;
    int localDropped = 0;

    while (isRunning) {
        // Capture frame
        bool success = videoCapture->captureFrame(frame);

        if (!success || frame.empty()) {
            std::cerr << "[Capture Thread] Failed to grab frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Create Frame object
        Frame capturedFrame;
        capturedFrame.data = frame.clone();
        capturedFrame.timestamp = std::chrono::steady_clock::now();
        capturedFrame.frameId = frameIdCounter++;

        // Try to push to buffer (drop if full to avoid blocking)
        if (!buffer->tryPush(capturedFrame)) {
            localDropped++;
            droppedFrames++;
        } else {
            localCaptured++;
            capturedFrames++;
        }
    }

    std::cout << "[Capture Thread] Stopped (Captured: " << localCaptured
              << ", Dropped: " << localDropped << ")" << std::endl;
}

// Detection thread function
void detectionThreadFunc(FaceDetector* faceDetector, FrameBuffer* buffer, float detectionThreshold) {
    std::cout << "[Detection Thread] Started" << std::endl;

    cv::namedWindow("Capture + Detection Test", cv::WINDOW_AUTOSIZE);

    int localProcessed = 0;
    auto lastFPSUpdate = std::chrono::steady_clock::now();
    double currentFPS = 0.0;

    while (isRunning) {
        // Get frame from buffer
        Frame frame = buffer->pop();

        if (!frame.isValid()) {
            // Buffer was shutdown
            continue;
        }

        // Detect faces
        std::vector<FaceBox> faces = faceDetector->detectFaces(frame.data);

        // Filter by confidence threshold
        std::vector<FaceBox> filteredFaces;
        for (const auto& face : faces) {
            if (face.confidence >= detectionThreshold) {
                filteredFaces.push_back(face);
            }
        }

        // Draw bounding boxes
        cv::Mat displayFrame = frame.data.clone();

        for (const auto& face : filteredFaces) {
            // Draw green rectangle around face
            cv::rectangle(displayFrame, face.box, cv::Scalar(0, 255, 0), 2);

            // Draw confidence score
            std::stringstream ss;
            ss << std::fixed << std::setprecision(0) << (face.confidence * 100) << "%";
            std::string confText = ss.str();

            // Background for text
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(confText, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            cv::Point textOrg(face.box.x, face.box.y - 5);
            cv::rectangle(displayFrame,
                         textOrg + cv::Point(0, baseline),
                         textOrg + cv::Point(textSize.width, -textSize.height),
                         cv::Scalar(0, 255, 0), -1);

            // Draw text
            cv::putText(displayFrame, confText, textOrg,
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Calculate FPS
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastFPSUpdate
        ).count();

        if (elapsed >= 1000) {
            currentFPS = processedFrames.load() / (elapsed / 1000.0);
            lastFPSUpdate = currentTime;
            processedFrames = 0;
        }

        // Add info overlay
        std::stringstream info;
        info << "FPS: " << std::fixed << std::setprecision(1) << currentFPS
             << " | Faces: " << filteredFaces.size()
             << " | Frame: " << frame.frameId;

        cv::putText(displayFrame, info.str(), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Display frame
        cv::imshow("Capture + Detection Test", displayFrame);

        // Check for ESC key
        int key = cv::waitKey(1);
        if (key == 27) {
            std::cout << "\n[Detection Thread] ESC pressed - stopping..." << std::endl;
            isRunning = false;
            break;
        }

        localProcessed++;
        processedFrames++;
    }

    cv::destroyAllWindows();

    std::cout << "[Detection Thread] Stopped (Processed: " << localProcessed << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Two-Threaded Pipeline Test..." << std::endl;
    std::cout << "Capture Thread + Detection Thread" << std::endl;

    // Load environment variables
    EnvLoader env;
    if (!env.load(ENV_PATH)) {
        std::cerr << "WARNING: Failed to load .env file, using defaults" << std::endl;
    }

    // Configuration
    std::string modelPath = env.get("FACE_DETECTOR", "../models/yolov8n-face.onnx");
    int cameraId = 0;
    int captureWidth = 640;
    int captureHeight = 480;
    float detectionThreshold = 0.5f;
    int bufferSize = 3;

    // Allow optional parameters
    if (argc >= 2) cameraId = std::atoi(argv[1]);
    if (argc >= 3) detectionThreshold = std::atof(argv[2]);

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model Path: " << modelPath << std::endl;
    std::cout << "  Camera ID: " << cameraId << std::endl;
    std::cout << "  Resolution: " << captureWidth << "x" << captureHeight << std::endl;
    std::cout << "  Detection Threshold: " << detectionThreshold << std::endl;
    std::cout << "  Buffer Size: " << bufferSize << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize VideoCapture
    std::cout << "\n[Main] Initializing VideoCapture..." << std::endl;
    VideoCapture videoCapture(cameraId, cv::CAP_ANY, captureWidth, captureHeight);

    if (!videoCapture.initialize()) {
        std::cerr << "[Main] ERROR: Failed to initialize camera" << std::endl;
        return -1;
    }

    // Initialize FaceDetector
    std::cout << "[Main] Initializing FaceDetector..." << std::endl;
    FaceDetector faceDetector(modelPath);

    if (!faceDetector.initialize()) {
        std::cerr << "[Main] ERROR: Failed to initialize face detector" << std::endl;
        videoCapture.release();
        return -1;
    }

    // Create frame buffer
    FrameBuffer buffer(bufferSize);

    std::cout << "[Main] All components initialized successfully" << std::endl;
    std::cout << "[Main] Starting threads..." << std::endl;
    std::cout << "\nPress ESC in the window to stop\n" << std::endl;

    // Start threads
    int frameIdCounter = 0;
    auto startTime = std::chrono::steady_clock::now();

    std::thread captureThread(captureThreadFunc, &videoCapture, &buffer, std::ref(frameIdCounter));
    std::thread detectionThread(detectionThreadFunc, &faceDetector, &buffer, detectionThreshold);

    std::cout << "[Main] Threads started" << std::endl;

    // Wait for threads to finish
    captureThread.join();

    // Shutdown buffer to wake up detection thread
    buffer.shutdown();

    detectionThread.join();

    // Cleanup
    videoCapture.release();

    // Calculate statistics
    auto endTime = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(
        endTime - startTime
    ).count();

    std::cout << "TEST COMPLETE" << std::endl;
    std::cout << "Total Runtime: " << totalTime << " seconds" << std::endl;
    std::cout << "Frames Captured: " << capturedFrames.load() << std::endl;
    std::cout << "Frames Dropped: " << droppedFrames.load() << std::endl;

    return 0;
}
