// src/capture/video_capture.cpp
#include "capture/video_capture.hpp"
#include <iostream>
#include <chrono>

VideoCapture::VideoCapture(int deviceID, int apiID, int targetWidth, int targetHeight)
    : deviceID_(deviceID)
    , apiID_(apiID)
    , targetWidth_(targetWidth)
    , targetHeight_(targetHeight)
    , isRunning_(false)
    , frameCount_(0)
{ 
}

VideoCapture::~VideoCapture() {
    stop();
}

bool VideoCapture::initialize() {
    cap_.open(deviceID_, apiID_);

    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Unable to open camera (Device ID: " << deviceID_ << ")" << std::endl;
        return false;
    }

    // Set target resolution if specified
    if (targetWidth_ > 0 && targetHeight_ > 0) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, targetWidth_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, targetHeight_);
    }

    // Get actual resolution
    int actualWidth = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int actualHeight = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap_.get(cv::CAP_PROP_FPS);

    std::cout << "Camera Initialized: " << std::endl;
    std::cout << "  Resolution: " << actualWidth << "x" << actualHeight << std::endl;
    std::cout << "  FPS: " << fps << std::endl;

    return true;
}

bool VideoCapture::captureFrame(cv::Mat& frame) {
    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Camera not initialized. Call initialize() first." << std::endl;
        return false;
    }

    // Capture frame directly
    bool success = cap_.read(frame);

    if (success && !frame.empty()) {
        frameCount_++;
        return true;
    }

    return false;
}

void VideoCapture::start(FrameBuffer* outputBuffer) {
    if (isRunning_) {
        std::cerr << "WARNING: Capture thread is already running" << std::endl;
        return;
    }

    if (!cap_.isOpened()) {
        std::cerr << "ERROR: Capture thread is not initialized. Call initialize() first." << std::endl;
        return;
    }

    outputBuffer_ = outputBuffer;
    isRunning_ = true;
    captureThread_ = std::thread(&VideoCapture::captureLoop, this);

    std::cout << "Capture thread started" << std::endl;
}

void VideoCapture::stop() {
    if (!isRunning_) {
        return;
    }

    isRunning_ = false;

    if (captureThread_.joinable()){
        captureThread_.join();
    }

    if (cap_.isOpened()) {
        cap_.release();
    }

    std::cout << "Capture thread stopped (Total frames: " << frameCount_ << ")" << std::endl;
}

bool VideoCapture::isRunning() const {
    return isRunning_;
}

int VideoCapture::getFrameCount() const {
    return frameCount_;
}

void VideoCapture::captureLoop() {
    cv::Mat frame;
    auto lastLog = std::chrono::steady_clock::now();
    int framesThisSecond = 0;

    while(isRunning_) {
        // Capture frame
        bool success = cap_.read(frame);

        if (!success || frame.empty()) {
            std::cerr << "ERROR: Failed to grab frame" << std::endl;

            // Attempt to recover
            if (!attemptReconnect()) {
                isRunning_ = false;
                break;
            }
            continue;
        }
    
        // Create Frame object with timestamp
        Frame capturedFrame;
        capturedFrame.data = frame.clone();
        capturedFrame.timestamp = std::chrono::steady_clock::now();
        capturedFrame.frameId = frameCount_++;

        // Push to output buffer (thread-safe)
        if (outputBuffer_) {
            outputBuffer_->push(capturedFrame);
        }

        framesThisSecond++;

        // Log FPS every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastLog);
        if (elapsed.count() >= 1) {
            std::cout << "[Capture] FPS: " << framesThisSecond << std::endl;
            framesThisSecond = 0;
            lastLog = now;
        }

        // Can add small sleep here to prevent busy-waiting if wanted to adjsut based on target fps
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Capture loop exited" << std::endl;
}

bool VideoCapture::attemptReconnect() {
    std::cerr << "Attempting to reconnect camera..." << std::endl;

    cap_.release();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    cap_.open(deviceID_, apiID_);

    if (cap_.isOpened()) {
        std::cout << "Camera reconnected successfully" << std::endl;
        return true;
    }

    std::cerr << "Camera failed to reconnect" << std::endl;
    return false;
}