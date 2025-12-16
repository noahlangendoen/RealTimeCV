// src/capture/video_capture.cpp
#include "capture/video_capture.hpp"
#include <iostream>
#include <chrono>

VideoCapture::VideoCapture(int deviceID, int apiID, int targetWidth, int targetHeight)
    : deviceID_(deviceID)
    , apiID_(apiID)
    , targetWidth_(targetWidth)
    , targetHeight_(targetHeight)
    , frameCount_(0)
{
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

void VideoCapture::release() {
    if (cap_.isOpened()) {
        cap_.release();
        std::cout << "Camera released (Total frames: " << frameCount_ << ")" << std::endl;
    }
}

int VideoCapture::getFrameCount() const {
    return frameCount_;
}