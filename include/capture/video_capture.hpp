#ifndef VIDEO_CAPTURE_HPP
#define VIDEO_CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include "core/types.hpp"

class VideoCapture {
public:
    VideoCapture(int deviceID = 0, int apiID = cv::CAP_ANY,
                 int target_width = 640, int target_height = 480);
    ~VideoCapture() = default;

    // Initialize the camera
    bool initialize();

    // Capture a single frame (for external thread control)
    bool captureFrame(cv::Mat& frame);

    // Release the camera
    void release();

    // Get statistics
    int getFrameCount() const;

private:
    cv::VideoCapture cap_;

    int deviceID_;
    int apiID_;
    int targetWidth_;
    int targetHeight_;

    std::atomic<int> frameCount_;
};

#endif