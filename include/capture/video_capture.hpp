#ifndef VIDEO_CAPTURE_HPP
#define VIDEO_CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
#include <atomic>
#include "core/frame_buffer.hpp"
#include "core/types.hpp"

class VideoCapture {
public:
    VideoCapture(int deviceID = 0, int apiID = cv::CAP_ANY,
                 int target_width = 640, int target_height = 480);
    ~VideoCapture();

    // Initialize the camera
    bool initialize();

    // Start capture thread
    void start(FrameBuffer* outputBuffer);

    // Stop capture thread
    void stop();

    // Check if running
    bool isRunning() const;

    // Get statistics
    int getFrameCount() const;

private:
    void captureLoop();
    bool attemptReconnect();

    cv::VideoCapture cap_;
    std::thread captureThread_;
    std::atomic<bool> isRunning_;

    FrameBuffer* outputBuffer_;

    int deviceID_;
    int apiID_;
    int targetWidth_;
    int targetHeight_;

    std::atomic<int> frameCount_;
};

#endif