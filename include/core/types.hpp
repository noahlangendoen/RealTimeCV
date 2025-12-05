// include/core/types.hpp
#ifndef TYPES_HPP
#define TYPES_HPP

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <string>

// Frame structure with metadata
struct Frame {
    cv::Mat data;
    std::chrono::steady_clock::time_point timestamp;
    int frameId;

    // Default constructor
    Frame() : frameId(-1) {}

    // Constructor with data
    Frame(const cv::Mat& img, int id = -1)
        : data(img)
        , timestamp(std::chrono::steady_clock::now())
        , frameId(id)
    {}

    // Check if frame is valid
    bool isValid() const {
        return !data.empty();
    }
};

#endif