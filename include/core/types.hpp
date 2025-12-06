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

// Bounding Box Structure
struct BBox {
    float x;            // Top X coordinate
    float y;            // Top Y coordinate
    float width;        // Width of box
    float height;       // Height of box
    float confidence;   // Detection confidence (bound 0 to 1)

    BBox() : x(0), y(0), width(0), height(0), confidence(0.0f) {}

    BBox(float x_, float y_, float width_, float height_, float conf = 1.0f)
        : x(x_), y(y_), width(width_), height(height_), confidence(conf) {}

    // Convert to cv::Rect for OpenCV Operations
    cv::Rect toRect() const {
        return cv::Rect(static_cast<int>(x), static_cast<int>(y),
                        static_cast<int>(width), static_cast<int>(height));
    }

    // Get center point
    cv::Point2f center() const {
        return cv::Point2f(x + width / 2.0f, y + height / 2.0f);
    }
};

// Face detection result
struct DetectionResult {
    BBox bbox;
    cv::Mat faceROI;
    int frameId;
    std::chrono::steady_clock::time_point timestamp;

    DetectionResult() : frameId(-1) {}

    bool isValid() const {
        return !faceROI.empty();
    }
};

struct FaceBox {
    cv::Rect box;
    float confidence;
};

#endif