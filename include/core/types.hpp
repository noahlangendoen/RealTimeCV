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

// Face BBox Structure
struct FaceBox {
    cv::Rect box;
    float confidence;

    FaceBox() : box(), confidence(0.0f) {}

    FaceBox(const cv::Rect& rect, float conf)
        : box(rect), confidence(conf) {}
};

// Expression Classification Result
struct ExpressionResult {
    std::string label;
    float confidence;
    int classId;

    ExpressionResult() : label("Unknown"), confidence(0.0f), classId(-1) {}

    ExpressionResult(const std::string& lbl, float conf, int id)
        : label(lbl), confidence(conf), classId(id) {}
};

// Structure to hold detected face with metadata
struct DetectedFace {
    cv::Mat faceROI;
    FaceBox bbox;
    int frameId;
    std::chrono::steady_clock::time_point timestamp;

    DetectedFace() : frameId(-1) {}

    bool isValid() const {
        return !faceROI.empty() && frameId >= 0;
    }
};

// Structure to hold classified face with metadata
struct ClassifiedFace {
    FaceBox bbox;
    ExpressionResult expression;
    int frameId;
    std::chrono::steady_clock::time_point timestamp;

    ClassifiedFace() : frameId(-1) {}

    bool isValid() const {
        return frameId >= 0 && !expression.label.empty();
    }
};

// Structure to hold frame with all detected and classified faces
struct ProcessedFrame {
    cv::Mat frame;
    std::vector<ClassifiedFace> faces;
    int frameId;
    std::chrono::steady_clock::time_point timestamp;

    ProcessedFrame () : frameId(-1) {}

    bool isValid() const {
        return !frame.empty() && frameId >= 0;
    }
};

#endif