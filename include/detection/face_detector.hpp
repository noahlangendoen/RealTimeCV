// include/detection/face_dector.hpp
#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "core/types.hpp"


class FaceDetector {
public:
    explicit FaceDetector(const std::string& model_path);
    ~FaceDetector();

    // Initialize ONNX Model
    bool initialize();

    // Detect faces in a cv::Mat
    std::vector<FaceBox> detectFaces(const cv::Mat& frame);


private:
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    std::string modelPath_;

    bool isInitialized_;
};

#endif // FACE_DETECTOR_HPP