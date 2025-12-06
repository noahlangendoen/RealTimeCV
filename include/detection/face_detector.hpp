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

    // Detect Faces within Frame
    // std::vector<DetectionResult> detect(const Frame& frame);

    // Detect faces in a cv::Mat
    std::vector<FaceBox> detectFaces(const cv::Mat& frame);

    // Get model info
    // bool isInitialized() const { return isInitialized_; }
    // std::string getModelPath() const { return modelPath_; }

private:
    // Preprocessing
    // cv::Mat preprocessImage(const cv::Mat& image);

    // // Postprocessing
    // std::vector<BBox> postprocess(float* output, int outputSize,
    //                               int originalWidth, int originalHeight);

    // // Non-Maximum Supression
    // std::vector<BBox> nonMaximumSuppression(std::vector<BBox>& boxes, float iouThreshold);

    // // Calculate IoU
    // float calculateIoU(const BBox& box1, const BBox& box2);

    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    // std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    // Ort::AllocatorWithDefaultOptions allocator_;

    // Model metadata
    std::string modelPath_;
    // std::vector<const char*> inputNames_;
    // std::vector<const char*> outputNames_;
    // std::vector<int64_t> inputShape_;

    // // Detection Parameters
    // float confThreshold_;
    // float iouThreshold_;
    // int inputWidth_;
    // int inputHeight_;

    bool isInitialized_;
};

#endif // FACE_DETECTOR_HPP