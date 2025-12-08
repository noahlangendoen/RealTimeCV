// include/classification/expression_classifier.hpp
#ifndef EXPRESSION_CLASSIFIER_HPP
#define EXPRESSION_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "core/types.hpp"


class ExpressionClassifier {
public:
    ExpressionClassifier(const std::string& modelPath);
    ~ExpressionClassifier() = default;
    bool initialize();
    ExpressionResult classify(const cv::Mat& faceROI);

private:
    cv::Mat preprocessFace(const cv::Mat& face);

    std::string modelPath_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    bool isInitialized_;

    // Expression labels matching training
    const std::vector<std::string> labels = {
        "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
    };

    // Model expects 48x48 RGB images
    const int inputSize_ = 48;
};

#endif