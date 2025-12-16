// src/classification/expression_classifier.cpp

#include "classification/expression_classifier.hpp"
#include <iostream>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

ExpressionClassifier::ExpressionClassifier(const std::string& modelPath)
    : modelPath_(modelPath), isInitialized_(false) {}


bool ExpressionClassifier::initialize() {
    try {
        std::cout << "Initializing Expression Classifier..." << std::endl;

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ExpressionClassifier");
        Ort::SessionOptions sessionOptions;

#ifdef _WIN32
        std::wstring wideModelPath(modelPath_.begin(), modelPath_.end());
        session_ = std::make_unique<Ort::Session>(*env_, wideModelPath.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath_.c_str(), sessionOptions);
#endif

        isInitialized_ = true;
        std::cout << "Expression classifier initialized successfully" << std::endl;
        std::cout << "Classes: ";
        for (const auto& label : labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing classifier: " << e.what() << std::endl;
        return false;
    }
}

ExpressionResult ExpressionClassifier::classify(const cv::Mat& faceROI) {
    ExpressionResult result;
    result.classId = 1;
    result.confidence = 0.0f;
    result.label = "Unknown";

    if (!isInitialized_ || faceROI.empty()) {
        return result;
    }

    try {
        // Preprocess facial image
        cv::Mat preprocessed = preprocessFace(faceROI);

        // Convert to CHW format (Channels, width, height)
        std::vector<float> inputTensorValues(3 * inputSize_ * inputSize_);
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);

        for (int c = 0; c < 3; c++) {
            std::memcpy(inputTensorValues.data() + c * inputSize_ * inputSize_,
                        channels[c].data, inputSize_ * inputSize_ * sizeof(float));
        }

        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, inputSize_, inputSize_};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size()
        );

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session_->GetInputNameAllocated(0, allocator);
        auto outputName = session_->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        // Run inference
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        // Get output (7 classes)
        float* rawOutput = outputTensors[0].GetTensorMutableData<float>();

        // Apply softmax to get probabilities
        std::vector<float> probabilities(7);
        float maxLogit = *std::max_element(rawOutput, rawOutput + 7);
        float sumExp = 0.0f;

        for (int i = 0; i < 7; i++) {
            probabilities[i] = std::exp(rawOutput[i] - maxLogit);
            sumExp += probabilities[i];
        }

        for (int i = 0; i < 7; i++) {
            probabilities[i] /= sumExp;
        }

        // Find class with highest probability
        auto maxIt = std::max_element(probabilities.begin(), probabilities.end());
        int maxIndex = std::distance(probabilities.begin(), maxIt);

        result.classId = maxIndex;
        result.label = labels[maxIndex];
        result.confidence = probabilities[maxIndex];
    } catch (const std::exception& e) {
        std::cerr << "Classification error: " << e.what() << std::endl;
    }

    return result;
}

cv::Mat ExpressionClassifier::preprocessFace(const cv::Mat& face) {
    // Resize to 48x48
    cv::Mat resized;
    cv::resize(face, resized, cv::Size(inputSize_, inputSize_));

    // Convert BGR to RGB (OpenCV uses BGR by default!)
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    cv::Mat floatImage;
    rgb.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // FIXED: Apply simple normalization matching training (mean=0.5, std=0.5)
    // This converts [0, 1] range to [-1, 1] range
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);

    // RGB order now: channels[0]=R, channels[1]=G, channels[2]=B
    channels[0] = (channels[0] - 0.5) / 0.5;  // R: normalize to [-1, 1]
    channels[1] = (channels[1] - 0.5) / 0.5;  // G: normalize to [-1, 1]
    channels[2] = (channels[2] - 0.5) / 0.5;  // B: normalize to [-1, 1]

    cv::Mat normalized;
    cv::merge(channels, normalized);

    return normalized;
}