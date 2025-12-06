// src/detection/face_detector.cpp
#include "detection/face_detector.hpp"
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

FaceDetector::FaceDetector(const std::string& modelPath)
    : modelPath_(modelPath), isInitialized_(false) {}

FaceDetector::~FaceDetector() {
    // ONNX Runtime cleanup handled by unique_ptr
}

bool FaceDetector::initialize() {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceDetector");
        Ort::SessionOptions sessionOptions;
        
#ifdef _WIN32
        std::wstring wideModelPath(modelPath_.begin(), modelPath_.end());
        session_ = std::make_unique<Ort::Session>(*env_, wideModelPath.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath_.c_str(), sessionOptions);
#endif
        
        isInitialized_ = true;
        std::cout << "Face detector initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<FaceBox> FaceDetector::detectFaces(const cv::Mat& frame) {
    std::vector<FaceBox> faces;
    
    if (!isInitialized_ || frame.empty()) {
        return faces;
    }
    
    try {
        // YOLOv8 expects 640x640 input
        const int inputWidth = 640;
        const int inputHeight = 640;
        
        // Preprocess: resize and normalize
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(inputWidth, inputHeight));
        
        cv::Mat floatImage;
        resized.convertTo(floatImage, CV_32F, 1.0 / 255.0);
        
        // Convert to RGB (YOLO expects RGB)
        cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);
        
        // Convert to CHW format (channels, height, width)
        std::vector<float> inputTensorValues(3 * inputHeight * inputWidth);
        std::vector<cv::Mat> channels(3);
        cv::split(floatImage, channels);
        
        for (int c = 0; c < 3; c++) {
            std::memcpy(inputTensorValues.data() + c * inputHeight * inputWidth,
                       channels[c].data, inputHeight * inputWidth * sizeof(float));
        }
        
        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size()
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
        
        // Get output tensor
        float* rawOutput = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // YOLOv8 output shape: [1, 84, 8400] 
        // 84 = 4 (bbox) + 80 (classes)
        int numDetections = outputShape[2];  // 8400
        int numElements = outputShape[1];     // 84
        
        // Calculate scale factors
        float scaleX = (float)frame.cols / inputWidth;
        float scaleY = (float)frame.rows / inputHeight;
        
        // Parse detections (YOLOv8 format is transposed)
        for (int i = 0; i < numDetections; i++) {
            // Extract values for this detection
            float cx = rawOutput[i];                    // center x
            float cy = rawOutput[i + numDetections];    // center y
            float w = rawOutput[i + numDetections * 2]; // width
            float h = rawOutput[i + numDetections * 3]; // height
            
            // Get max class score (skip first 4 bbox values)
            float maxScore = 0.0f;
            for (int j = 4; j < numElements; j++) {
                float score = rawOutput[i + numDetections * j];
                if (score > maxScore) {
                    maxScore = score;
                }
            }
            
            // Filter by confidence threshold
            if (maxScore > 0.5f) {
                // Convert from center format to corner format
                float x1 = (cx - w / 2.0f) * scaleX;
                float y1 = (cy - h / 2.0f) * scaleY;
                float x2 = (cx + w / 2.0f) * scaleX;
                float y2 = (cy + h / 2.0f) * scaleY;
                
                // Clamp to frame boundaries
                x1 = std::max(0.0f, std::min(x1, (float)frame.cols));
                y1 = std::max(0.0f, std::min(y1, (float)frame.rows));
                x2 = std::max(0.0f, std::min(x2, (float)frame.cols));
                y2 = std::max(0.0f, std::min(y2, (float)frame.rows));
                
                FaceBox face;
                face.box = cv::Rect(
                    (int)x1, (int)y1,
                    (int)(x2 - x1), (int)(y2 - y1)
                );
                face.confidence = maxScore;
                
                faces.push_back(face);
            }
        }
        
        std::cout << "Detected " << faces.size() << " faces" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Detection error: " << e.what() << std::endl;
    }
    
    return faces;
}