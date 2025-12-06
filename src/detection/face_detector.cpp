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

// bool FaceDetector::initialize() {
//     try {
//         std::cout << "Initializing Face Detector..." << std::endl;
//         std::cout << "  Model: " << modelPath_ << std::endl;

//         // Create ONNX Runtime Environment
//         env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceDetector");

//         // Create session options
//         sessionOptions_ = std::make_unique<Ort::SessionOptions>();
//         sessionOptions_->SetIntraOpNumThreads(1);

//         // Enabling CUDA
//         sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

//         // Create Session
// #ifdef _WIN32
//         std::wstring wideModelPath(modelPath_.begin(), modelPath_.end());
//         session_ = std::make_unique<Ort::Session>(*env_, wideModelPath.c_str(), *sessionOptions_);
// #else
//         sesion_ = std::make_unique<Ort::Session>(*env_, modelPath_.c_str(), *sessionOptions_);
// #endif

//         // Get input info
//         size_t numInputNodes = session_->GetInputCount();
//         if (numInputNodes == 0) {
//             std::cerr << "ERROR: Model has no input nodes" << std::endl;
//             return false;
//         }

//         // Get input name
//         Ort::AllocatedStringPtr inputNameAllocated = session_->GetInputNameAllocated(0, allocator_);
//         inputNames_.push_back(inputNameAllocated.get());

//         // Get input shape
//         Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
//         auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
//         inputShape_ = tensorInfo.GetShape();

//         // YOLOv8 input shape: [1, 3, 640, 640]
//         if (inputShape_.size() == 4) {
//             inputHeight_ = static_cast<int>(inputShape_[2]);
//             inputWidth_ = static_cast<int>(inputShape_[3]);
//         }

//         // Get output info
//         size_t numOutputNodes = session_->GetOutputCount();
//         for (size_t i; i < numOutputNodes; i++) {
//             Ort::AllocatedStringPtr outputNameAllocated = session_->GetOutputNameAllocated(i, allocator_);
//             outputNames_.push_back(outputNameAllocated.get());
//         }

//         std::cout << "  Input shape: [" << inputShape_[0] << ", " << inputShape_[1] 
//                   << ", " << inputShape_[2] << ", " << inputShape_[3] << "]" << std::endl;
//         std::cout << "Confidence Threshold: " << confThreshold_ << std::endl;
//         std::cout << "IoU threshold: " << iouThreshold_ << std::endl;
//         std::cout << "Face Detector Initialized successfully" << std::endl;

//         isInitialized_ = true;
//         return true;
//     } catch (const Ort::Exception& e) {
//         std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
//         return false;
//     } catch (const std::exception& e) {
//         std::cerr << "Error Initializing Face Detector: " << e.what() << std::endl;
//         return false;
//     }
// }

// std::vector<DetectionResult> FaceDetector::detect(const Frame& frame) {
//     std::vector<DetectionResult> results;

//     if (!isInitialized_ || !frame.isValid()) {
//         return results;
//     }

//     std::vector<BBox> boxes = detectFaces(frame.data);

//     // Convert BBox to Detection Result with ROI extraction
//     for (const auto& box : boxes) {
//         DetectionResult result;
//         result.bbox = box;
//         result.frameId = frame.frameId;
//         result.timestamp = frame.timestamp;

//         // Extract face ROI
//         cv::Rect roi = box.toRect();

//         // Clamp ROI to image bounds
//         roi.x = std::max(0, roi.x);
//         roi.y = std::max(0, roi.y);
//         roi.width = std::min(roi.width, frame.data.cols - roi.x);
//         roi.height = std::min(roi.height, frame.data.rows - roi.y);

//         if (roi.width > 0 && roi.height > 0) {
//             result.faceROI = frame.data(roi).clone();
//             results.push_back(result);
//         }
//     }

//     return results;
// }

// std::vector<BBox> FaceDetector::detectFaces(const cv::Mat& image) {
//     if (!isInitialized_) {
//         std::cerr << "ERROR: Face Detector not initialized" << std::endl;
//         return {};
//     }

//     try {
//         // Preprocess img
//         cv::Mat preprocessed = preprocessImage(image);

//         // Create input tensor
//         std::vector<int64_t> inputShapeTensor = {1, 3, inputHeight_, inputWidth_};
//         size_t inputTensorSize = 1 * 3 * inputHeight_ * inputWidth_;

//         std::vector<float> inputTensorValues(inputTensorSize);

//         // Convert CHW format: 
//     }
// }