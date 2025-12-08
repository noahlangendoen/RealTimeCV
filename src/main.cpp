#include <opencv2/opencv.hpp>
#include "detection/face_detector.hpp"
#include "classification/expression_classifier.hpp"
#include "core/env_loader.hpp"

int main() {

    EnvLoader env;

    if (!env.load("../.env")) {
        std::cerr << "ERROR: .env file not found";
        return 0;
    }

    std::string detector_path = env.get("FACE_DETECTOR");
    std::string classifier_path = env.get("BEST_CLASSIFIER");

    std::cout << "Detector Path: " << detector_path << "\nClassifier Path: " << classifier_path << std::endl;

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Can't open camera" << std::endl;
        return -1;
    }
    
    // Initialize face detector
    FaceDetector detector(detector_path);
    if (!detector.initialize()) {
        return -1;
    }
    
    // Initialize expression classifier
    ExpressionClassifier classifier(classifier_path);
    if (!classifier.initialize()) {
        return -1;
    }
    
    cv::Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) break;
        
        // Detect faces
        std::vector<FaceBox> faces = detector.detectFaces(frame);
        
        // Process each detected face
        for (const auto& face : faces) {
            // Draw bounding box
            cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);
            
            // Extract face ROI
            cv::Mat faceROI = frame(face.box);
            
            // Classify expression
            ExpressionResult expression = classifier.classify(faceROI);
            
            // Display result
            std::string text = expression.label + " (" + 
                             std::to_string((int)(expression.confidence * 100)) + "%)";
            
            cv::putText(frame, text, 
                       cv::Point(face.box.x, face.box.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6,
                       cv::Scalar(0, 255, 0), 2);
            
            std::cout << "Detected: " << expression.label 
                     << " (confidence: " << expression.confidence << ")" << std::endl;
        }
        
        cv::imshow("Expression Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC to exit
    }
    
    return 0;
}