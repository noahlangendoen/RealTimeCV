#include <opencv2/opencv.hpp>
#include "detection/face_detector.hpp"

int main() {
    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Can't open camera" << std::endl;
        return -1;
    }
    
    // Initialize face detector
    FaceDetector detector("../models/yolov8n-face.onnx");
    if (!detector.initialize()) {
        return -1;
    }
    
    cv::Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) break;
        
        // Detect faces
        std::vector<FaceBox> faces = detector.detectFaces(frame);
        
        // Draw boxes
        for (const auto& face : faces) {
            cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imshow("Webcam", frame);
        if (cv::waitKey(1) == 27) break; // ESC to exit
    }
    
    return 0;
}