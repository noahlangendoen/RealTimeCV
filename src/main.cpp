#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    
    cv::Mat frame;
    cv::VideoCapture cap;

    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    
    cap.open(deviceID, apiID);

    if (!cap.isOpened()) {
        std::cerr << "ERROR --- Unable to open the camera." << std::endl;
        return -1;
    }


    std::cout << "START\n";

    for (;;) {
        // wait for a new frame from camera and store it in frame.
        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "ERROR --- Blank frame grabbed\n";
            break;
        }

        cv::imshow("Live", frame);
        if (cv::waitKey(5) >= 0) {
            break;
        }
    }


    return 0;
}