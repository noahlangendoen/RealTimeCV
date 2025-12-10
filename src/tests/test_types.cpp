#include "core/types.hpp"
#include <iostream>
#include <cassert>

void testFrame() {
    std::cout << "TEST FRAME STRUCT" << std::endl;

    // Test default construction
    Frame f1;
    assert(!f1.isValid());

    // Test with data
    cv::Mat img(480, 640, CV_8UC3);
    Frame f2(img, 1);
    assert(f2.isValid());
    assert(f2.frameId == 1);
    
    std::cout << "Frame tests passed" << std::endl;
}

void testDetectedFace() {
    std::cout << "TEST DETECTED FACE STRUCT" << std::endl;

    DetectedFace face;
    assert(!face.isValid());

    face.frameId = 10;
    face.faceROI = cv::Mat(48, 48, CV_8UC3);
    assert(face.isValid());

    std::cout << "Detected face tests passed" << std::endl;
}

void testClassifiedFace() {
    std::cout << "TEST CLASSIFIEDFACE STRUCT" << std::endl;
    ClassifiedFace face;
    assert(!face.isValid());

    face.frameId = 5;
    face.expression.label = "happy";
    face.expression.confidence = 0.95f;
    assert(face.isValid());

    std::cout << "Classified face tests passed";
}

int main() {

    std::cout << "Testing types..." << std::endl;

    testFrame();
    testDetectedFace();
    testClassifiedFace();

    std::cout << "All type tests passed!" << std::endl;
    return 0;
}