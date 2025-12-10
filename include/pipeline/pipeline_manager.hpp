#ifndef PIPELINE_MANAGER_HPP
#define PIPELINE_MANAGER_HPP

#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <mutex>
#include <condition_variable>

#include "core/frame_buffer.hpp"
#include "core/types.hpp"
#include "capture/video_capture.hpp"
#include "detection/face_detector.hpp"
#include "classification/expression_classifier.hpp"

class PipelineManager {
public:
    PipelineManager(
        const std::string& detectorPath,
        const std::string& classifierPath,
        int cameraId = 0,
        int captureWidth = 640,
        int captureHeight = 480
    );

    ~PipelineManager();

    bool initialize();

    void start();

    void stop();

    bool isRunning();

    void setDetectionThreshold(float threshold);

    void setClassificationThreshold(float threshold);

    void setMaxQueueSize(size_t size);

private:
    // Webcam Connection & Read Thread
    void captureThreadFunc();

    // Face Detector Thread
    void detectionThreadFunc();

    // Classifier Thread
    void classificationThreadFunc();

    // Display Thread
    void displayThreadFunc();

    // Draw Results of BBox & Labels
    void drawResults(cv::Mat& frame, const std::vector<ClassifiedFace>& faces);


    /*
    COMPONENTS
    */

    std::unique_ptr<VideoCapture> videoCapture_;
    std::unique_ptr<FaceDetector> faceDetector_;
    std::unique_ptr<ExpressionClassifier> expressionClassifier_;

    // Frame Buffers for Queues
    std::unique_ptr<FrameBuffer> captureBuffer_;
    std::unique_ptr<FrameBuffer> detectionBuffer_;

    // Classification Queues
    std::queue<DetectedFace> classificationQueue_;
    std::mutex classificationQueueMutex_;
    std::condition_variable classificationQueueCV_;

    // Display Queue for Processed Frames
    std::queue<ProcessedFrame> displayQueue_;
    std::mutex displayQueueMutex_;
    std::condition_variable displayQueueCV_;

    // Map to Store Classifications for Frames Processing
    std::map<int, std::vector<ClassifiedFace>> frameClassifications_;
    std::map<int, int> frameExpectedFaces_;
    std::mutex frameMapMutex_;

    /*
    THREADS
    */

   std::thread captureThread_;
   std::thread detectionThread_;
   std::thread classificationThread_;
   std::thread displayThread_;

    /*
    CONTROL FLAGS
    */

    std::atomic<bool> isRunning_;
    std::atomic<bool> isInitialized_;

    /*
    CONFIG
    */

    std::string detectorPath_;
    std::string classifierPath_;
    int cameraId_;
    int captureWidth_;
    int captureHeight_;

    std::atomic<float> detectionThreshold_;
    std::atomic<float> classificationThreshold_;
    size_t maxQueueSize_;

    std::atomic<int> frameIdCounter_;
};



#endif
