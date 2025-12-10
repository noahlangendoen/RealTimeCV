// src/pipeline/pipeline_manager.cpp
#include "pipeline/pipeline_manager.hpp"
#include <iostream>
#include <iomanip>

PipelineManager::PipelineManager(
    const std::string& detectorPath,
    const std::string& classifierPath,
    int cameraId,
    int captureWidth,
    int captureHeight
)
    : detectorPath_(detectorPath)
    , classifierPath_(classifierPath)
    , cameraId_(cameraId)
    , captureWidth_(captureWidth)
    , captureHeight_(captureHeight)
    , isRunning_(false)
    , isInitialized_(false)
    , detectionThreshold_(0.5f)
    , classificationThreshold_(0.3f)
    , maxQueueSize_(10)
    , frameIdCounter_(0)
{
}

PipelineManager::~PipelineManager() {
    stop();
}

bool PipelineManager::initialize() {
    std::cout << "Initializing Pipeline Manager..." << std::endl;

    // Initialize Live Capture

    std::cout << "Initializing Live Caputre..." << std::endl;
    videoCapture_ = std::make_unique<VideoCapture>(
        cameraId_, cv::CAP_ANY, captureWidth_, captureHeight_
    );

    if (!videoCapture_->initialize()) {
        std::cerr << "Failed to Initialize Live Capture" << std::endl;
        return false;
    }

    // Initialize Detection Model
    std::cout << "Initializing Face Detector..." << std::endl;
    faceDetector_ = std::make_unique<FaceDetector>(detectorPath_);

    if (!faceDetector_->initialize()) {
        std::cerr << "Failed to Initialize Face Detector" << std::endl;
        return false;
    }

    // Initialize Classifier Model
    std::cout << "Initializing Classifier..." << std::endl;
    expressionClassifier_ = std::make_unique<ExpressionClassifier>(classifierPath_);

    if (!expressionClassifier_->initialize()) {
        std::cerr << "Failed to Initialize Classifier" << std::endl;
        return false;
    }

    // Initialize Buffers
    captureBuffer_ = std::make_unique<FrameBuffer>(maxQueueSize_);
    detectionBuffer_ = std::make_unique<FrameBuffer>(maxQueueSize_);

    isInitialized_ = true;
    std::cout << "Succesfully Initialized Pipeline" << std::endl;

    return true;
}

void PipelineManager::start() {
    if (!isInitialized_) {
        std::cerr << "Please initialize Pipeline Manager First" << std::endl;
        return;
    }

    if (isRunning_) {
        std::cerr << "Pipeline Already Running" << std::endl;
        return;
    }

    std::cout << "Starting Pipeline Threads..." << std::endl;

    isRunning_ = true;

    // Start all threads
    captureThread_ = std::thread(&PipelineManager::captureThreadFunc, this);
    detectionThread_ = std::thread(&PipelineManager::detectionThreadFunc, this);
    classificationThread_ = std::thread(&PipelineManager::classificationThreadFunc, this);
    displayThread_ = std::thread(&PipelineManager::displayThreadFunc, this);

    std::cout << "All threads started successfully\nPress ESC to stop" << std::endl;
}

void PipelineManager::stop() {
    if (!isRunning_) {
        return;
    }

    std::cout << "Stopping Pipeline..." << std::endl;

    isRunning_ = false;

    // Shutdown buffers
    if (captureBuffer_) captureBuffer_->shutdown();
    if (detectionBuffer_) detectionBuffer_->shutdown();

    // Wake up classification and display threads
    classificationQueueCV_.notify_all();
    displayQueueCV_.notify_all();

    // Join all threads
    if (captureThread_.joinable()) {
        std::cout << "Waiting for Capture Thread..." << std::endl;
        captureThread_.join();
    }

    if (detectionThread_.joinable()) {
        std::cout << "Waiting for Capture Thread..." << std::endl;
        detectionThread_.join();
    }

    if (classificationThread_.joinable()) {
        std::cout << "Waiting for Capture Thread..." << std::endl;
        classificationThread_.join();
    }

    if (displayThread_.joinable()) {
        std::cout << "Waiting for Capture Thread..." << std::endl;
        displayThread_.join();
    }

    std::cout << "All Threads Stopped" << std::endl;
}

bool PipelineManager::isRunning() {
    return isRunning_;
}

void PipelineManager::setDetectionThreshold(float threshold) {
    detectionThreshold_ = threshold;
}

void PipelineManager::setClassificationThreshold(float threshold) {
    classificationThreshold_ = threshold;
}

void PipelineManager::captureThreadFunc() {
    std::cout << "Capture Thread Started" << std::endl;

    // Open camera directly
    cv::VideoCapture cap(cameraId_);
    if (!cap.isOpened()) {
        std::cerr << "Failed to Open Camera in Capture Thread" << std::endl;
        isRunning_ = false;
        return;
    }

    // Set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, captureWidth_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, captureHeight_);

    std::cout << "Camera opened successfully in capture thread" << std::endl;

    cv::Mat frame;
    int frameCount = 0;

    while (isRunning_) {
        // Capture frame
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to Read Frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Create frame with metadata
        Frame capturedFrame;
        capturedFrame.data = frame.clone();
        capturedFrame.timestamp = std::chrono::steady_clock::now();
        capturedFrame.frameId = frameIdCounter_++;

        // Push to detection buffer (will block if full)
        detectionBuffer_->push(capturedFrame);

        frameCount++;

        // Debug output every 30 frames
        if (frameCount % 30 == 0) {
            std::cout << "Captured " << frameCount << " frames" << std::endl;
        }
    }

    cap.release();
    std::cout << "Stopped Camera -- Total Frames: " << frameCount << std::endl;
}

void PipelineManager::detectionThreadFunc() {
    std::cout << "Detection Thread Started" << std::endl;

    int frameCount = 0;

    while (isRunning_) {
        Frame frame = detectionBuffer_->pop();

        if (!frame.isValid()) {
            // If buffer was shutdown
            continue;
        }

        // Detect faces
        std::vector<FaceBox> faces = faceDetector_->detectFaces(frame.data);

        // Filter faces by confidence threshold
        std::vector<FaceBox> filteredFaces;
        for (const auto& face : faces) {
            if (face.confidence >= detectionThreshold_) {
                filteredFaces.push_back(face);
            }
        }

        if (!filteredFaces.empty()) {
            // Store how many faces we expect to classify for this frame
            {
                std::lock_guard<std::mutex> lock(frameMapMutex_);
                frameExpectedFaces_[frame.frameId] = filteredFaces.size();
                frameClassifications_[frame.frameId] = std::vector<ClassifiedFace>();
            }

            // Push each detected face to classification queue
            std::unique_lock<std::mutex> lock(classificationQueueMutex_);
            for (const auto& faceBox : filteredFaces) {
                // Extract face ROI
                cv::Rect roi = faceBox.box;
                roi.x = std::max(0, roi.x);
                roi.y = std::max(0, roi.y);
                roi.width = std::min(roi.width, frame.data.cols - roi.x);
                roi.height = std::min(roi.height, frame.data.rows - roi.y);

                if (roi.width > 0 && roi.height > 0) {
                    DetectedFace detectedFace;
                    detectedFace.faceROI = frame.data(roi).clone();
                    detectedFace.bbox = faceBox;
                    detectedFace.frameId = frame.frameId;
                    detectedFace.timestamp = frame.timestamp;

                    classificationQueue_.push(detectedFace);
                }
            }

            lock.unlock();
            classificationQueueCV_.notify_one();

            // Also send frame to display queue (will be drawn with classifications later)
            ProcessedFrame processedFrame;
            processedFrame.frame = frame.data.clone();
            processedFrame.frameId = frame.frameId;
            processedFrame.timestamp = frame.timestamp;

            std::unique_lock<std::mutex> dispLock(displayQueueMutex_);
            displayQueue_.push(processedFrame);
            dispLock.unlock();
            displayQueueCV_.notify_one();
        } else {
            // No faces detected, send frame to display
            ProcessedFrame processedFrame;
            processedFrame.frame = frame.data.clone();
            processedFrame.timestamp = frame.timestamp;
            processedFrame.frameId = frame.frameId;

            std::unique_lock<std::mutex> lock(displayQueueMutex_);
            displayQueue_.push(processedFrame);
            lock.unlock();
            displayQueueCV_.notify_one();
        }

        frameCount++;
    }

    std::cout << "Detection Thread Stopped\nFrame Count: " << frameCount << std::endl;
}

void PipelineManager::classificationThreadFunc() {
    std::cout << "Classification Thread Started" << std::endl;

    int faceCount = 0;

    while (isRunning_) {
        // Wait for a detected face
        std::unique_lock<std::mutex> lock(classificationQueueMutex_);
        classificationQueueCV_.wait(lock, [this] {
            return !classificationQueue_.empty() || !isRunning_;
        });

        if (!isRunning_ && classificationQueue_.empty()) {
            break;
        }

        if (classificationQueue_.empty()) {
            continue;
        }

        DetectedFace detectedFace = classificationQueue_.front();
        classificationQueue_.pop();
        lock.unlock();

        // Classify expression
        ExpressionResult expression = expressionClassifier_->classify(detectedFace.faceROI);

        // Created classified face
        ClassifiedFace classifiedFace;
        classifiedFace.bbox = detectedFace.bbox;
        classifiedFace.timestamp = detectedFace.timestamp;
        classifiedFace.expression = expression;
        classifiedFace.frameId = detectedFace.frameId;
    
        // Add frame to classification list
        {
            std::lock_guard<std::mutex> lock(frameMapMutex_);

            if (frameClassifications_.find(detectedFace.frameId) != frameClassifications_.end()) {
                frameClassifications_[detectedFace.frameId].push_back(classifiedFace);
            }
        }

        faceCount++;
    }

    std::cout << "Classification Thread Stopped\nTotal Faces: " << faceCount << std::endl;
}

void PipelineManager::displayThreadFunc() {
    std::cout << "Display Thread Started" << std::endl;

    cv::namedWindow("Expression Detection", cv::WINDOW_AUTOSIZE);

    int frameCount = 0;

    while (isRunning_) {
        // Wait for a frame from the display queue
        ProcessedFrame processedFrame;
        {
            std::unique_lock<std::mutex> lock(displayQueueMutex_);
            displayQueueCV_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !displayQueue_.empty() || !isRunning_;
            });

            if (!isRunning_ && displayQueue_.empty()) {
                break;
            }

            if (displayQueue_.empty()) {
                continue;
            }

            processedFrame = displayQueue_.front();
            displayQueue_.pop();
        }

        if (!processedFrame.isValid()) {
            continue;
        }

        // Get classifications for this frame
        std::vector<ClassifiedFace> facesToDraw;
        {
            std::lock_guard<std::mutex> lock(frameMapMutex_);

            // Find classifications for this specific frame
            auto it = frameClassifications_.find(processedFrame.frameId);
            if (it != frameClassifications_.end()) {
                facesToDraw = it->second;
            }

            // Clean up old entries
            auto cleanIt = frameClassifications_.begin();
            while (cleanIt != frameClassifications_.end()) {
                if (cleanIt->first < processedFrame.frameId - 10) {
                    frameExpectedFaces_.erase(cleanIt->first);
                    cleanIt = frameClassifications_.erase(cleanIt);
                } else {
                    ++cleanIt;
                }
            }
        }

        // Draw classifications on the frame
        drawResults(processedFrame.frame, facesToDraw);

        // Show frame
        cv::imshow("Expression Detection", processedFrame.frame);

        // Check for ESC
        int key = cv::waitKey(1);
        if (key == 27) {
            isRunning_ = false;
            break;
        }

        frameCount++;
    }

    cv::destroyAllWindows();

    std::cout << "Display Thread Stopped\nTotal Frames: " << frameCount << std::endl;
}

void PipelineManager::drawResults(cv::Mat& frame, const std::vector<ClassifiedFace>& faces) {
    for (const auto& face : faces) {
        // Draw BBox
        cv::rectangle(frame, face.bbox.box, cv::Scalar(0, 255, 0), 2);

        // Filter by classification threshold
        if (face.expression.confidence < classificationThreshold_) {
            continue;
        }

        // Draw expression label with confidence
        std::stringstream ss;

        ss << face.expression.label << " "
           << std::fixed << std::setprecision(0)
           << (face.expression.confidence * 100) << "%";

        std::string text = ss.str();

        // Background for text
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_COMPLEX, 0.6, 2, &baseline);

        cv::Point textOrg(face.bbox.box.x, face.bbox.box.y - 10);
        cv::rectangle(frame, 
                      textOrg + cv::Point(0, baseline),
                      textOrg + cv::Point(textSize.width, -textSize.height),
                      cv::Scalar(0, 255, 0), -1);
        
        // Draw Text
        cv::putText(frame, text, textOrg,
                    cv::FONT_HERSHEY_COMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 2);

    }
}