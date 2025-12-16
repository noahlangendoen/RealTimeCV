// src/core/frame_buffer.cpp
#include "core/frame_buffer.hpp"
#include <iostream>

FrameBuffer::FrameBuffer(size_t maxSize)
    : maxSize_(maxSize)
    , isShutdown_(false)
{
    if (maxSize_ == 0) {
        maxSize_ = 1;
    }
}

void FrameBuffer::push(const Frame& frame) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until buffer is not full or shutdown
    condNotFull_.wait(lock, [this] {
        return buffer_.size() < maxSize_ || isShutdown_;
    });

    if (isShutdown_) {
        return;
    }

    buffer_.push(frame);

    // Notify one waiting consumer
    condNotEmpty_.notify_one();
}

bool FrameBuffer::tryPush(const Frame& frame) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (isShutdown_ || buffer_.size() >= maxSize_) {
        return false;
    }

    buffer_.push(frame);
    condNotEmpty_.notify_one();
    return true;
}

Frame FrameBuffer::pop() {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until buffer is not empty or shutdown
    condNotEmpty_.wait(lock, [this] {
        return !buffer_.empty() || isShutdown_;
    });

    if (isShutdown_ && buffer_.empty()) {
        // Return empty frame on shutdown
        return Frame();
    }

    // Use move semantics to avoid copying cv::Mat data
    Frame frame = std::move(buffer_.front());
    buffer_.pop();

    // Notify one waiting producer
    condNotFull_.notify_one();

    return frame;
}

std::optional<Frame> FrameBuffer::tryPop() {
    std::unique_lock<std::mutex> lock(mutex_);

    if (buffer_.empty()) {
        return std::nullopt;
    }

    // Use move semantics to avoid copying
    Frame frame = std::move(buffer_.front());
    buffer_.pop();

    condNotFull_.notify_one();

    return frame;
}

std::optional<Frame> FrameBuffer::popWithTimeout(int milliseconds) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait with timeout
    bool success = condNotEmpty_.wait_for(
        lock,
        std::chrono::milliseconds(milliseconds),
        [this] { return !buffer_.empty() || isShutdown_; }
    );

    if (!success || buffer_.empty()) {
        return std::nullopt;
    }

    // Use move semantics to avoid copying
    Frame frame = std::move(buffer_.front());
    buffer_.pop();

    condNotFull_.notify_one();

    return frame;
}

bool FrameBuffer::isEmpty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

bool FrameBuffer::isFull() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size() >= maxSize_;
}

size_t FrameBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

size_t FrameBuffer::capacity() const {
    return maxSize_;
}

void FrameBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Clear the queue
    std::queue<Frame> empty;
    std::swap(buffer_, empty);

    // Notify all waiting producers that space is available
    condNotFull_.notify_all();
}

void FrameBuffer::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        isShutdown_ = true;
    }

    // Wake up all waiting threads
    condNotEmpty_.notify_all();
    condNotFull_.notify_all();

    std::cout << "Frame Buffer Shutdown" << std::endl;
}