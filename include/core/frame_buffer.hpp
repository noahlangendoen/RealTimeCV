// include/core/frame_buffer.hpp
#ifndef FRAME_BUFFER_HPP
#define FRAME_BUFFER_HPP

#include <queue>
#include <mutex>
#include <optional>
#include <condition_variable>
#include "types.hpp"


class FrameBuffer {
public:
    explicit FrameBuffer(size_t maxSize = 10);
    ~FrameBuffer() = default;

    // Push a frame to the buffer (blocks if full)
    void push(const Frame& frame);

    // Try to push without blocking (returns false if null)
    bool tryPush(const Frame& frame);

    // Pop a frame from the buffer (blocks if empty)
    Frame pop();

    // Try to pop without blocking (returns empty optional if empty)
    std::optional<Frame> tryPop();

    // Pop with timeout (returns empty optional if timeout)
    std::optional<Frame> popWithTimeout(int milliseconds);

    // Check if buffer is empty
    bool isEmpty() const;

    // Check if buffer is full
    bool isFull() const;

    // Get current size
    size_t size() const;

    // Get max capacity
    size_t capacity() const;

    // Clear all frames
    void clear();

    // Shutdown buffer (wake up all waiting threads)
    void shutdown();

private:
    std::queue<Frame> buffer_;
    mutable std::mutex mutex_;
    std::condition_variable condNotEmpty_;
    std::condition_variable condNotFull_;

    size_t maxSize_;
    bool isShutdown_;
};

#endif