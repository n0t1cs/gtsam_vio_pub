#pragma once
// Thread hand-off primitives for the multi-threaded pipeline:
//   Thread 1 (frontend) --StampedObservation--> Thread 2 (VIO) --keyframes--> Thread 3 (PGO)
// One generic bounded blocking queue serves every edge. The IMU stream fans out to
// both the frontend (gyro-aided KLT prediction, task b) and the VIO (preintegration)
// through two independent queues.
//
// Design:
//  * bounded  -> natural backpressure; offline this just paces the producer, and it
//    caps memory. A future real-time mode can prefer drop-oldest over blocking.
//  * close()  -> graceful shutdown: once the image source is exhausted the producer
//    closes the queue; pop() then drains remaining items and returns false when empty,
//    so the consumer loop terminates cleanly (`while (q.pop(x)) { ... }`).

#include "vio_types.h"
#include <opencv2/core.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>

namespace vio {

// Timestamped payloads on the thread edges (t_ns = the shared pipeline clock).
struct StampedImage       { int64_t t_ns = 0; cv::Mat img; };
struct StampedObservation { int64_t t_ns = 0; Observation obs; };

template <typename T>
class ThreadSafeQueue {
public:
    // max_size = 0 -> unbounded (no backpressure).
    explicit ThreadSafeQueue(std::size_t max_size = 0) : max_size_(max_size) {}

    // Blocking push (waits if bounded and full). No-op after close().
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return closed_ || max_size_ == 0 || q_.size() < max_size_; });
        if (closed_) return;
        q_.push_back(std::move(v));
        lk.unlock();
        not_empty_.notify_one();
    }

    // Real-time variant: never blocks; if full, drops the OLDEST to keep latency
    // bounded. Returns true if something was dropped. (KLT needs consecutive frames,
    // so drop on the VIO edge, never on the image edge.)
    bool push_drop_oldest(T v) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) return false;
        bool dropped = false;
        if (max_size_ && q_.size() >= max_size_) { q_.pop_front(); dropped = true; }
        q_.push_back(std::move(v));
        lk.unlock();
        not_empty_.notify_one();
        return dropped;
    }

    // Blocking pop. Returns false only when the queue is closed AND drained.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;              // closed and drained
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    // Non-blocking pop.
    bool try_pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    // No more pushes; wakes every waiter so consumers can drain and exit.
    void close() {
        { std::lock_guard<std::mutex> lk(m_); closed_ = true; }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t size() const { std::lock_guard<std::mutex> lk(m_); return q_.size(); }
    bool closed() const      { std::lock_guard<std::mutex> lk(m_); return closed_; }

private:
    mutable std::mutex m_;
    std::condition_variable not_empty_, not_full_;
    std::deque<T> q_;
    std::size_t   max_size_;
    bool          closed_ = false;
};

} // namespace vio
