#pragma once
// FrontendThread: runs a FeatureTracker on its OWN thread. Consumes StampedImage from
// an input queue, produces StampedObservation to an output queue. This is the "1 thread
// to frontend" tier of the pipeline; the VIO runs on another thread draining `out`.
//
// Usage (offline EuRoC; producer = a disk reader, real-time = the camera callback):
//   ThreadSafeQueue<StampedImage>       img_q(8);      // camera-rate images in
//   ThreadSafeQueue<StampedObservation> obs_q(8);      // tracked features out
//   FrontendThread fe(calib, {.max_pts=150}, img_q, obs_q);
//   fe.start();
//   // producer thread:  for each frame  img_q.push({t_ns, img});  then img_q.close();
//   // VIO thread:  StampedObservation so;  while (obs_q.pop(so)) { backend.add_frame(so.obs...) }
//   fe.join();
//
// Task (b) will extend the worker to read the IMU stream and pass a gyro-predicted
// rotation into the KLT step for large-motion robustness.

#include "frontend.h"
#include "frame_queue.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <thread>

namespace vio {

class FrontendThread {
public:
    struct Params {
        int    max_pts   = 150;
        int    min_dist  = 30;
        double fb_err    = 1.0;    // forward-backward KLT consistency [px]
        double ransac_th = 1.0;    // F-matrix RANSAC threshold [px]
        double quality   = 0.01;   // Shi-Tomasi quality level
        int    rng_seed  = -1;     // >=0: seed THIS thread's cv RNG (findFundamentalMat) for
                                   // reproducibility -- cv::theRNG() is thread-local, so the
                                   // main thread's setRNGSeed does NOT reach the worker. -1 = leave.
    };

    // imu_in (optional): the gyro stream for IMU-aided KLT prediction (task b). nullptr
    // -> plain prev-frame seeding. IMU must be fed ahead of / alongside the images so the
    // samples up to each image stamp are available when it is tracked.
    FrontendThread(const CamCalib& calib, const Params& p,
                   ThreadSafeQueue<StampedImage>& in,
                   ThreadSafeQueue<StampedObservation>& out,
                   ThreadSafeQueue<ImuSample>* imu_in = nullptr);
    ~FrontendThread();

    void start();          // launch the worker (idempotent)
    void join();           // block until worker finishes (input closed & drained)
    void request_stop();   // early stop; also closes `out`

    bool     running()          const { return running_.load(); }
    uint64_t frames_processed() const { return n_.load(); }

private:
    void run();
    // Body-frame rotation gyro-integrated over (prev_t_ns_, t_cur]. Consumes imu_in_
    // into imu_buf_ and advances prev_t_ns_. Identity if no IMU / first frame.
    Eigen::Matrix3d integrate_gyro(int64_t t_cur);

    FeatureTracker tracker_;
    ThreadSafeQueue<StampedImage>&       in_;
    ThreadSafeQueue<StampedObservation>& out_;
    ThreadSafeQueue<ImuSample>*          imu_in_ = nullptr;
    std::deque<ImuSample>                imu_buf_;
    int64_t                              prev_t_ns_ = -1;
    int                                  rng_seed_  = -1;
    std::thread           th_;
    std::atomic<bool>     stop_{false};
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> n_{0};
};

} // namespace vio
