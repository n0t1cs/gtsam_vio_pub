#include "frontend_thread.h"   // Eigen (incl. Geometry/AngleAxis) via vio_types.h -> Eigen/Dense

namespace vio {

// SO(3) exponential of a rotation vector (Eigen AngleAxis; small-angle safe).
static Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
    const double th = w.norm();
    if (th < 1e-9) {
        Eigen::Matrix3d W;                       // I + [w]_x (first order)
        W <<    0, -w.z(),  w.y(),
             w.z(),     0, -w.x(),
            -w.y(),  w.x(),     0;
        return Eigen::Matrix3d::Identity() + W;
    }
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

FrontendThread::FrontendThread(const CamCalib& calib, const Params& p,
                               ThreadSafeQueue<StampedImage>& in,
                               ThreadSafeQueue<StampedObservation>& out,
                               ThreadSafeQueue<ImuSample>* imu_in)
    : tracker_(calib, p.max_pts, p.min_dist, p.fb_err, p.ransac_th, p.quality),
      in_(in), out_(out), imu_in_(imu_in), rng_seed_(p.rng_seed) {}

Eigen::Matrix3d FrontendThread::integrate_gyro(int64_t t_cur) {
    if (!imu_in_) return Eigen::Matrix3d::Identity();

    // Pull all IMU that has arrived up to (a hair past) this image stamp.
    ImuSample s;
    while (imu_buf_.empty() || imu_buf_.back().t_ns < t_cur) {
        if (!imu_in_->try_pop(s)) break;         // real-time: could block until t>=t_cur
        imu_buf_.push_back(s);
    }

    // First frame: no interval to integrate; just drop consumed IMU and prime the clock.
    if (prev_t_ns_ < 0) {
        while (!imu_buf_.empty() && imu_buf_.front().t_ns <= t_cur) imu_buf_.pop_front();
        return Eigen::Matrix3d::Identity();
    }

    // Right-multiply body-frame increments over (prev_t_ns_, t_cur].
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    int64_t t_last = prev_t_ns_;
    while (!imu_buf_.empty() && imu_buf_.front().t_ns <= t_cur) {
        const ImuSample& im = imu_buf_.front();
        const double dt = (im.t_ns - t_last) * 1e-9;
        if (dt > 0.0 && dt < 0.1) R = R * so3_exp(im.w * dt);
        t_last = im.t_ns;
        imu_buf_.pop_front();
    }
    return R;
}

FrontendThread::~FrontendThread() {
    request_stop();
    join();
}

void FrontendThread::start() {
    if (running_.exchange(true)) return;             // already started
    th_ = std::thread(&FrontendThread::run, this);
}

void FrontendThread::run() {
    // Seed THIS thread's OpenCV RNG (findFundamentalMat RANSAC). cv::theRNG() is
    // thread-local, so determinism set on the main thread does not reach the worker.
    if (rng_seed_ >= 0) cv::setRNGSeed(rng_seed_);
    StampedImage si;
    while (!stop_.load(std::memory_order_relaxed) && in_.pop(si)) {
        const Eigen::Matrix3d dR_body = integrate_gyro(si.t_ns);   // gyro since last frame
        StampedObservation so;
        so.t_ns = si.t_ns;
        so.obs  = tracker_.step(si.img, dR_body);    // IMU-aided KLT
        out_.push(std::move(so));
        prev_t_ns_ = si.t_ns;
        n_.fetch_add(1, std::memory_order_relaxed);
    }
    out_.close();                                    // unblock the VIO thread's drain loop
    running_.store(false);
}

void FrontendThread::request_stop() {
    stop_.store(true);
    in_.close();
    out_.close();
}

void FrontendThread::join() {
    if (th_.joinable()) th_.join();
}

} // namespace vio
