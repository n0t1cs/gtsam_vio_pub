#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstdint>
#include <time.h>

namespace vio {

struct CamCalib {
    Eigen::Matrix3d K;
    Eigen::Vector4d dist;           // [k1, k2, p1, p2]
    Eigen::Matrix4d T_BC;           // camera-in-body
    int width;
    int height;
    int rate_hz;
};

struct ImuCalib {
    double gyro_noise_density;      // rad / s / sqrt(Hz)
    double gyro_random_walk;        // rad / s^2 / sqrt(Hz)
    double accel_noise_density;     // m / s^2 / sqrt(Hz)
    double accel_random_walk;       // m / s^3 / sqrt(Hz)
    Eigen::Matrix4d T_BI;           // IMU-in-body
    int rate_hz;
};

struct Frame {
    int64_t t_ns;
    std::string path;
};

struct ImuSample {
    int64_t t_ns;
    Eigen::Vector3d w;              // gyro (rad/s)
    Eigen::Vector3d a;              // accel (m/s^2)
};

struct GtSample {
    int64_t t_ns;
    Eigen::Vector3d p;
    Eigen::Vector4d q_wxyz;
    Eigen::Vector3d v;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

struct EurocData {
    CamCalib cam_calib;
    ImuCalib imu_calib;
    std::vector<Frame> frames;
    std::vector<ImuSample> imu;
    std::vector<GtSample> gt;
};

// One frame's tracked observations from the frontend.
struct Observation {
    std::vector<int64_t> ids;
    // Each row is a point (x, y). RowMajor so .row(i) is a 2-vector.
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts_raw;
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts_norm;
    std::vector<int> track_cnt;
    // Per-observation forward-backward KLT residual [px] this frame (0 for a freshly
    // detected track). Localization-quality signal for heteroscedastic reprojection
    // covariance -- high on hard/fast-rotation tracking, low on clean corners.
    std::vector<double> fb_res;
};

// Cost study: the tracker's stage times for one frame [ms, wall], filled by FeatureTracker::step,
// and the front-end thread's CPU clock when it finished the frame [ms]. Logged with the frame,
// never read by the estimator.
struct FrontendCost {
    double clahe = 0, predict = 0, klt_fwd = 0, klt_bwd = 0, reject = 0, detect = 0, copy = 0, obs = 0;
    double thread_cpu_ms = 0;
};

// CPU time consumed by the calling thread so far [ms].
inline double thread_cpu_ms() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// One keyframe in the SfM init buffer.
struct KeyframeEntry {
    int idx;                        // global camera-frame index
    double t;                       // seconds since dataset t0
    std::vector<int64_t> ids;
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts_raw;
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts_norm;
};

} // namespace vio