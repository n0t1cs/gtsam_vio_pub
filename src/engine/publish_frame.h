#pragma once
// The published VIO frame. ModalAI's VIO servers publish VIO_FRAME_BODY_ALIGNED (libmodal-pipe
// vio_data_t.h): origin at the IMU at initialisation, gravity aligned, FRD (z down), and
// voxl-vision-hub reads it that way (it rejects a frame more than 35 deg off level or upside down).
// The estimator's world is z UP with gravity along -z (vins_realign) and an arbitrary yaw, so
//   M = Rz(yaw_target - yaw0) * diag(1, -1, -1),   p_pub = M p_w + offset,   R_pub = M R_w
// with yaw0 the heading of the IMU x axis at the session's first output. A first session or a
// HARD reset starts at the origin with zero heading. A SOFT reset (an automatic one included)
// continues from the last published position and heading, so a consumer sees no jump.
//
// Covariance: GTSAM's tangent is [rotation, translation] in the BODY frame. vio_data_t wants
// [x, y, z, roll, pitch, yaw] in the VIO frame as the upper triangle of a 6x6 (21 floats, row
// major). Both blocks are rotated with R_pub; the rotation block is used for roll, pitch and yaw as
// a small-angle approximation. Only keyframes carry a covariance, frames in between repeat it.
#include "engine.h"

#include <Eigen/Dense>

namespace gvio {

struct PublishedState {
    bool valid = false;                        // state OK and converted
    Eigen::Vector3d p = Eigen::Vector3d::Zero(), v = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();   // IMU to VIO frame
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 1);   // unit, in the VIO frame (FRD: +z)
    bool cov_valid = false;
    float pose_cov21[21];
};

class PublishFrame {
public:
    explicit PublishFrame(bool soft_reset_keeps_frame = true) : keep_(soft_reset_keeps_frame) {}
    PublishedState convert(const EngineOutput& o);
    static void upper21(const Eigen::Matrix<double, 6, 6>& P, float out[21]);
    // GTSAM [rot, trans] body covariance -> [x, y, z, roll, pitch, yaw] in a frame rotated by R.
    static Eigen::Matrix<double, 6, 6> to_xyzrpy(const Eigen::Matrix<double, 6, 6>& C, const Eigen::Matrix3d& R);

private:
    bool keep_;
    int session_ = -1;
    bool aligned_ = false;
    Eigen::Matrix3d M_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d off_ = Eigen::Vector3d::Zero();
    bool have_last_ = false;
    Eigen::Vector3d last_p_ = Eigen::Vector3d::Zero();
    double last_yaw_ = 0.0;
    bool have_cov_ = false;
    float last_cov_[21];
};

} // namespace gvio
