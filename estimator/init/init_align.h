#pragma once
#include "init_sfm.h"
#include "vio_types.h"

namespace vio {

struct AlignedInit {
    SfmResult sfm;
    bool success = false;
    double scale = 1.0;
    Eigen::Vector3d g_w = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_g = Eigen::Vector3d::Zero();
    Eigen::Vector3d b_a = Eigen::Vector3d::Zero();   // accel bias seed: zero (not observable here)
    std::vector<Eigen::Vector3d> velocities;

    // Set by vins_realign:
    std::vector<Eigen::Matrix4d> T_wb;
};

// Gyro bias, then the linear [v, g, s] solve, then gravity refined on the sphere tangent.
// The accelerometer bias stays at zero (co-estimating it was measured and removed: the
// bias/gravity split is unobservable in a low-excitation window).
// gravity_from_accel (init.gravity_from_accel): after the refined solve, the gravity direction
// is replaced by the one the accelerometer measures over the window (the summed, rotated specific
// force, the window's net velocity change neglected), and the velocities are re-solved with that gravity
// and the linear solve's scale fixed. The tilt then carries the true accelerometer bias (seed-independent)
// instead of the SfM solve's error.
AlignedInit init_align(const SfmResult& sfm, const EurocData& data, bool gravity_from_accel = false);

} // namespace vio