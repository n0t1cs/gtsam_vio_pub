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
    Eigen::Vector3d b_a = Eigen::Vector3d::Zero();   // accel bias (only set if est_accel_bias)
    std::vector<Eigen::Vector3d> velocities;

    // Set by vins_realign:
    std::vector<Eigen::Matrix4d> T_wb;
};

// est_accel_bias: co-estimate the accelerometer bias in the gravity refinement so
// it is not absorbed into the gravity direction (fixes low-excitation init tilt,
// e.g. V1_01). Default false = the original behaviour (b_a fixed at zero).
AlignedInit init_align(const SfmResult& sfm, const EurocData& data,
                       bool est_accel_bias = false);

} // namespace vio