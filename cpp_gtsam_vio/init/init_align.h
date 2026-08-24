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
    std::vector<Eigen::Vector3d> velocities;

    // Set by vins_realign:
    std::vector<Eigen::Matrix4d> T_wb;
};

AlignedInit init_align(const SfmResult& sfm, const EurocData& data);

} // namespace vio