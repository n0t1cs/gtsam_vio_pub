#pragma once
#include "vio_types.h"
#include <Eigen/Dense>
#include <map>
#include <vector>

namespace vio {

struct SfmResult {
    bool success;
    int ref;                                     // index in `frames` used as world origin
    std::vector<Eigen::Matrix4d> T_wc;            // camera-in-world
    std::vector<Eigen::Matrix4d> Tcw;             // world-to-cam
    std::map<int64_t, Eigen::Vector3d> points;    // id -> world point
    std::vector<KeyframeEntry> frames;            // pass-through
    CamCalib calib;                               // pass-through
    double scale;                                 // always 1.0 here

    // Bundle-adjust conditioning (filled by bundle_adjust). graph.error is
    // 0.5 * sum(pixel residual^2) with sigma = 1px, so the per-observation RMS
    // reprojection is sqrt(2*err/n_obs). The PRE-BA value measures how self-
    // consistent the raw multi-view SfM is and is the cue used to reject
    // rotation-dominant / small-baseline (pickup-phase) windows.
    double ba_err_before = 0.0;
    double ba_err_after  = 0.0;
    int    ba_n_obs      = 0;

    SfmResult() : success(false), ref(-1), scale(1.0) {}
};

SfmResult init_sfm(const std::vector<KeyframeEntry>& frames, const CamCalib& calib);

// SE(3) utility (also used by other modules).
Eigen::Matrix4d inv_se3(const Eigen::Matrix4d& T);

} // namespace vio