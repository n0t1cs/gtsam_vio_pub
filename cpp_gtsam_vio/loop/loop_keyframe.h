#pragma once
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "BRIEF.h"   // DVision::BRIEF::bitset

namespace vio {

// One harvested keyframe for offline loop closure (mirrors VINS loop_fusion's
// KeyFrame, trimmed to what the pose-graph stage needs). Two descriptor sets:
//   - point_desc   : BRIEF at tracked features that HAVE a 3D estimate; used for
//                    PnP loop verification (aligned 1:1 with point_3d/point_2d_*).
//   - brief_desc   : BRIEF at FAST corners over the whole image; used for DBoW
//                    place recognition and descriptor matching.
struct LoopKeyframe {
    int     index = -1;          // sequential KF index = pose-graph node id
    int     frame_idx = -1;      // global camera-frame index (for the image path)
    int     traj_idx = -1;       // row in the estimated-trajectory arrays (for LC writeback)
    double  t_s = 0.0;           // timestamp [s]
    Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();  // VIO body pose (odometry)

    // Tracked features with a 3D world estimate (PnP).
    std::vector<cv::Point3f> point_3d;       // world-frame 3D (from VIO)
    std::vector<cv::Point2f> point_2d_uv;    // pixel
    std::vector<cv::Point2f> point_2d_norm;  // normalized image plane
    std::vector<int64_t>     point_id;       // frontend feature id
    std::vector<DVision::BRIEF::bitset> point_desc;

    // FAST corners over the image (DBoW + matching).
    std::vector<cv::KeyPoint> keypoints;       // pixel
    std::vector<cv::KeyPoint> keypoints_norm;  // normalized image plane
    std::vector<DVision::BRIEF::bitset> brief_desc;
};

// One verified loop closure: the measured relative body pose of the CURRENT
// keyframe expressed in the OLD keyframe's frame (T_old_cur = T_old^-1 * T_cur),
// recovered by PnP. Feeds a BetweenFactor(X(old), X(cur)) in the pose graph.
struct LoopEdge {
    int index_cur = -1;
    int index_old = -1;
    Eigen::Matrix4d T_old_cur = Eigen::Matrix4d::Identity();
    int inliers = 0;
};

// Loop-closure tuning, threaded from VioParams (keeps loop/ decoupled from
// params.h). Defaults reproduce the previously-hardcoded behaviour.
struct LoopParams {
    // Detection gates (loop_database).
    int    skip_recent     = 50;
    int    min_inliers     = 30;
    int    hamming_th      = 80;
    double score_best      = 0.05;
    double score_cand      = 0.015;
    double fund_thresh_px  = 3.0;
    double max_trans_m     = 3.0;
    double max_yaw_deg     = 30.0;
    // Pose-graph use (pose_graph) — Front B. Defaults = today's behaviour.
    int    backbone_hops    = 1;      // odometry edges per node to its N predecessors (VINS-Mono uses 4); 1 = single-hop (committed)
    double loop_trans_sigma = 0.1;
    double loop_rot_sigma   = 0.05;
    double cauchy_c         = 1.0;
    bool   inlier_weighting = false;  // scale loop-edge sigma by sqrt(min_inliers/inliers)
    double min_correction_m = 0.0;    // drift-consistency gate [m]; 0 = off
    double loop_drift_rate  = 0.0;    // signal-vs-noise gate: skip if correction > rate*path_length; 0 = off
};

} // namespace vio
