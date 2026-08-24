#pragma once
#include <vector>
#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include "loop_keyframe.h"

namespace vio {

// Persistent 4-DoF (yaw + xyz) pose graph, grown ONLINE one keyframe at a time
// and re-optimized whenever a loop closure is added (VINS-Mono pose_graph analog).
// It runs alongside the fixed-lag VIO — never inside it, since the smoother has
// already marginalized the old keyframes a loop connects to.
//
// The graph + initial values are kept in the VIO (odometry) frame and never
// overwritten; optimize() re-solves from those initials and caches the corrected
// estimate. Roll & pitch are pinned to their VIO values (gravity-observable), so
// loops only move yaw and position -> effective 4-DoF.
class PoseGraph {
public:
    explicit PoseGraph(const LoopParams& lp = LoopParams());

    // Add a keyframe node (id == keyframe index) at its VIO body pose, with an
    // odometry edge from the previous node and a roll/pitch pin.
    void add_keyframe(int node_id, const Eigen::Matrix4d& T_wb);

    // Add a verified loop edge (robust). Does NOT optimize; call optimize() after.
    void add_loop(const LoopEdge& e);

    // Batch re-optimize from the VIO-frame initials; caches the corrected estimate.
    void optimize();

    int  num_loops() const { return n_loops_; }

    // Best current estimate for a node (corrected if optimized, else VIO pose).
    Eigen::Matrix4d corrected(int node_id) const;

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;      // VIO-frame initials (never overwritten)
    gtsam::Values corrected_;   // latest optimize() result (+ VIO pose for new nodes)
    int prev_id_ = -1;
    int n_loops_ = 0;

    LoopParams lp_;
    gtsam::SharedNoiseModel prior_noise_, odo_noise_, rp_noise_, loop_noise_;
};

} // namespace vio
