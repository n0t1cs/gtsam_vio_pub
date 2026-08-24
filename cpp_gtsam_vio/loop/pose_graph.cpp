#include "pose_graph.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/base/numericalDerivative.h>

#include <algorithm>
#include <functional>
#include <iostream>

namespace vio {

using gtsam::symbol_shorthand::X;

// Unary factor pinning a pose's roll & pitch to fixed (VIO) targets. With this
// on every node, the only rotational DoF the graph can move is yaw -> 4-DoF.
class RollPitchFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
    double roll0_, pitch0_;
public:
    RollPitchFactor(gtsam::Key k, double roll0, double pitch0,
                    const gtsam::SharedNoiseModel& m)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(m, k), roll0_(roll0), pitch0_(pitch0) {}

    gtsam::Vector residual(const gtsam::Pose3& p) const {
        const gtsam::Vector3 rpy = p.rotation().rpy();   // [roll, pitch, yaw]
        return (gtsam::Vector(2) << rpy(0) - roll0_, rpy(1) - pitch0_).finished();
    }

    gtsam::Vector evaluateError(const gtsam::Pose3& p,
        boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(
            std::bind(&RollPitchFactor::residual, this, std::placeholders::_1), p);
        return residual(p);
    }
};

static gtsam::Pose3 eig2pose(const Eigen::Matrix4d& T) {
    return gtsam::Pose3(gtsam::Rot3(T.block<3, 3>(0, 0)), gtsam::Point3(T.block<3, 1>(0, 3)));
}
static Eigen::Matrix4d pose2eig(const gtsam::Pose3& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = p.rotation().matrix();
    T.block<3, 1>(0, 3) = p.translation();
    return T;
}

// Build a robust loop-edge noise model from a rotation/translation sigma pair.
static gtsam::SharedNoiseModel make_loop_noise(double rot_sig, double trans_sig, double cauchy_c) {
    auto base = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << rot_sig, rot_sig, rot_sig,
                             trans_sig, trans_sig, trans_sig).finished());
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(cauchy_c), base);
}

PoseGraph::PoseGraph(const LoopParams& lp) : lp_(lp) {
    prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
    odo_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.02, 0.02, 0.02, 0.05, 0.05, 0.05).finished());
    rp_noise_ = gtsam::noiseModel::Isotropic::Sigma(2, 1e-3);
    loop_noise_ = make_loop_noise(lp_.loop_rot_sigma, lp_.loop_trans_sigma, lp_.cauchy_c);
}

void PoseGraph::add_keyframe(int node_id, const Eigen::Matrix4d& T_wb) {
    const gtsam::Pose3 pose = eig2pose(T_wb);
    values_.insert(X(node_id), pose);
    corrected_.insert(X(node_id), pose);   // best estimate until next optimize()

    if (prev_id_ < 0) {
        graph_.addPrior(X(node_id), pose, prior_noise_);   // gauge anchor
    } else {
        // Multi-hop odometry backbone (VINS-Mono ties each node to its 4
        // predecessors, not just the immediate one). Extra hops stiffen the local
        // chain so a noisy loop edge distorts the trajectory less -- this is what
        // bounds the low-drift-room LC-hurt. Node ids are contiguous (kf.index),
        // so node_id-j is the j-th predecessor. backbone_hops=1 == single-hop
        // (committed behaviour, bit-identical). Same odo sigma on every hop, as in
        // VINS -- the deliberate over-stiffening is the point.
        const int hops = std::max(1, lp_.backbone_hops);
        for (int j = 1; j <= hops; ++j) {
            const int pid = node_id - j;
            if (pid < 0 || !values_.exists(X(pid))) break;
            const gtsam::Pose3 rel = values_.at<gtsam::Pose3>(X(pid)).between(pose);
            graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(pid), X(node_id), rel, odo_noise_);
        }
    }

    const gtsam::Vector3 rpy = pose.rotation().rpy();
    graph_.emplace_shared<RollPitchFactor>(X(node_id), rpy(0), rpy(1), rp_noise_);
    prev_id_ = node_id;
}

void PoseGraph::add_loop(const LoopEdge& e) {
    if (!values_.exists(X(e.index_old)) || !values_.exists(X(e.index_cur))) return;
    const gtsam::Pose3 meas = eig2pose(e.T_old_cur);

    // Front B — drift-consistency gate. The odometry chain between the two nodes
    // already encodes one relative pose; the loop measures another. Their
    // disagreement IS the accumulated drift the loop would correct. If that is
    // below the loop edge's own (PnP) noise floor, closing the loop can only
    // inject error -> skip it. This is what stops a valid-but-pointless loop from
    // hurting a low-drift sequence (the V-room failure). min_correction_m=0 = off.
    const gtsam::Pose3 odo = values_.at<gtsam::Pose3>(X(e.index_old))
                                 .between(values_.at<gtsam::Pose3>(X(e.index_cur)));
    const double correction = (odo.translation() - meas.translation()).norm();

    if (lp_.min_correction_m > 0.0 && correction < lp_.min_correction_m) {
        std::cout << "[loop] gate: KF " << e.index_cur << "<->" << e.index_old
                  << "  correction=" << correction << "m < " << lp_.min_correction_m
                  << "m -> skipped (drift below loop-edge noise)\n";
        return;
    }

    // Signal-vs-noise gate. The correction a loop demands should be consistent
    // with the odometry drift plausibly accumulated over the loop's path. A
    // short-path loop demanding a large correction cannot be drift -- it is
    // edge (PnP) noise, and applying it injects that noise into an already-good
    // trajectory (the Vicon-room LC-hurt: GT diagnosis showed harmful loops span
    // 50-80 KF, helpful ones 200-600 KF). loop_drift_rate=0 = off.
    if (lp_.loop_drift_rate > 0.0) {
        double path = 0.0;
        for (int k = e.index_old; k < e.index_cur; ++k)
            if (values_.exists(X(k)) && values_.exists(X(k + 1)))
                path += (values_.at<gtsam::Pose3>(X(k + 1)).translation()
                         - values_.at<gtsam::Pose3>(X(k)).translation()).norm();
        const double plausible = lp_.loop_drift_rate * path;
        if (correction > plausible) {
            std::cout << "[loop] snr-gate: KF " << e.index_cur << "<->" << e.index_old
                      << "  correction=" << correction << "m > " << plausible
                      << "m (=" << lp_.loop_drift_rate << "*path " << path
                      << "m) -> skipped (implausible drift for span)\n";
            return;
        }
    }

    // Front B — inlier-weighted noise. Tighten the loop edge as the PnP inlier
    // count rises above the gate; a marginal (min_inliers) loop stays at the base
    // sigma, a strong loop is trusted more. inlier_weighting=false = fixed sigma.
    gtsam::SharedNoiseModel nm = loop_noise_;
    if (lp_.inlier_weighting && e.inliers > lp_.min_inliers) {
        const double s = std::sqrt((double)lp_.min_inliers / (double)e.inliers);
        nm = make_loop_noise(lp_.loop_rot_sigma * s, lp_.loop_trans_sigma * s, lp_.cauchy_c);
    }

    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(e.index_old), X(e.index_cur), meas, nm);
    ++n_loops_;
}

void PoseGraph::optimize() {
    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(50);
    try {
        corrected_ = gtsam::LevenbergMarquardtOptimizer(graph_, values_, params).optimize();
    } catch (const std::exception& ex) {
        std::cerr << "[loop] PGO failed: " << ex.what() << " (keeping previous estimate)\n";
    }
}

Eigen::Matrix4d PoseGraph::corrected(int node_id) const {
    if (corrected_.exists(X(node_id))) return pose2eig(corrected_.at<gtsam::Pose3>(X(node_id)));
    if (values_.exists(X(node_id)))    return pose2eig(values_.at<gtsam::Pose3>(X(node_id)));
    return Eigen::Matrix4d::Identity();
}

} // namespace vio
