#pragma once
#include "vio_types.h"
#include "params.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include "keyframe_smoother.h"

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vio {

class VioBackend {
public:
    VioBackend(const Eigen::Matrix3d& K,
               const Eigen::Matrix4d& T_BC,
               boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params,
               const VioParams& params);

    void initialize(const Eigen::Matrix4d& T_wb,
                    const Eigen::Vector3d& vel_w,
                    const Eigen::Vector3d& bg,
                    const Eigen::Vector3d& ba = Eigen::Vector3d::Zero(),
                    double t_s = 0.0);

    // Continuous-window VI-BA seed. Instead of
    // seeding a single state, push the ENTIRE bootstrap window into the smoother:
    // every init keyframe (X,V,B), a CombinedImuFactor between consecutive ones,
    // and the bootstrap landmarks as inverse-depth vision factors. The fixed-lag
    // smoother then keeps re-optimizing these states (and the early online ones,
    // warm-started by the bootstrap map) every keyframe until each slides out of
    // the lag window. Online processing continues from the last init state.
    void initialize_window(const std::vector<Eigen::Matrix4d>& T_wb,
                           const std::vector<Eigen::Vector3d>& vels,
                           const std::vector<KeyframeEntry>& frames,
                           const std::map<int64_t, Eigen::Vector3d>& points,
                           const Eigen::Vector3d& bg,
                           const Eigen::Vector3d& ba,
                           const std::vector<ImuSample>& imu,
                           std::int64_t t0_ns);

    // Refined body pose of an in-window state (for writing the init-window block
    // to the CSV). Returns false if the state has already marginalized out.
    bool get_pose_at(int idx, Eigen::Matrix4d& out) const;

    // 6x6 marginal covariance of pose X(idx), tangent ordering [rot(3), trans(3)]
    // (GTSAM Pose3 convention). For the NEES consistency study; must be queried
    // ONLINE while the state is still in the window. Returns false if the state
    // has marginalized out or the linear system is indeterminate.
    bool get_pose_covariance(int idx, Eigen::Matrix<double, 6, 6>& out) const;

    // 6x6 covariance of the RELATIVE pose X(ia)^-1 * X(ib), tangent ordering
    // [rot(3), trans(3)]. This is what a pose-graph BetweenFactor needs, and it is
    // NOT obtainable from the two separate marginals: those share the accumulated
    // drift and the unobservable global position/yaw, so differencing them
    // double-counts the common-mode part. Taken from the 12x12 JOINT and pushed
    // through the between() Jacobians, so the cross-block cancels what it should.
    // Both states must still be in the window -> query ONLINE, at insertion.
    // Returns false if either state has marginalized out or the system is
    // indeterminate.
    bool get_relative_pose_covariance(int ia, int ib,
                                      Eigen::Matrix<double, 6, 6>& out) const;

    // The two queries above, answered on another thread. A snapshot holds what they read: the
    // window's factors (shared, no factor changes once built) and copies of the linearization
    // point and of the estimate's poses. One gtsam::Marginals of it serves every query of a
    // keyframe. On the SLKS the numbers are the live queries' bit for bit (each live query builds
    // that same Marginals). iSAM2's live pose query reads its Bayes tree instead, which a snapshot
    // cannot share (ISAM2::marginalizeLeaves edits conditionals in place): it is cheap, so
    // callers keep it live (cheap_pose_covariance).
    struct CovSnapshot {
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values theta;   // linearization point
        gtsam::Values poses;   // X(i) of the estimate (relative-pose Jacobians)
    };
    bool covariance_snapshot(CovSnapshot& s) const;
    bool cheap_pose_covariance() const { return isam_view_ != nullptr; }
    static bool pose_covariance(const CovSnapshot& s, const gtsam::Marginals& m, int idx,
                                Eigen::Matrix<double, 6, 6>& out);
    static bool relative_pose_covariance(const CovSnapshot& s, const gtsam::Marginals& m, int ia, int ib,
                                         Eigen::Matrix<double, 6, 6>& out);

    void add_imu(double dt, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro);

    // observations: feature id -> (u, v) pixel coords (undistorted)
    // apply_zupt: this frame is stationary -> pin velocity (and lock relative
    //             translation to the previous stationary keyframe).
    // Returns true if this frame became a keyframe (a new graph state).
    bool add_frame(const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
                   double t_s, bool apply_zupt);

    Eigen::Matrix4d get_current_pose();

    // Online output (read-only, the estimate is unchanged): the newest keyframe's smoothed state,
    // and the state at the newest frame predicted from it with pim_. A non-keyframe only adds IMU
    // to pim_, so this prediction is exactly the prior the next keyframe would receive.
    // ok = false before the first state exists.
    struct NavOut {
        bool ok = false;
        Eigen::Matrix4d T_kf  = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_now = Eigen::Matrix4d::Identity();
        Eigen::Vector3d v_kf  = Eigen::Vector3d::Zero();
        Eigen::Vector3d v_now = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg    = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba    = Eigen::Vector3d::Zero();
    };
    NavOut get_nav_out() const;

    int  get_state_idx() const { return state_idx_; }

private:
    VioParams params_;                   // central config (set in ctor)
    double lag_;                         // window length in KEYFRAMES (= kf_window_size, N_kf)
    // iSAM2 fixed-lag smoother (backend_mode: incremental), driven through the
    // FixedLagSmoother interface (update / calculateEstimate); isam_view_ reaches its
    // covariances.
    std::unique_ptr<gtsam::FixedLagSmoother> smoother_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;

    gtsam::Pose3 body_P_cam_;
    Eigen::Matrix3d K_;
    gtsam::Cal3_S2::shared_ptr calib_;

    boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;
    gtsam::imuBias::ConstantBias current_bias_;
    std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;

    gtsam::SharedNoiseModel cam_noise_;

    int state_idx_;
    bool initialized_;

    gtsam::Values latest_estimate_;

    // Landmark bookkeeping (inverse-depth path)
    std::unordered_map<int64_t, std::vector<std::pair<int, Eigen::Vector2d>>> feature_obs_;
    std::unordered_set<int64_t> landmarks_in_graph_;

    gtsam::IncrementalFixedLagSmoother* isam_view_ = nullptr;  // non-owning view of smoother_
    // KeyframeSmoother is STANDALONE (not a gtsam::FixedLagSmoother), so it is owned
    // separately from smoother_. Exactly one of {smoother_, kf_smoother_} is non-null.
    std::unique_ptr<KeyframeSmoother> kf_smoother_;           // owning (backend_mode: keyframe)

    // Backend dispatch: run the active solver. keyframe uses the standalone smoother
    // (own window + marginalization, keyframe indices); incremental uses smoother_.
    void          run_update(const gtsam::NonlinearFactorGraph& g, const gtsam::Values& v,
                             const gtsam::FixedLagSmoother::KeyTimestampMap& ts);
    gtsam::Values run_estimate() const;

    // Fixed-lag window bookkeeping. Every "time" below is the window coordinate, which
    // is the KEYFRAME INDEX (win_coord); the names predate the removal of the time window.
    double current_time_ = 0.0;                  // latest state's keyframe index
    std::map<int, double> state_time_;           // pose idx -> keyframe index stamp
    std::unordered_map<int64_t, double> lm_last_seen_;  // landmark id -> keyframe index of last obs
    std::unordered_map<int64_t, int>    lm_anchor_;     // landmark id -> anchor keyframe idx
    std::unordered_map<int64_t, std::uint64_t> lm_key_; // feature id -> landmark symbol index
    std::uint64_t next_lm_key_ = 0;                     // fresh key per (re)initialization

    // The windowing coordinate is the KEYFRAME INDEX (Phase 2b), for both backends:
    // add_frame/initialize* stamp state_time_, lm_last_seen_, the smoother's
    // KeyTimestampMap and current_time_ with it, so the back-end's re-anchoring stays
    // locked to the smoother's count-window (else an evicted anchor leaves a dangling
    // inverse-depth key). The wall-clock time window was removed on 2026-10-08.
    double win_coord(int state_idx) const { return static_cast<double>(state_idx); }

    // ---- Keyframe gating (non-KF frames add no graph state; pim_ keeps integrating) ----
    std::unordered_map<int64_t, Eigen::Vector2d> last_kf_obs_;  // reference for parallax test
    int    frames_since_kf_ = 0;
    bool   prev_kf_stationary_ = false;  // for zero-relative-motion between stationary KFs
    bool   decide_keyframe(const std::unordered_map<int64_t, Eigen::Vector2d>& obs);

    // Exact re-anchoring (landmark.reanchor: exact).
    bool   reanchor_exact_ = false;             // resolved in ctor from params_.lm_reanchor
    gtsam::SharedNoiseModel reanchor_noise_;    // Isotropic(3, lm_reanchor_sigma)
    std::unordered_map<int64_t, int> lm_chain_reanchors_;  // track -> re-anchors of its current chain

    // helpers
    // get_pose with explicit Values cache: avoids re-querying ISAM2.
    // `cache` must be a snapshot taken BEFORE state_idx_ was inserted into
    // initial_estimates_ (i.e. the values returned by isam_.calculateEstimate()
    // at the top of add_frame). For state_idx_ itself, this falls back to
    // initial_estimates_.
    gtsam::Pose3 get_pose_cached(int idx, const gtsam::Values& cache) const;

    bool triangulate_dlt(int idx1, int idx2,
                         const Eigen::Vector2d& uv1, const Eigen::Vector2d& uv2,
                         const gtsam::Values& cache,
                         Eigen::Vector3d& X_w_out);
};

} // namespace vio
