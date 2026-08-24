#pragma once
#include "vio_types.h"
#include "params.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
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

    void add_imu(double dt, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro);

    // observations: feature id -> (u, v) pixel coords (undistorted)
    // apply_zupt: this frame is stationary -> pin velocity (and lock relative
    //             translation to the previous stationary keyframe).
    // Returns true if this frame became a keyframe (a new graph state).
    bool add_frame(const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
                   double t_s, bool apply_zupt);

    Eigen::Matrix4d get_current_pose();

    int  get_state_idx() const { return state_idx_; }
    void print_kf_stats() const;   // keyframe trigger breakdown

    // Loop-closure harvest: world-frame positions of currently in-graph point
    // landmarks (inverse-depth -> world via each landmark's anchor pose), keyed
    // by feature id. Uses the latest smoothed estimate; landmarks whose anchor
    // or value has marginalized out are skipped.
    void get_landmarks_world(std::unordered_map<int64_t, Eigen::Vector3d>& out) const;

    // --- TEMP diagnostics (read-only): current nav-state health + map sizes, for
    //     tracing V-room divergence (vel/bias runaway). ---
    struct Diag {
        double vel_norm = 0.0, ba_norm = 0.0, bg_norm = 0.0;
        int    n_pts_graph = 0;
    };
    Diag get_diag() const;

private:
    VioParams params_;                   // central config (set in ctor)
    double lag_;                         // sliding-window length [s]
    // Batch (LM) or incremental (ISAM2) fixed-lag smoother, chosen by
    // runtime.backend_mode. Both expose the same FixedLagSmoother interface
    // (update / calculateEstimate); the smart-point path additionally needs the
    // incremental one's getISAM2Result() for factor-slot bookkeeping.
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

    // ---- Smart-projection point backend (point_backend=smart; incremental only) ----
    // Points are NOT graph variables: each landmark lives inside one
    // SmartProjectionPoseFactor that connects only poses and Schur-marginalizes the
    // point at every linearization (Kimera design). Lifecycle mirrors Kimera:
    // buffer measurements until 2 views, create the factor, then clone+append on
    // each later keyframe (delete old slot, re-add). Slots are read back from the
    // ISAM2 result. See vio_backend.cpp add_frame() smart branch.
    using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;
    bool   smart_mode_ = false;          // resolved in ctor: point_backend==smart && incremental
    gtsam::SmartProjectionParams smart_params_;
    gtsam::SharedNoiseModel      smart_noise_;     // MUST be isotropic (SmartFactorBase requirement)
    gtsam::IncrementalFixedLagSmoother* isam_view_ = nullptr;  // non-owning view of smoother_ for getISAM2Result()
    gtsam::BatchFixedLagSmoother*       batch_view_ = nullptr; // non-owning view of smoother_ for marginalCovariance()
    // Backend dispatch: run the active solver.
    // (own window + marginalization, no timestamps); batch/incremental use smoother_.
    void          run_update(const gtsam::NonlinearFactorGraph& g, const gtsam::Values& v,
                             const gtsam::FixedLagSmoother::KeyTimestampMap& ts);
    gtsam::Values run_estimate() const;
    std::unordered_map<int64_t, std::vector<std::pair<int, Eigen::Vector2d>>> sf_track_;        // id -> buffered (state_idx, pixel) pre-creation
    std::unordered_map<int64_t, std::pair<SmartFactor::shared_ptr, int>>       sf_old_;          // id -> {factor, slot}; slot -1 = staged
    bool smart_point_update(const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
                            const gtsam::FixedLagSmoother::KeyTimestampMap& ts);

    // ---- World-frame point backend (point_backend=world) — the L4 control ----
    // Landmarks are world-frame Point3 variables observed through
    // GenericProjectionFactor (OKVIS / SVO-pro homogeneous-point design). They are
    // NOT anchored to a keyframe, so marginalizing a pose never orphans a landmark:
    // the Schur complement simply absorbs that pose's projection factors into the
    // marginal while the landmark variable survives with its accumulated
    // information. This removes the re-anchoring / depth-transfer problem entirely
    // rather than solving it. Landmarks leave only by last-seen staleness.
    // Kept as a SEPARATE update so the inverse-depth path stays byte-for-byte
    // unchanged and remains an exact control.
    bool world_mode_ = false;            // resolved in ctor: point_backend == world
    void world_point_update(const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
                            const gtsam::Values& current,
                            gtsam::FixedLagSmoother::KeyTimestampMap& ts,
                            double w,
                            std::vector<int64_t>& observed_points);

    // Fixed-lag window bookkeeping
    double current_time_ = 0.0;                  // latest state timestamp [s]
    std::map<int, double> state_time_;           // pose idx -> timestamp [s]
    std::unordered_map<int64_t, double> lm_last_seen_;  // landmark id -> last obs time
    std::unordered_map<int64_t, int>    lm_anchor_;     // landmark id -> anchor keyframe idx
    std::unordered_map<int64_t, std::uint64_t> lm_key_; // feature id -> landmark symbol index
    std::uint64_t next_lm_key_ = 0;                     // fresh key per (re)initialization

    // A pose/landmark is still in the window (not yet marginalized) iff its
    // timestamp is newer than current_time_ - lag_.
    bool in_window(double t) const { return (current_time_ - t) < lag_; }

    // Windowing coordinate for `state_idx`: the frame time t_s. Both back-ends are
    // time-lag fixed-lag smoothers, so the state index is not used.
    double win_coord(int /*state_idx*/, double t_s) const { return t_s; }

    // ---- Keyframe gating (non-KF frames add no graph state; pim_ keeps integrating) ----
    std::unordered_map<int64_t, Eigen::Vector2d> last_kf_obs_;  // reference for parallax test
    int    frames_since_kf_ = 0;
    double kf_parallax_px_  = 30.0;   // avg pixel parallax to trigger a keyframe
    int    kf_min_tracked_  = 40;     // force KF if fewer than this carried over
    int    kf_max_interval_ = 10;     // force KF every N frames (bounds pim_)
    // DSO-style three-term flow score: w_t*flow_T + w_r*flow_R + w_rt*flow_RT >= kf_parallax_px_,
    // where the IMU-predicted rotation splits the flow into translation-only / rotation-only /
    // full parts. (0,0,1) == the plain raw-displacement gate.
    double kf_w_t_          = 0.0;
    double kf_w_r_          = 0.0;
    double kf_w_rt_         = 1.0;
    // Diagnostics: mean per-decision flow, split three ways. rot/raw is how much of our
    // "parallax" was rotation carrying no triangulation baseline.
    double kf_par_raw_sum_  = 0.0;
    double kf_par_comp_sum_ = 0.0;
    double kf_par_rot_sum_  = 0.0;
    long   kf_par_n_        = 0;
    bool   prev_kf_stationary_ = false;  // for zero-relative-motion between stationary KFs
    bool   decide_keyframe(const std::unordered_map<int64_t, Eigen::Vector2d>& obs);

    // keyframe trigger diagnostics
    long   kf_first_ = 0, kf_interval_ = 0, kf_lowtrack_ = 0, kf_parallax_n_ = 0;

    // IMU preintegration-length diagnostics (read-only). Per keyframe: the interval
    // deltaTij; the first-order BIAS-CORRECTION displacement (how far re-predicting the
    // endpoint at the optimized bias moves it from the biasHat prediction -- the term
    // whose linearization error grows with the interval); and the factor's own claimed
    // sigma from preintMeasCov. Quantifies whether collapsing the between-keyframe IMU
    // into one long CombinedImuFactor is lossy at our keyframe rates.
    std::vector<double> imu_dt_, imu_bcorr_pos_, imu_bcorr_rot_, imu_cov_pos_, imu_cov_rot_;

    // NIS (Normalized Innovation Squared) consistency -- GT-free, so the only consistency
    // check available online on the real vehicle (feeds the global factor graph's trust in
    // this VIO's covariance). Two innovations, both gated on dump_covariance:
    //  A = reprojection PRE-FIT: per keyframe, existing-landmark obs evaluated at the
    //      IMU-predicted pose; S = H_obs P- H_obs^T + sigma_px^2 I (POSE-ONLY innovation
    //      cov -- anchor/landmark uncertainty omitted -> S understated -> NIS conservative).
    //  B = IMU-PREDICTION pose: nu = Log(T_pred^-1 T_opt), S = preintMeasCov pose block,
    //      dof 6. Optimistic: T_pred (prior) and T_opt (posterior) share the graph.
    // Each stored as NIS/dof (target ~1) plus a per-KF within-95% chi-square gate flag.
    std::vector<double> nis_a_ndof_, nis_a_gate_, nis_b_ndof_, nis_b_gate_;

    // Post-fit reprojection residual diagnostic (vision-side covariance calibration).
    // Per keyframe, this KF's reprojection factors evaluated at the OPTIMIZED pose:
    //   reproj_rms_  = RMS residual [px] per component
    //   reproj_chi2_ = R-normalized chi2 per dof (||r||^2 / sigma_px^2), target ~1.
    // RMS >> sigma_px => sigma_px mis-specified; RMS ~ sigma_px yet NEES over-confident
    // => visual information over-counting (correlated tracks), not per-measurement noise.
    std::vector<double> reproj_rms_, reproj_chi2_;

    long   pt_obs_rejected_ = 0;   // point observations dropped by dynamic reprojection reject
    long   pt_cull_removed_ = 0;   // point landmarks soft-removed by the post-solve depth cull
    long   pt_depth_transferred_ = 0;  // landmarks re-anchored by geometric depth transfer

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