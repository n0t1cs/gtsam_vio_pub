#pragma once
// Central VIO configuration (VINS-Mono style). Loaded once from a YAML file at
// startup; every field defaults to the exact value previously hardcoded, so a
// missing or partial YAML reproduces the committed behaviour bit-for-bit.
//
// Loader: utils/params.cpp (yaml-cpp). See config/vio_params.yaml for the file
// the binary reads and the documentation of each knob.

#include <string>

namespace vio {

struct VioParams {
    // -------------------- runtime --------------------
    bool   dump_frontend     = false;  // write frontend_{frames,points}.csv
    bool   diag              = false;  // per-KF [viodiag] + [initdiag] diagnostic prints
    bool   dump_covariance   = false;  // dump per-KF pose + 6x6 marginal covariance for NEES
    bool   deterministic     = false;  // pin TBB to 1 thread for bit-reproducible runs (NEES needs it)
    double smoother_lag_s    = 3.0;    // fixed-lag window length [s] (batch/incremental)
    bool   loop_closure      = true;   // DBoW2 detection + 4-DoF pose graph (non-invasive:
                                       // est stays clean no-LC, lc = pose-graph trajectory).
                                       // false = pure VIO, no pose graph (no-LC A/B baseline)
    // Frontend source. DEFAULT is "folder": read pre-computed point features and
    // IMU from frontend_dir/{points,imu}.csv (extract_frontend_features.py output
    // — the PL-VINS/VINS-Mono 20Hz frontend bags). "images" = the legacy KLT tracker
    // on the dataset images (kept only as an ablation). Loop closure works in folder
    // mode: each feature frame is matched to its nearest EuRoC image for BRIEF.
    std::string frontend_source = "folder";  // "folder" (default system) | "images" (ablation)
    // frontend_dir: dir with points.csv/imu.csv. Left EMPTY in the shared
    // configs and AUTO-DERIVED per sequence as <dataset_parent>/<frontend_root>/<name>,
    // so one config works across every dataset without a per-dataset path.
    std::string frontend_dir    = "";
    std::string frontend_root   = "Euroc_frontend_20HZ";  // subdir under the dataset parent
                                                          // holding the per-sequence feature folders
    // Backend smoother + point representation (config-gated).
    std::string backend_mode  = "batch";    // "batch" = BatchFixedLagSmoother (LM, time lag), THE DEFAULT
                                            // "incremental" = IncrementalFixedLagSmoother (ISAM2)
    std::string point_backend = "invdepth"; // "invdepth" = explicit inverse-depth landmarks
                                            // "smart" = SmartProjectionPoseFactor (marginalized);
                                            //           REQUIRES backend_mode=incremental
                                            // "world" = world-frame Point3 + GenericProjectionFactor
                                            //           (OKVIS/SVO-pro parametrization). NO anchor,
                                            //           so a marginalized keyframe never orphans a
                                            //           landmark and no re-anchoring/depth transfer
                                            //           is needed. The L4 control arm.
    // ISAM2 knobs (backend_mode=incremental only). Lower relin_threshold = more
    // frequent relinearization → nearer batch accuracy (corrects stale factor
    // linearizations), at higher cost. Sweep these to close the batch gap.
    double isam_relin_threshold = 0.01;     // ISAM2 relinearizeThreshold
    int    isam_relin_skip      = 1;        // ISAM2 relinearizeSkip (relinearize every Nth update)
    // Bias sanity clamp (incremental robustness). If the smoother's post-update IMU
    // bias exceeds these physical bounds the solve is diverging (accel bias
    // absorbing real acceleration); the preintegrator is NOT reseeded with it -- the
    // last sane bias is kept, breaking the bias->prediction->velocity runaway.
    // Healthy runs sit at |ba|<=0.54; MH_04 divergence hits |ba|>1.4. 0 disables.
    double bias_accel_max = 1.0;            // reject post-update accel bias [m/s^2] above this
    double bias_gyro_max  = 0.5;            // reject post-update gyro  bias [rad/s] above this

    // -------------------- point front-end (FeatureTracker) --------------------
    int    pt_max            = 150;    // max KLT points
    int    pt_min_dist_px    = 30;     // min spacing between points [px]
    double pt_fb_err         = 1.0;    // forward-backward consistency [px]
    double pt_ransac_th      = 1.0;    // F-matrix RANSAC threshold [px]
    double pt_quality        = 0.01;   // goodFeaturesToTrack quality level
    bool   pt_imu_aided_klt  = true;   // gyro-predict the KLT seed (image mode, threaded frontend)

    // -------------------- IMU noise (dataset densities x code multipliers) --------------------
    // Densities are copied from the EuRoC imu0/sensor.yaml. The covariance
    // multipliers below are the inflation the code applies (variance domain).
    double imu_acc_n              = 2.0000e-3;  // accelerometer_noise_density
    double imu_gyr_n              = 1.6968e-4;  // gyroscope_noise_density
    double imu_acc_w              = 3.0000e-3;  // accelerometer_random_walk
    double imu_gyr_w              = 1.9393e-5;  // gyroscope_random_walk
    double imu_acc_cov_mult       = 100.0;      // x acc_n^2  (sigma x10)
    double imu_gyr_cov_mult       = 10.0;       // x gyr_n^2
    double imu_acc_bias_cov_mult  = 100.0;      // x acc_w^2
    double imu_gyr_bias_cov_mult  = 10.0;       // x gyr_w^2
    double imu_integration_cov    = 1e-4;       // setIntegrationCovariance
    double imu_bias_acc_omega_int = 1e-5;       // CombinedImuFactor cross-term

    // -------------------- initialization --------------------
    int    init_window            = 10;    // InitBuffer keyframe window
    double init_parallax_px       = 10.0;  // InitBuffer keyframe parallax gate [px]
    int    init_min_common        = 20;    // InitBuffer min common tracks
    double init_excitation_var    = 0.25;  // check_imu_excitation accel-var threshold
    double init_max_pre_ba_rms_px = 8.0;   // defer-until-well-conditioned gate [px]
                                           // (8px suits the 20Hz folder init windows; was 5px for images)
    double init_max_vel_mps       = 6.0;   // VI-consistency gate: reject windows whose aligned
                                           // init velocity is implausible (bad scale -> divergence)
    bool   init_continuous_window = true;  // ABLATION A5 (Drones revision). true = the shipped
                                           // behaviour: the ENTIRE bootstrap sub-graph enters the
                                           // smoother and keeps re-optimizing until it slides out
                                           // of the lag. false = VINS-Mono-style FROZEN seed: only
                                           // the LAST bootstrap keyframe is handed over, as one
                                           // hard-prior state. Default true => bit-identical to the
                                           // pre-patch binary (verified on arm A0).
    // continuous-window VI-BA gauge priors
    double iw_pose0_sigma     = 1e-3;      // X(0) prior sigma (rot & trans)
    double iw_vel0_sigma      = 0.3;       // V(0) prior sigma [m/s]
    double iw_bias0_acc_sigma = 0.1;       // B(0) accel-bias prior sigma
    double iw_bias0_gyr_sigma = 0.02;      // B(0) gyro-bias prior sigma

    // -------------------- point reprojection factor --------------------
    double pt_huber_k        = 0.1;    // Huber threshold (whitened residual)
    double pt_sigma_px       = 1.0;    // isotropic pixel sigma

    // -------------------- smart projection factor (point_backend=smart) --------------------
    // SmartProjectionPoseFactor<Cal3_S2> (mono) params; mirror Kimera's knobs.
    double smart_sigma_px      = 1.0;    // isotropic pixel noise
    double smart_rank_tol      = 1.0;    // triangulation rank tolerance
    double smart_landmark_dist = 10.0;   // reject landmarks farther than this [m] (Kimera default)
    double smart_retriang_th   = 1e-3;   // retriangulation threshold [m] (cache invalidation)
    double smart_outlier_px    = 3.0;    // dynamic outlier reprojection rejection [px]

    // -------------------- loop closure (DBoW2 detect + 4-DoF pose graph) --------------------
    // Detection gates (loop/loop_database). Defaults reproduce committed values.
    int    lc_skip_recent      = 50;     // ignore KFs within this temporal gap
    int    lc_fast_threshold   = 20;     // FAST corner threshold for the DBoW harvest (lower = more corners; low-texture recall)
    int    lc_min_inliers      = 30;     // PnP inlier gate
    int    lc_hamming_th       = 80;     // BRIEF match accept threshold (of 256)
    double lc_score_best       = 0.05;   // best DBoW score must exceed this
    double lc_score_cand       = 0.015;  // candidate DBoW score must exceed this
    double lc_fund_thresh_px   = 3.0;    // fundamental-matrix RANSAC px threshold
    double lc_max_trans_m      = 3.0;    // loop relative-translation sanity gate [m]
    double lc_max_yaw_deg      = 30.0;   // loop relative-yaw sanity gate [deg]
    // Pose-graph use (loop/pose_graph) — Front-B levers. Defaults = today's behaviour
    // (weighting off, gate off), so the committed config reproduces current LC.
    int    lc_backbone_hops    = 1;      // odometry edges per pose-graph node to its N predecessors (VINS-Mono=4); 1 = single-hop (committed)
    double lc_loop_trans_sigma = 0.1;    // base loop-edge translation sigma [m]
    double lc_loop_rot_sigma   = 0.05;   // base loop-edge rotation sigma [rad]
    double lc_cauchy_c         = 1.0;    // robust (Cauchy) kernel scale
    bool   lc_inlier_weighting = false;  // scale loop-edge sigma by sqrt(min_inliers/inliers)
    double lc_min_correction_m = 0.0;    // drift-consistency gate: add loop only if it corrects > this [m] (0 = off)
    double lc_loop_drift_rate  = 0.0;    // signal-vs-noise gate: skip loop if correction > rate*path_length (implausible drift = edge noise; 0 = off)

    // -------------------- keyframe gating --------------------
    double kf_parallax_px    = 30.0;   // avg parallax to trigger a keyframe [px]
    int    kf_min_tracked    = 40;     // force KF if fewer carried over
    int    kf_max_interval   = 10;     // force KF every N frames
    // DSO-style three-term flow score (FullSystem::needNewKf). The IMU-predicted rotation
    // splits the per-feature optical flow into a translation-only part (rotation warped out),
    // a rotation-only part, and the full displacement; the gate fires when
    //     w_t*flow_T + w_r*flow_R + w_rt*flow_RT >= kf_parallax_px.
    // Translation flow is the triangulation baseline; rotation flow carries NO baseline but
    // does destroy co-visibility, which is why DSO weights it zero on its own yet keeps the
    // combined term at half the translation weight (maxShiftWeightT/R/RT = 0.04/0.0/0.02).
    // Defaults (0,0,1) reproduce the plain raw-displacement gate exactly. Note the threshold
    // is in the same units only if (w_t + w_rt) == 1; scale kf_parallax_px by (w_t + w_rt) to
    // hold the pure-translation trigger point fixed. Measured: dropping the combined term
    // entirely (w = 1,0,0) is WORSE on ATE and NEES at every threshold, including at
    // matched keyframe rate.
    double kf_shift_weight_t  = 0.0;   // weight on TRANSLATION-only flow  (DSO 0.04)
    double kf_shift_weight_r  = 0.0;   // weight on ROTATION-only flow     (DSO 0.00)
    double kf_shift_weight_rt = 1.0;   // weight on FULL flow              (DSO 0.02)

    // -------------------- landmark delayed init --------------------
    double lm_min_baseline_m    = 0.10;  // min triangulation baseline [m]
    double lm_tri_depth_min     = 0.5;   // triangulate_dlt cheirality min [m]
    double lm_tri_depth_max     = 30.0;  // triangulate_dlt cheirality max [m]
    double lm_depth_min         = 0.3;   // accepted landmark depth min [m]
    double lm_depth_max         = 40.0;  // accepted landmark depth max [m]
    double lm_floor_prior_sigma = 1.0;   // point inverse-depth floor prior
    double lm_world_prior_sigma_m = 1.0; // point_backend=world: weak per-landmark world-frame
                                         // prior [m], the XYZ analog of the inverse-depth floor
                                         // prior. World points have no anchor to regularize them,
                                         // so this is the guard against the low-parallax
                                         // IndeterminantLinearSystem that killed the smart path.
    // Point-landmark bulletproofing (VINS-Mono / Kimera analogs). All <=0 = off.
    double pt_obs_reject_px       = 0.0;   // per-observation dynamic rejection (Kimera outlierRejection,
                                           // default 8 px there): before adding an InvDepthProjFactor for an
                                           // in-graph landmark, predict its pixel from the current estimate;
                                           // if reprojection error exceeds this [px] (or behind camera) the
                                           // OBSERVATION is dropped (no factor, no last-seen refresh).
    double pt_cull_depth_min      = 0.0;   // post-solve depth cull (VINS-Mono setDepth/removeFailures
    double pt_cull_depth_max      = 0.0;   // analog): after each update, a point landmark whose optimized
                                           // inverse depth went non-positive or whose depth left
                                           // [min,max] m stops receiving factors and marginalizes by lag.
    bool   pt_depth_transfer      = false; // VINS-Mono FeatureManager::removeBackShiftDepth analog:
                                           // when a landmark's anchor keyframe leaves the window,
                                           // TRANSFER its refined depth geometrically into the new
                                           // anchor camera instead of discarding it and waiting for a
                                           // fresh 2-view re-triangulation. false = legacy re-triangulate.
                                           // Also enables observation-history retention for in-graph
                                           // landmarks (VINS FeatureManager keeps every view), which the
                                           // re-host needs to re-attach the surviving cross-view factors.
    bool   pt_depth_transfer_oldest = true; // re-host on the OLDEST surviving view (VINS shifts frame
                                           // 1 -> 0: max remaining baseline, but re-hosts every keyframe).
                                           // false = re-host on the current keyframe: longest lifetime
                                           // before the next re-host, far less key churn.

    // -------------------- ZUPT (stationary) --------------------
    double zupt_acc_var   = 0.1;    // accel variance threshold [(m/s^2)^2]
    double zupt_gyr_var   = 0.002;  // gyro variance threshold [(rad/s)^2]
    double zupt_disp_px   = 1.5;    // median feature displacement threshold [px]
    double zupt_vel_sigma = 0.01;   // zero-velocity prior sigma [m/s]

    // Load from YAML. On any read error returns built-in defaults (and warns).
    static VioParams load(const std::string& yaml_path);
};

} // namespace vio
