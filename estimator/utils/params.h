#pragma once
// Central VIO configuration (VINS-Mono style). Loaded once from a YAML file at
// startup; every field defaults to the exact value previously hardcoded, so a
// missing or partial YAML reproduces the committed behaviour bit-for-bit.
//
// Loader: utils/params.cpp (yaml-cpp). See config/vio_params.yaml for the file
// the binary reads and the documentation of each knob.

#include <string>
#include <vector>

namespace vio {

struct VioParams {
    // -------------------- runtime --------------------
    bool   dump_frontend     = false;  // write frontend_{frames,points}.csv
    bool   diag              = false;  // per-KF [viodiag] + [initdiag] diagnostic prints
    // Consistency outputs.
    bool   dump_covariance   = false;  // dump per-KF pose + 6x6 marginal covariance for NEES
    bool   dump_relcov       = false;  // dump per-KF relative pose + covariance (spans 1, 8 KF) for relative NEES
    bool   deterministic     = false;  // pin TBB to 1 thread for bit-reproducible runs (NEES needs it)
    int    rng_seed          = 0;      // seed of every OpenCV/std RANSAC (main and front-end threads).
                                       // 0 = the historical seed. Other values give runs that differ
                                       // only in random sampling (seed sets).
    // The window is keyframe-indexed. VioParams::load rejects configs that ask for a
    // time window (smoother_lag_s, window_coord: time).
    int    kf_window_size    = 10;     // sliding window length in KEYFRAMES (N_kf)
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
    std::string backend_mode  = "incremental"; // IncrementalFixedLagSmoother (iSAM2), the only backend
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
    std::string isam_factorization = "qr";  // ISAM2 factorization: "qr" (default, every earlier result) | "cholesky" (GTSAM's default)
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
    std::string pt_ransac_method = "ransac"; // frame-to-frame outlier rejection: ransac (OpenCV 8-point F
                                       // RANSAC, the historical default) | magsac (OpenCV USAC MAGSAC++
                                       // on F) | twopt (translation-only 2-point RANSAC on E with the gyro
                                       // rotation, Kimera-style; falls back to ransac without a rotation)
    double pt_ransac_conf    = 0.99;   // RANSAC confidence (ransac, magsac)
    int    pt_ransac_iters   = 2000;   // RANSAC iteration cap (ransac, magsac)
    double pt_quality        = 0.01;   // goodFeaturesToTrack quality level
    bool   pt_imu_aided_klt  = true;   // gyro-predict the KLT seed (image mode, threaded frontend)
    std::string pt_histeq    = "none"; // histogram equalization: none | equalize (global, OpenVINS EuRoC)
                                       // | clahe (adaptive). OFF by default (byte-identical); Mantis
                                       // images are already equalized so keep "none" there.
    double pt_clahe_clip     = 4.0;    // CLAHE clip limit (only when histeq=clahe); lower = gentler
    int    pt_grid_x         = 1;      // detection grid columns: 1 = global goodFeaturesToTrack (strongest
    int    pt_grid_y         = 1;      // corners image-wide, THE DEFAULT); >1 = OpenVINS-style per-cell
                                       // quota so features cover the frame instead of clumping in texture.
                                       // 5x5 is OpenVINS' value; targets the fast-rotation conditioning
                                       // (V1_01 NEES) the observability study traced to feature geometry.

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
    bool   init_est_ba            = false; // co-estimate accel bias in the gravity refinement so it
                                           // is not absorbed into gravity (fixes low-excitation init
                                           // tilt, e.g. V1_01). false = original behaviour (b_a=0).
    bool   init_oracle_gravity    = false; // Stage-0 DIAGNOSTIC only: rotate init attitude so the
                                           // first init KF's body-gravity matches GT (roll/pitch),
                                           // to measure the NEES ceiling of a perfect gravity init.
                                           // Uses ground truth -> NEVER for production.
    bool   init_continuous_window = true;  // ABLATION A5 (paper revision R1). true = push the ENTIRE
                                           // bootstrap sub-graph into the smoother so the seed keeps
                                           // re-optimizing until it slides out of the lag (the shipped
                                           // behaviour). false = VINS-Mono-style FROZEN seed: only the
                                           // LAST bootstrap keyframe enters, as a single hard-prior
                                           // state. Isolates the continuous-window contribution.
    // continuous-window VI-BA gauge priors
    double iw_pose0_sigma     = 1e-3;      // X(0) prior sigma (rot & trans)
    double iw_vel0_sigma      = 0.3;       // V(0) prior sigma [m/s]
    double iw_bias0_acc_sigma = 0.1;       // B(0) accel-bias prior sigma
    double iw_bias0_gyr_sigma = 0.02;      // B(0) gyro-bias prior sigma

    // -------------------- point reprojection factor --------------------
    double pt_huber_k        = 0.1;    // Huber threshold (whitened residual)
    double pt_sigma_px       = 1.0;    // isotropic pixel sigma
    // Heteroscedastic per-observation reprojection covariance (default OFF = byte-identical).
    // Weights each ONLINE reprojection factor by track age: sigma_obs = sigma_px *
    // clamp(age_full / age, 1, sigma_max). Young/short tracks (dominant on fast-rotation
    // sequences like V1_01) get inflated sigma -> less information over-counting. The
    // evidence-directed consistency lever after OC + grid both refuted (measurement model).
    bool   pt_hetero_cov       = false; // enable per-observation reprojection weighting
    std::string pt_hetero_metric = "age"; // "age" = track age (blunt: young tracks exist everywhere) |
                                          // "fb"  = forward-backward KLT residual (localization quality;
                                          //         inert -- the fb_err gate already truncates it) |
                                          // "parallax" = anchor<->obs triangulation angle (depth
                                          //         information; low parallax -> inflate -> targets the
                                          //         low-baseline fast-rotation over-confidence directly,
                                          //         and leaves well-translating obs at baseline)
    double pt_hetero_age_full  = 5.0;   // [age]      track age [frames] at/above which weight = 1
    double pt_hetero_fb_ref    = 0.3;   // [fb]       FB residual [px] at/below which weight = 1
    double pt_hetero_parallax_ref = 1.0;// [parallax] parallax angle [deg] at/above which weight = 1
    double pt_hetero_sigma_max = 4.0;   // max sigma multiplier

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
    double lc_odo_trans_sigma  = 0.05;   // pose-graph odometry edge translation sigma [m] (was hard-coded)
    double lc_odo_rot_sigma    = 0.02;   // pose-graph odometry edge rotation sigma [rad] (was hard-coded)
    bool   lc_odo_from_estimator  = false; // source the backbone edge noise from the estimator's relative covariance
    double lc_odo_sigma_inflation = 1.0;   // sigma multiplier on that covariance (chain compounding ignores correlation)
    double lc_odo_sigma_floor_m   = 0.0;   // additive translation sigma floor [m]; 0 = off
    double lc_odo_sigma_floor_rad = 0.0;   // additive rotation sigma floor [rad]; 0 = off
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
    // entirely (w = 1,0,0) is WORSE on ATE and NEES at every threshold, including at matched
    // keyframe rate.
    double kf_shift_weight_t  = 0.0;   // weight on TRANSLATION-only flow  (DSO 0.04)
    double kf_shift_weight_r  = 0.0;   // weight on ROTATION-only flow     (DSO 0.00)
    double kf_shift_weight_rt = 1.0;   // weight on FULL flow              (DSO 0.02)

    // -------------------- landmark delayed init --------------------
    double lm_min_baseline_m    = 0.10;  // min triangulation baseline [m]
    double lm_min_parallax_deg  = 0.0;   // min angle between the two triangulation rays at the point
                                         // [deg]. 0 = off. A scale-free gate: 2 cm at 1.5 m (MANTIS) and
                                         // 10 cm at 5 m (EuRoC) are both about 1 degree.
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
                                           // before the next re-host, ~N_kf times less key churn.
    // Landmark continuity. Both off = the plain system,
    // bit for bit. invdepth only, and neither combines with point_depth_transfer.
    bool   lm_init_current_obs    = false; // the observation of the keyframe at which a landmark
                                           // is (re)initialised becomes a factor. Off: it is not
                                           // in `live` (its pose is not in the snapshot yet) and is
                                           // erased with the track history, so it is never used.
    std::string lm_reanchor       = "abandon"; // anchor exit of a still-tracked landmark:
                                           // "abandon" = drop the variable and re-triangulate from
                                           // scratch (a chunk boundary);
                                           // "exact" = new variable anchored at the current keyframe,
                                           // tied to the old one by InvDepthReanchorFactor (a change
                                           // of variables), so the track keeps its
                                           // information and no measurement is used twice.
    double lm_reanchor_sigma      = 1.0e-4; // sigma of the re-anchor tie (rad, rad, 1/m). Must be
                                           // far below the landmark's own uncertainty (bearing
                                           // ~sigma_px/f ~ 2.6e-3 rad on MANTIS, 2.2e-3 on EuRoC).
    int    lm_reanchor_max        = 0;     // cap on consecutive exact re-anchors of one landmark
                                           // chain. 0 = unlimited (the switch as first built). At
                                           // the cap the anchor exit falls back to abandon, so the
                                           // track is re-triangulated as a fresh chain.
    bool   lm_reanchor_retire     = false; // exact re-anchoring: re-stamp the old variable with its
                                           // outgoing anchor's keyframe, so both leave the window in
                                           // the same update. Off: it keeps its last-seen stamp and
                                           // lingers up to N_kf keyframes, reachable only through the
                                           // marginal.
    int    lm_reanchor_live_max   = 0;     // cap on LIVE re-anchor chains (OpenVINS max_slam analog).
                                           // 0 = unlimited. When full, an anchor exit that would start
                                           // a chain falls back to abandon, running chains continue.

    // -------------------- extrinsics override --------------------
    // Optional replacement for the dataset's mav0/{cam0,imu0}/sensor.yaml T_BS blocks,
    // so an extrinsic can be corrected or swept WITHOUT re-extracting the dataset.
    // Each is 16 doubles, row-major 4x4. Empty (the default) leaves the dataset's own
    // sensor.yaml untouched, so every existing config reproduces bit-for-bit.
    //   ext_T_BC : camera-in-body (overrides cam0/sensor.yaml T_BS)
    //   ext_T_BI : imu-in-body    (overrides imu0/sensor.yaml T_BS)
    // "Body" is whichever frame the dataset's imu0 lives in -- on MANTIS that is the
    // IMU actually written to mav0/imu0, which is NOT necessarily imu_apps.
    std::vector<double> ext_T_BC;
    std::vector<double> ext_T_BI;

    // -------------------- ZUPT (stationary) --------------------
    double zupt_acc_var   = 0.1;    // accel variance threshold [(m/s^2)^2]
    double zupt_gyr_var   = 0.002;  // gyro variance threshold [(rad/s)^2]
    double zupt_disp_px   = 1.5;    // median feature displacement threshold [px]
    double zupt_vel_sigma = 0.01;   // zero-velocity prior sigma [m/s]

    // Load from YAML. On any read error returns built-in defaults (and warns).
    static VioParams load(const std::string& yaml_path);
};

} // namespace vio
