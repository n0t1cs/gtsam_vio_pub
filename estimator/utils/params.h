#pragma once
// Central VIO configuration (VINS-Mono style). Loaded once from a YAML file at startup
// (config/estimator_mantis2.yaml on the vehicle). A missing key keeps the default below.
//
// Loader: utils/params.cpp (yaml-cpp). Switches that were removed are accepted only at the value
// the system ran with, so an old config keeps its meaning and one that relied on a removed switch
// is refused instead of running as something else (VioParams::load).

#include <string>

namespace vio {

struct VioParams {
    // -------------------- runtime --------------------
    bool   dump_covariance   = false;  // per-KF pose + 6x6 marginal covariance (vio_cov.csv), and the
                                       // published covariance
    bool   dump_relcov       = false;  // per-KF relative pose + covariance (spans 1, 8 KF) for relative NEES
    bool   deterministic     = false;  // pin TBB to 1 thread for bit-reproducible runs (NEES needs it)
    int    rng_seed          = 0;      // seed of every OpenCV/std RANSAC (main and front-end threads).
                                       // 0 = the historical seed. Other values give runs that differ
                                       // only in random sampling (seed sets).
    // The window is ALWAYS keyframe-indexed: both backends hold the same N_kf window.
    int    kf_window_size    = 10;     // sliding window length in KEYFRAMES (N_kf), both backends
    bool   kf_use_fej        = true;   // backend_mode=keyframe ONLY: first-estimate Jacobians at the marginalization
                                       // boundary. true = linearize the marginalized subgraph at each
                                       // boundary variable's FROZEN first estimate (consistency);
                                       // false = at the current estimate (the noFEJ A/B arm).
    int    kf_max_iterations = 10;     // backend_mode=keyframe ONLY: Gauss-Newton iterations per update().
                                       // The filter<->smoother knob: 10 = a re-linearizing sliding-window
                                       // smoother; 1 = a single linearization per step, i.e. an EKF update.
    std::string kf_fej_scope = "xv";   // backend_mode=keyframe ONLY: WHICH states are held at first estimates.
                                       // "xv"  = pose+velocity only (the VI unobservable directions:
                                       //         global position + yaw) — biases/landmarks relinearize;
                                       // "xvb" = also freeze the IMU biases;
                                       // "all" = also freeze landmarks (widest, most linearization error).
    // Backend smoother. Code defaults are not the system: the committed configs select
    // backend_mode: incremental (iSAM2 on the keyframe window, the system default since 2026-10-08).
    std::string backend_mode  = "keyframe"; // "incremental" = IncrementalFixedLagSmoother (ISAM2, Cholesky)
                                            // "keyframe" = KeyframeSmoother (SLKS): standalone N_kf-keyframe
                                            //   window + Schur marginalization + switchable, scoped FEJ
    // ISAM2 knobs (backend_mode=incremental only).
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
    double pt_ransac_conf    = 0.99;   // RANSAC confidence
    int    pt_ransac_iters   = 2000;   // RANSAC iteration cap
    double pt_quality        = 0.01;   // goodFeaturesToTrack quality level
    std::string pt_histeq    = "none"; // histogram equalization: none | equalize (global, OpenVINS EuRoC)
                                       // | clahe (adaptive)
    double pt_clahe_clip     = 4.0;    // CLAHE clip limit (only when histeq=clahe); lower = gentler

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
    double init_max_vel_mps       = 6.0;   // VI-consistency gate: reject windows whose aligned
                                           // init velocity is implausible (bad scale -> divergence)
    // continuous-window VI-BA gauge priors
    double iw_pose0_sigma     = 1e-3;      // X(0) prior sigma (rot & trans)
    double iw_vel0_sigma      = 0.3;       // V(0) prior sigma [m/s]
    double iw_bias0_acc_sigma = 0.1;       // B(0) accel-bias prior sigma
    double iw_bias0_gyr_sigma = 0.02;      // B(0) gyro-bias prior sigma

    // -------------------- point reprojection factor --------------------
    double pt_huber_k        = 0.1;    // Huber threshold (whitened residual)
    double pt_sigma_px       = 1.0;    // isotropic pixel sigma

    // -------------------- keyframe gating --------------------
    double kf_parallax_px    = 30.0;   // avg parallax to trigger a keyframe [px]
    int    kf_min_tracked    = 40;     // force KF if fewer carried over
    int    kf_max_interval   = 10;     // force KF every N frames

    // -------------------- landmark delayed init --------------------
    double lm_min_baseline_m    = 0.10;  // min triangulation baseline [m]
    double lm_tri_depth_min     = 0.5;   // triangulate_dlt cheirality min [m]
    double lm_tri_depth_max     = 30.0;  // triangulate_dlt cheirality max [m]
    double lm_depth_min         = 0.3;   // accepted landmark depth min [m]
    double lm_depth_max         = 40.0;  // accepted landmark depth max [m]
    double lm_floor_prior_sigma = 1.0;   // point inverse-depth floor prior
    double pt_obs_reject_px     = 0.0;   // per-observation dynamic rejection (Kimera outlierRejection,
                                         // default 8 px there): before adding an InvDepthProjFactor for an
                                         // in-graph landmark, predict its pixel from the current estimate;
                                         // if reprojection error exceeds this [px] (or behind camera) the
                                         // OBSERVATION is dropped (no factor, no last-seen refresh). 0 = off.
    // Landmark continuity. invdepth only.
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
    bool   lm_reanchor_retire     = false; // exact re-anchoring: re-stamp the old variable with its
                                           // outgoing anchor's keyframe, so both leave the window in
                                           // the same update. Off: it keeps its last-seen stamp and
                                           // lingers up to N_kf keyframes, reachable only through the
                                           // marginal.
    int    lm_reanchor_live_max   = 0;     // cap on LIVE re-anchor chains (OpenVINS max_slam analog).
                                           // 0 = unlimited. When full, an anchor exit that would start
                                           // a chain falls back to abandon, running chains continue.

    // -------------------- ZUPT (stationary) --------------------
    double zupt_acc_var   = 0.1;    // accel variance threshold [(m/s^2)^2]
    double zupt_gyr_var   = 0.002;  // gyro variance threshold [(rad/s)^2]
    double zupt_disp_px   = 1.5;    // median feature displacement threshold [px] (0 = ZUPT off)
    double zupt_vel_sigma = 0.01;   // zero-velocity prior sigma [m/s]

    // Load from YAML. On a read error returns built-in defaults (and warns); a config that
    // relies on a removed switch exits with status 2.
    static VioParams load(const std::string& yaml_path);
};

} // namespace vio
