#include "vio_backend.h"
#include "inv_depth_factor.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vio {

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;

using ProjFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

// 95% upper quantile of chi-square(k) via Wilson--Hilferty -- the within-95% NIS gate.
static inline double chi2_p95(double k) {
    const double a = 2.0 / (9.0 * k);
    const double x = 1.0 - a + 1.6448536269514722 * std::sqrt(a);   // 1.6449 = z_{0.95}
    return k * x * x * x;
}

// ISAM2 parameters for the incremental smoother. findUnusedFactorSlots MUST be
// true: it lets removed-factor slots be reused, which is the precondition for
// the smart-factor delete-and-re-add slot bookkeeping (Phase 0b). Without it,
// nullptr slots accumulate and getISAM2Result().newFactorsIndices drifts.
// relinearize_threshold / skip come from config. The threshold does not act as an
// FEJ (threshold 0 changes nothing). iSAM2's FEJ-like behaviour is
// its built-in marginal freeze (fixedVariables_).
static gtsam::ISAM2Params make_isam2_params(double relin_threshold, int relin_skip,
                                            const std::string& factorization) {
    gtsam::ISAM2Params p;
    p.findUnusedFactorSlots = true;
    p.relinearizeThreshold  = relin_threshold;
    p.relinearizeSkip       = relin_skip;
    // CHOLESKY (GTSAM's default) is what the committed configs select: at the
    // keyframe window it gives an estimate identical to QR on 11/11 EuRoC sequences
    // at about a quarter of the latency. QR stays
    // the code default so earlier configs reproduce. It was adopted under the removed
    // 3 s time window, where CHOLESKY let the accel bias absorb real acceleration on
    // MH_04's fast end segment.
    if (factorization == "cholesky") {
        p.factorization = gtsam::ISAM2Params::CHOLESKY;
    } else {
        if (factorization != "qr")
            std::cerr << "[VIO Backend] WARNING: isam_factorization='" << factorization
                      << "' unknown; falling back to qr.\n";
        p.factorization = gtsam::ISAM2Params::QR;
    }
    return p;
}

VioBackend::VioBackend(const Eigen::Matrix3d& K,
                       const Eigen::Matrix4d& T_BC,
                       boost::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params,
                       const VioParams& params)
    : params_(params),
      // Window length in keyframes, for both backends. gtsam::FixedLagSmoother uses its
      // KeyTimestampMap only to order and evict keys, so the back-end stamps it with the
      // KEYFRAME INDEX (win_coord()), exactly as for the KeyframeSmoother. The GTSAM lag
      // is N_kf, not N_kf - 1: findKeysBefore(current - lag) evicts stamps strictly
      // below, marginalizeOlderThan(current - N_kf) evicts indices strictly below, and
      // both marginalize AFTER their solve, so lag = N_kf holds the identical N_kf + 1
      // keyframe window.
      lag_(static_cast<double>(params.kf_window_size)),
      body_P_cam_(gtsam::Rot3(T_BC.block<3, 3>(0, 0)), gtsam::Point3(T_BC.block<3, 1>(0, 3))),
      K_(K),
      imu_params_(imu_params),
      state_idx_(0),
      initialized_(false) {

    // Backend smoother: incremental (ISAM2, the committed default) or the
    // KeyframeSmoother. The smart point backend requires the incremental one. The
    // batch backend (BatchFixedLagSmoother) was removed on 2026-10-08 and is rejected
    // by VioParams::load.
    if (params_.backend_mode == "incremental") {
        auto isam = std::make_unique<gtsam::IncrementalFixedLagSmoother>(
            lag_,
            make_isam2_params(params_.isam_relin_threshold, params_.isam_relin_skip,
                              params_.isam_factorization));
        isam_view_ = isam.get();          // non-owning view for getISAM2Result()/getFactors()
        smoother_  = std::move(isam);
        smart_mode_ = (params_.point_backend == "smart");
        std::cout << "[VIO Backend] smoother = IncrementalFixedLagSmoother (ISAM2), factorization = "
                  << (params_.isam_factorization == "cholesky" ? "CHOLESKY" : "QR") << "\n";
        if (smart_mode_)
            std::cout << "[VIO Backend] point_backend = smart (SmartProjectionPoseFactor)\n";
    } else if (params_.backend_mode == "kf_fej" || params_.backend_mode == "keyframe") {
        // The SLKS. STANDALONE keyframe-indexed fixed-lag smoother with
        // explicit Schur marginalization (keyframe_smoother.h) — NOT a
        // gtsam::FixedLagSmoother, so it is owned in its own pointer (kf_smoother_)
        // and smoother_ stays null; VioBackend drives both through run_update()/
        // run_estimate(). The window is N_kf keyframes, owned inside the smoother.
        // lag_ remains the back-end's OWN re-anchoring horizon (keyframe count), which
        // matches the smoother's window by construction. Supports the inverse-depth and
        // world point backends; smart requires incremental.

        // FEJ scope is a CALLER policy — the smoother is symbol-agnostic. Translate
        // the config string into a key predicate over our own symbol convention:
        // "xv" (default) = the states carrying the VI unobservable directions (global
        // position + yaw); "xvb" adds biases (observable and drifting, so freezing
        // them costs linearization quality for no consistency gain); "all" adds
        // landmarks. Empty predicate would mean "all keys", so build it explicitly.
        const std::string scope = params_.kf_fej_scope;
        KeyframeSmoother::FejKeyFilter fej_filter =
            [scope](gtsam::Key k) {
                if (scope == "all") return true;
                const unsigned char c = gtsam::Symbol(k).chr();
                if (scope == "xvb") return c == 'x' || c == 'v' || c == 'b';
                return c == 'x' || c == 'v';
            };

        const bool kf_use_gn = (params_.kf_optimizer == "gn");
        // Solve-guard: only LANDMARKS may be dropped when the window's linear system goes
        // indeterminate. A pose/velocity/bias degeneracy is a real failure and must NOT be
        // masked, so the predicate excludes them — the guard rethrows in that case.
        KeyframeSmoother::CullKeyFilter cull_filter =
            [](gtsam::Key k) { return gtsam::Symbol(k).chr() == 'l'; };
        // S1 lever: square-root (QR) vs normal-equation (Cholesky) Schur marginalization.
        const bool kf_qr_marg = (params_.kf_marg_elim == "qr");
        if (params_.kf_marg_elim != "qr" && params_.kf_marg_elim != "cholesky")
            std::cerr << "[VIO Backend] WARNING: kf_marg_elim='" << params_.kf_marg_elim
                      << "' unknown; falling back to cholesky.\n";
        kf_smoother_ = std::make_unique<KeyframeSmoother>(params_.kf_window_size,
                                                          params_.kf_use_fej,
                                                          std::move(fej_filter),
                                                          params_.kf_max_iterations,
                                                          kf_use_gn,
                                                          std::move(cull_filter),
                                                          kf_qr_marg);
        if (params_.kf_diag_cond) {
            kf_smoother_->setCondDiagnostics(true);
            std::cout << "[VIO Backend] S2 conditioning diagnostic ON — dense SVD per "
                         "marginalization; this run's timing is NOT Pareto-comparable.\n";
        }
        std::cout << "[VIO Backend] smoother = KeyframeSmoother (standalone, keyframe window N_kf="
                  << params_.kf_window_size
                  << ", FEJ=" << (params_.kf_use_fej ? "on" : "off")
                  << ", scope=" << params_.kf_fej_scope
                  << ", opt=" << (kf_use_gn ? "GN" : "LM")
                  << ", marg=" << (kf_qr_marg ? "QR" : "Cholesky") << ")\n";
        if (params_.point_backend == "smart")
            std::cerr << "[VIO Backend] WARNING: point_backend=smart needs backend_mode="
                         "incremental; keyframe smoother stays on inverse-depth.\n";
    } else {
        throw std::invalid_argument("[VIO Backend] backend_mode='" + params_.backend_mode +
                                    "' unknown (incremental | keyframe; batch was removed)");
    }
    // Echo the window actually built, so a campaign can grep back that its config took.
    if (smoother_)
        std::cout << "[VIO Backend] fixed-lag window = keyframe index, lag=" << lag_ << " KF\n";

    // World-frame point backend (L4 control): valid under every smoother, since it
    // needs nothing from ISAM2. Mutually exclusive with smart by construction
    // (both keyed off the same string).
    world_mode_ = (params_.point_backend == "world");
    if (world_mode_)
        std::cout << "[VIO Backend] point_backend = world (Point3 + GenericProjectionFactor, "
                     "no anchor, prior sigma " << params_.lm_world_prior_sigma_m << " m)\n";

    // Smart-factor params (mono). Noise MUST be isotropic; outlier rejection is a
    // SmartProjectionParams knob, not a robust kernel. HESSIAN linearization →
    // RegularHessianFactor (ISAM2-friendly); ZERO_ON_DEGENERACY → empty factor on
    // a still-untriangulable / degenerate track instead of crashing.
    smart_noise_ = gtsam::noiseModel::Isotropic::Sigma(2, params_.smart_sigma_px);
    smart_params_ = gtsam::SmartProjectionParams(gtsam::HESSIAN, gtsam::ZERO_ON_DEGENERACY,
                                                 /*throwCheirality=*/false,
                                                 /*verboseCheirality=*/false);
    smart_params_.setRankTolerance(params_.smart_rank_tol);
    smart_params_.setLandmarkDistanceThreshold(params_.smart_landmark_dist);
    smart_params_.setRetriangulationThreshold(params_.smart_retriang_th);
    smart_params_.setDynamicOutlierRejectionThreshold(params_.smart_outlier_px);

    // Camera calibration
    double fx = K(0, 0), fy = K(1, 1);
    double cx = K(0, 2), cy = K(1, 2);
    calib_ = boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0.0, cx, cy);

    // Preintegration (initial zero bias)
    current_bias_ = gtsam::imuBias::ConstantBias();
    pim_.reset(new gtsam::PreintegratedCombinedMeasurements(imu_params_, current_bias_));

    // Robust vision noise (points): heavy Huber robustification.
    cam_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params_.pt_huber_k),
        gtsam::noiseModel::Isotropic::Sigma(2, params_.pt_sigma_px));

    // Landmark continuity (both off = the committed system; VioParams::load rejects
    // unsupported combinations).
    reanchor_exact_ = (params_.lm_reanchor == "exact");
    if (reanchor_exact_)
        reanchor_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, params_.lm_reanchor_sigma);
    if (reanchor_exact_ || params_.lm_init_current_obs)
        std::cout << "[VIO Backend] landmark continuity: reanchor=" << params_.lm_reanchor
                  << " (sigma " << params_.lm_reanchor_sigma << ")  init_current_obs="
                  << (params_.lm_init_current_obs ? "on" : "off")
                  << "  retire=" << (params_.lm_reanchor_retire ? "on" : "off")
                  << "  live_max=" << params_.lm_reanchor_live_max << "\n";

    // Keyframe gating thresholds from config.
    kf_parallax_px_  = params_.kf_parallax_px;
    kf_min_tracked_  = params_.kf_min_tracked;
    kf_max_interval_ = params_.kf_max_interval;
    kf_w_t_          = params_.kf_shift_weight_t;
    kf_w_r_          = params_.kf_shift_weight_r;
    kf_w_rt_         = params_.kf_shift_weight_rt;
    if (kf_w_t_ != 0.0 || kf_w_r_ != 0.0)
        std::cout << "[VIO Backend] keyframe gate = DSO three-term flow score  w(T,R,RT)=("
                  << kf_w_t_ << "," << kf_w_r_ << "," << kf_w_rt_ << ")  threshold="
                  << kf_parallax_px_ << "px\n";
}

// Per-observation reprojection noise (heteroscedastic-covariance lever). OFF ->
// the shared cam_noise_ (byte-identical). ON -> inflate sigma for young/short tracks
// (age < age_full), which dominate fast-rotation sequences and drive the reprojection
// over-confidence the OC + grid studies isolated to the measurement model. Not found
// (age unknown) -> full trust.
gtsam::SharedNoiseModel VioBackend::cam_noise_for(int64_t id) const {
    if (!params_.pt_hetero_cov) return cam_noise_;
    // Parallax is geometry-based and applied at the cross-view sites; here (anchor's
    // own factor, or any id-only caller) it has no parallax -> baseline weight.
    if (params_.pt_hetero_metric == "parallax") return cam_noise_;
    auto it = obs_quality_.find(id);
    const bool have = (it != obs_quality_.end());
    double mult;
    if (params_.pt_hetero_metric == "fb") {
        // FB residual [px]: high (hard/fast-rotation tracking) -> inflate. Missing/fresh -> 0 -> full trust.
        double fb = have ? it->second : 0.0;
        mult = fb / params_.pt_hetero_fb_ref;
    } else {
        // Track age [frames]: young -> inflate. Missing -> age_full -> full trust.
        double age = have ? it->second : params_.pt_hetero_age_full;
        mult = params_.pt_hetero_age_full / std::max(age, 1.0);
    }
    if (mult < 1.0) mult = 1.0;
    if (mult > params_.pt_hetero_sigma_max) mult = params_.pt_hetero_sigma_max;
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params_.pt_huber_k),
        gtsam::noiseModel::Isotropic::Sigma(2, params_.pt_sigma_px * mult));
}

// Parallax metric: the anchor<->observer triangulation angle carries the depth
// information of THIS cross-view factor. A small angle (low translational baseline,
// e.g. fast pure rotation) means the observation barely constrains rho yet would be
// weighted at full sigma_px -> over-confidence. Inflate below parallax_ref; a
// well-translating view (large angle) stays at baseline, so healthy sequences are
// left untouched -- the isolation that age (young-everywhere) and fb (gate-truncated)
// could not achieve.
gtsam::SharedNoiseModel VioBackend::cam_noise_parallax(double angle_deg) const {
    double a = std::max(angle_deg, 1e-6);
    double mult = params_.pt_hetero_parallax_ref / a;
    if (mult < 1.0) mult = 1.0;
    if (mult > params_.pt_hetero_sigma_max) mult = params_.pt_hetero_sigma_max;
    return gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Huber::Create(params_.pt_huber_k),
        gtsam::noiseModel::Isotropic::Sigma(2, params_.pt_sigma_px * mult));
}

// Angle [deg] subtended at world point Pw between the rays from two camera centers.
static double parallax_angle_deg(const gtsam::Point3& Pw,
                                 const gtsam::Point3& c0, const gtsam::Point3& c1) {
    const gtsam::Vector3 r0 = Pw - c0, r1 = Pw - c1;
    const double n0 = r0.norm(), n1 = r1.norm();
    if (n0 < 1e-9 || n1 < 1e-9) return 0.0;
    double c = r0.dot(r1) / (n0 * n1);
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / M_PI;
}

// Backend dispatch. KeyframeSmoother owns a keyframe-indexed window; incremental is
// a gtsam::FixedLagSmoother over the same keyframe index. Exactly one of
// {kf_smoother_, smoother_} is active.
//
// Both consume the SAME per-key stamp map: every stamp written by
// initialize*/add_frame is win_coord() = the KEYFRAME INDEX
// (see the ts[...] = w sites), so the conversion below is exact, not a
// reinterpretation. That is what lets the smoother stay symbol-agnostic: the caller
// already encodes "stamp once at creation" for states and "re-stamp on every
// observation" for landmarks, which is precisely the eviction policy each needs.
void VioBackend::run_update(const gtsam::NonlinearFactorGraph& g, const gtsam::Values& v,
                            const gtsam::FixedLagSmoother::KeyTimestampMap& ts) {
    if (kf_smoother_) {
        KeyframeSmoother::KeyIndexMap idx;
        for (const auto& kv : ts)
            idx[kv.first] = static_cast<std::uint64_t>(kv.second);
        kf_smoother_->update(g, v, idx);
    } else {
        smoother_->update(g, v, ts);
    }
}

gtsam::Values VioBackend::run_estimate() const {
    return kf_smoother_ ? kf_smoother_->calculateEstimate() : smoother_->calculateEstimate();
}

void VioBackend::initialize(const Eigen::Matrix4d& T_wb,
                            const Eigen::Vector3d& vel_w,
                            const Eigen::Vector3d& bg,
                            const Eigen::Vector3d& ba,
                            double t_s) {
    gtsam::Pose3 pose(gtsam::Rot3(T_wb.block<3, 3>(0, 0)),
                      gtsam::Point3(T_wb.block<3, 1>(0, 3)));
    gtsam::imuBias::ConstantBias bias(ba, bg);

    current_bias_ = bias;
    pim_->resetIntegrationAndSetBias(bias);

    initial_estimates_.insert(X(state_idx_), pose);
    initial_estimates_.insert(V(state_idx_), gtsam::Vector3(vel_w));
    initial_estimates_.insert(B(state_idx_), bias);

    Eigen::Matrix<double, 6, 1> pose_sigmas;
    pose_sigmas.setConstant(1e-5);
    auto prior_pose = gtsam::noiseModel::Diagonal::Sigmas(pose_sigmas);
    auto prior_vel = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4);
    auto prior_bias = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);

    graph_.addPrior(X(state_idx_), pose, prior_pose);
    graph_.addPrior(V(state_idx_), gtsam::Vector3(vel_w), prior_vel);
    graph_.addPrior(B(state_idx_), bias, prior_bias);

    // Windowing coordinate: keyframe index in kf_fej (Phase 2b), else time.
    const double w = win_coord(state_idx_);
    gtsam::FixedLagSmoother::KeyTimestampMap ts;
    ts[X(state_idx_)] = w;
    ts[V(state_idx_)] = w;
    ts[B(state_idx_)] = w;

    run_update(graph_, initial_estimates_, ts);
    latest_estimate_ = run_estimate();

    graph_.resize(0);
    initial_estimates_.clear();
    initialized_ = true;

    current_time_ = w;
    state_time_[state_idx_] = w;
    std::cout << "[VIO Backend] Initialized smoother (lag=" << lag_
              << " KF) at state " << state_idx_ << "\n";
}

void VioBackend::initialize_window(const std::vector<Eigen::Matrix4d>& T_wb,
                                   const std::vector<Eigen::Vector3d>& vels,
                                   const std::vector<KeyframeEntry>& frames,
                                   const std::map<int64_t, Eigen::Vector3d>& points,
                                   const Eigen::Vector3d& bg,
                                   const Eigen::Vector3d& ba,
                                   const std::vector<ImuSample>& imu,
                                   std::int64_t t0_ns) {
    const int N = (int)T_wb.size();
    // Single-state seed instead of the continuous window, in two cases:
    //   (a) the window is degenerate (too short / inconsistent) -- always;
    //   (b) init.continuous_window == false -- ABLATION A5 (paper revision R1),
    //       which reproduces the VINS-Mono handoff: the bootstrap is FROZEN at the
    //       last keyframe and handed over as one hard-prior state, so the smoother
    //       never re-optimizes the seed. This is the control arm that isolates the
    //       continuous-window contribution.
    if (!params_.init_continuous_window ||
        N < 3 || (int)vels.size() != N || (int)frames.size() != N) {
        if (!params_.init_continuous_window)
            std::cout << "[initdiag] continuous_window=OFF -> frozen single-state seed "
                         "at bootstrap KF " << (N > 0 ? N - 1 : 0) << " (ablation A5)\n";
        initialize(T_wb.back(), vels.back(), bg, ba,
                   frames.empty() ? 0.0 : frames.back().t);
        return;
    }

    const double fx = K_(0, 0), fy = K_(1, 1), cx = K_(0, 2), cy = K_(1, 2);
    const gtsam::imuBias::ConstantBias bias0(ba, bg);
    current_bias_ = bias0;

    gtsam::FixedLagSmoother::KeyTimestampMap ts;

    // 1) Insert every init-keyframe navigation state.
    for (int k = 0; k < N; ++k) {
        gtsam::Pose3 pose(gtsam::Rot3(T_wb[k].block<3, 3>(0, 0)),
                          gtsam::Point3(T_wb[k].block<3, 1>(0, 3)));
        initial_estimates_.insert(X(k), pose);
        initial_estimates_.insert(V(k), gtsam::Vector3(vels[k]));
        initial_estimates_.insert(B(k), bias0);
        const double w_k = win_coord(k);   // KF index in kf_fej, else time
        ts[X(k)] = ts[V(k)] = ts[B(k)] = w_k;
        state_time_[k] = w_k;
    }

    // 2) Gauge anchor on the first state. X0 tight (fixes the 4 unobservable
    //    DOF + observable roll/pitch); V0/B0 soft so the window can refine them.
    const double ps = params_.iw_pose0_sigma;
    Eigen::Matrix<double, 6, 1> x0s; x0s << ps, ps, ps, ps, ps, ps;
    graph_.addPrior(X(0), initial_estimates_.at<gtsam::Pose3>(X(0)),
                    gtsam::noiseModel::Diagonal::Sigmas(x0s));
    graph_.addPrior(V(0), gtsam::Vector3(vels[0]),
                    gtsam::noiseModel::Isotropic::Sigma(3, params_.iw_vel0_sigma));
    const double bas = params_.iw_bias0_acc_sigma, bgs = params_.iw_bias0_gyr_sigma;
    Eigen::Matrix<double, 6, 1> b0s; b0s << bas, bas, bas, bgs, bgs, bgs;
    graph_.addPrior(B(0), bias0, gtsam::noiseModel::Diagonal::Sigmas(b0s));

    // 3) CombinedImuFactor between consecutive keyframes.
    int imu_factors = 0;
    for (int k = 1; k < N; ++k) {
        int64_t ta = t0_ns + (int64_t)(frames[k - 1].t * 1e9);
        int64_t tb = t0_ns + (int64_t)(frames[k].t * 1e9);
        gtsam::PreintegratedCombinedMeasurements pim(imu_params_, bias0);
        auto lo = std::lower_bound(imu.begin(), imu.end(), ta,
                                   [](const ImuSample& s, int64_t v){ return s.t_ns < v; });
        auto hi = std::upper_bound(imu.begin(), imu.end(), tb,
                                   [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
        int n = 0;
        auto prev = lo;
        if (prev != hi) {
            for (auto it = prev + 1; it != hi; ++it) {
                double dt = (it->t_ns - prev->t_ns) * 1e-9;
                if (dt > 0) { pim.integrateMeasurement(it->a, it->w, dt); ++n; }
                prev = it;
            }
        }
        if (n < 1 || pim.deltaTij() <= 0.0) continue;
        graph_.emplace_shared<gtsam::CombinedImuFactor>(
            X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), pim);
        ++imu_factors;
    }

    // 4) Bootstrap landmarks as inverse-depth (warm map). Anchor each at the
    //    NEWEST observing keyframe so it survives longest into the online phase.
    std::unordered_map<int64_t, std::vector<std::pair<int, Eigen::Vector2d>>> obs_of;
    for (int k = 0; k < N; ++k) {
        const auto& f = frames[k];
        for (size_t j = 0; j < f.ids.size(); ++j) {
            int64_t id = f.ids[j];
            if (!points.count(id)) continue;
            obs_of[id].push_back({k, Eigen::Vector2d(fx * f.pts_norm(j, 0) + cx,
                                                     fy * f.pts_norm(j, 1) + cy)});
        }
    }

    gtsam::Vector3 floor_sig; floor_sig.setConstant(params_.lm_floor_prior_sigma);
    auto floor_noise = gtsam::noiseModel::Diagonal::Sigmas(floor_sig);
    int n_lm = 0;
    // World-frame seeding (point_backend=world): the bootstrap map is ALREADY in
    // world coordinates, so it drops straight in — no anchor pick, no inverse-depth
    // conversion. Depth sanity is still checked from the newest observing view.
    if (world_mode_) {
        gtsam::Vector3 wp_sig; wp_sig.setConstant(params_.lm_world_prior_sigma_m);
        auto wp_noise = gtsam::noiseModel::Diagonal::Sigmas(wp_sig);
        for (auto& kv : obs_of) {
            auto& views = kv.second;
            if (views.size() < 2) continue;
            int ai = 0;
            for (size_t i = 1; i < views.size(); ++i)
                if (views[i].first > views[ai].first) ai = (int)i;
            const int a_idx = views[ai].first;

            const int64_t fid = kv.first;
            gtsam::Pose3 wTc_a(gtsam::Rot3(T_wb[a_idx].block<3, 3>(0, 0)),
                               gtsam::Point3(T_wb[a_idx].block<3, 1>(0, 3)));
            wTc_a = wTc_a.compose(body_P_cam_);
            const gtsam::Point3 Pw(points.at(fid));
            const gtsam::Point3 Pc = wTc_a.transformTo(Pw);
            if (!(Pc.z() > 0.3 && Pc.z() < 40.0)) continue;

            const gtsam::Key lk = L(next_lm_key_++);
            lm_key_[fid] = gtsam::Symbol(lk).index();
            const double w_a   = win_coord(a_idx);
            lm_last_seen_[fid] = w_a;
            landmarks_in_graph_.insert(fid);
            initial_estimates_.insert(lk, Pw);
            ts[lk] = w_a;

            graph_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(lk, Pw, wp_noise);
            for (auto& v : views)
                graph_.emplace_shared<ProjFactor>(
                    gtsam::Point2(v.second(0), v.second(1)), cam_noise_,
                    X(v.first), lk, calib_, body_P_cam_);
            ++n_lm;
        }
    }
    // smart mode seeds no explicit landmarks here; the online smart factors accrue
    // structure once tracks reach 2 views (init window keeps IMU + gauge priors).
    if (!smart_mode_ && !world_mode_) for (auto& kv : obs_of) {
        auto& views = kv.second;
        if (views.size() < 2) continue;
        int ai = 0;
        for (size_t i = 1; i < views.size(); ++i)
            if (views[i].first > views[ai].first) ai = (int)i;
        int a_idx = views[ai].first;
        const Eigen::Vector2d& a_uv = views[ai].second;

        int64_t fid = kv.first;
        gtsam::Pose3 wTc_a(gtsam::Rot3(T_wb[a_idx].block<3, 3>(0, 0)),
                           gtsam::Point3(T_wb[a_idx].block<3, 1>(0, 3)));
        wTc_a = wTc_a.compose(body_P_cam_);
        gtsam::Point3 Pc = wTc_a.transformTo(gtsam::Point3(points.at(fid)));
        if (!(Pc.z() > 0.3 && Pc.z() < 40.0)) continue;

        gtsam::Vector3 l = cam_point_to_invdepth(Pc);
        gtsam::Key lk = L(next_lm_key_++);
        lm_key_[fid]       = gtsam::Symbol(lk).index();
        lm_anchor_[fid]    = a_idx;
        const double w_a   = win_coord(a_idx);   // anchor's window coord
        lm_last_seen_[fid] = w_a;
        landmarks_in_graph_.insert(fid);
        initial_estimates_.insert(lk, l);
        ts[lk] = w_a;

        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(lk, l, floor_noise);
        graph_.emplace_shared<InvDepthAnchorFactor>(
            X(a_idx), lk, gtsam::Point2(a_uv(0), a_uv(1)), body_P_cam_, calib_, cam_noise_);
        for (auto& v : views) {
            if (v.first == a_idx) continue;
            graph_.emplace_shared<InvDepthProjFactor>(
                X(a_idx), X(v.first), lk,
                gtsam::Point2(v.second(0), v.second(1)), body_P_cam_, calib_, cam_noise_);
        }
        ++n_lm;
    }

    // 5) First joint solve; online continues from the last init state.
    state_idx_   = N - 1;
    current_time_ = win_coord(N - 1);

    run_update(graph_, initial_estimates_, ts);
    latest_estimate_ = run_estimate();
    current_bias_ = latest_estimate_.at<gtsam::imuBias::ConstantBias>(B(state_idx_));
    pim_->resetIntegrationAndSetBias(current_bias_);

    graph_.resize(0);
    initial_estimates_.clear();
    initialized_ = true;

    // Parallax reference for the first online keyframe gate (pixel coords).
    last_kf_obs_.clear();
    {
        const auto& f = frames[N - 1];
        for (size_t j = 0; j < f.ids.size(); ++j)
            last_kf_obs_[f.ids[j]] = Eigen::Vector2d(fx * f.pts_norm(j, 0) + cx,
                                                     fy * f.pts_norm(j, 1) + cy);
    }

    // Drop any init state already outside the lag (long init windows).
    for (auto it = state_time_.begin(); it != state_time_.end(); ) {
        if (it->first != state_idx_ && (current_time_ - it->second) >= lag_)
            it = state_time_.erase(it);
        else ++it;
    }

    std::cout << "[VIO Backend] Initialized continuous window VI-BA: " << N
              << " states, " << imu_factors << " IMU factors, " << n_lm
              << " landmarks (lag=" << lag_ << " KF"
              << "); online from state "
              << state_idx_ << "\n";
}

bool VioBackend::get_pose_at(int idx, Eigen::Matrix4d& out) const {
    if (!latest_estimate_.exists(X(idx))) return false;
    gtsam::Pose3 p = latest_estimate_.at<gtsam::Pose3>(X(idx));
    out = Eigen::Matrix4d::Identity();
    out.block<3, 3>(0, 0) = p.rotation().matrix();
    out.block<3, 1>(0, 3) = p.translation();
    return true;
}

bool VioBackend::get_pose_covariance(int idx, Eigen::Matrix<double, 6, 6>& out) const {
    // The state must still be in the window (marginalCovariance queries the live
    // factorization). marginalCovariance throws IndeterminantLinearSystem on a
    // rank-deficient window; treat that as "unavailable".
    if (!latest_estimate_.exists(X(idx))) return false;
    try {
        if (isam_view_) {
            out = isam_view_->marginalCovariance(X(idx));
        } else if (kf_smoother_) {
            out = kf_smoother_->marginalCovariance(X(idx));
        } else {
            return false;
        }
    } catch (const std::exception& e) {
        static int n = 0;
        if (n++ < 3) std::cerr << "[cov] marginalCovariance(X(" << idx << ")) failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

bool VioBackend::get_relative_pose_covariance(int ia, int ib,
                                              Eigen::Matrix<double, 6, 6>& out) const {
    if (ia == ib) return false;
    if (!latest_estimate_.exists(X(ia)) || !latest_estimate_.exists(X(ib))) return false;
    const gtsam::KeyVector keys{X(ia), X(ib)};
    try {
        gtsam::Matrix J12;   // 12x12 joint, block order [ia, ib]
        if (kf_smoother_) {
            J12 = kf_smoother_->jointMarginalCovariance(keys);
        } else if (isam_view_) {
            gtsam::Marginals marg(isam_view_->getFactors(),
                                  isam_view_->getLinearizationPoint());
            gtsam::JointMarginal jm = marg.jointMarginalCovariance(keys);
            J12.resize(12, 12);
            J12.block<6,6>(0,0) = jm.at(X(ia), X(ia));  J12.block<6,6>(0,6) = jm.at(X(ia), X(ib));
            J12.block<6,6>(6,0) = jm.at(X(ib), X(ia));  J12.block<6,6>(6,6) = jm.at(X(ib), X(ib));
        } else {
            return false;
        }
        // d(Ta^-1 Tb)/d(Ta), d(...)/d(Tb) -> J = [H1 H2] (6x12); Sigma_rel = J S J^T.
        const gtsam::Pose3 Ta = latest_estimate_.at<gtsam::Pose3>(X(ia));
        const gtsam::Pose3 Tb = latest_estimate_.at<gtsam::Pose3>(X(ib));
        Eigen::Matrix<double, 6, 6> H1, H2;
        Ta.between(Tb, H1, H2);
        Eigen::Matrix<double, 6, 12> J;
        J.block<6,6>(0,0) = H1;
        J.block<6,6>(0,6) = H2;
        out = J * J12 * J.transpose();
        out = 0.5 * (out + out.transpose());   // re-symmetrize against round-off
    } catch (const std::exception& e) {
        static int n = 0;
        if (n++ < 3) std::cerr << "[cov] jointMarginalCovariance(X(" << ia << "),X(" << ib
                               << ")) failed: " << e.what() << "\n";
        return false;
    }
    return out.allFinite();
}

void VioBackend::add_imu(double dt, const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro) {
    if (!initialized_) return;
    pim_->integrateMeasurement(accel, gyro, dt);
}

void VioBackend::get_landmarks_world(
        std::unordered_map<int64_t, Eigen::Vector3d>& out) const {
    out.clear();
    // World mode: the landmark IS the world point — no anchor composition needed.
    if (world_mode_) {
        for (int64_t fid : landmarks_in_graph_) {
            auto ik = lm_key_.find(fid);
            if (ik == lm_key_.end()) continue;
            const gtsam::Key lk = L(ik->second);
            if (!latest_estimate_.exists(lk)) continue;
            const gtsam::Point3 Pw = latest_estimate_.at<gtsam::Point3>(lk);
            out[fid] = Eigen::Vector3d(Pw.x(), Pw.y(), Pw.z());
        }
        return;
    }
    for (int64_t fid : landmarks_in_graph_) {
        auto ik = lm_key_.find(fid);
        auto ia = lm_anchor_.find(fid);
        if (ik == lm_key_.end() || ia == lm_anchor_.end()) continue;
        gtsam::Key lk = L(ik->second);
        gtsam::Key xa = X(ia->second);
        if (!latest_estimate_.exists(lk) || !latest_estimate_.exists(xa)) continue;
        gtsam::Vector3 l   = latest_estimate_.at<gtsam::Vector3>(lk);
        gtsam::Pose3   wTc = latest_estimate_.at<gtsam::Pose3>(xa).compose(body_P_cam_);
        gtsam::Point3  Pw  = wTc.transformFrom(invdepth_to_cam_point(l));
        out[fid] = Eigen::Vector3d(Pw.x(), Pw.y(), Pw.z());
    }
}

VioBackend::Diag VioBackend::get_diag() const {
    Diag d;
    if (latest_estimate_.exists(V(state_idx_)))
        d.vel_norm = latest_estimate_.at<gtsam::Vector3>(V(state_idx_)).norm();
    d.ba_norm       = current_bias_.accelerometer().norm();
    d.bg_norm       = current_bias_.gyroscope().norm();
    d.n_pts_graph   = smart_mode_ ? (int)sf_old_.size() : (int)landmarks_in_graph_.size();
    return d;
}

gtsam::Pose3 VioBackend::get_pose_cached(int idx, const gtsam::Values& cache) const{
    if (idx == state_idx_) return initial_estimates_.at<gtsam::Pose3>(X(idx));
    return cache.at<gtsam::Pose3>(X(idx));
}

bool VioBackend::triangulate_dlt(int idx1, int idx2,
                                const Eigen::Vector2d& uv1, const Eigen::Vector2d& uv2,
                                const gtsam::Values& cache,
                                Eigen::Vector3d& X_w_out) 
{
    Eigen::Matrix4d pose1_wc = get_pose_cached(idx1, cache).compose(body_P_cam_).matrix();
    Eigen::Matrix4d pose2_wc = get_pose_cached(idx2, cache).compose(body_P_cam_).matrix();

    double baseline = (pose2_wc.block<3, 1>(0, 3) - pose1_wc.block<3, 1>(0, 3)).norm();
    if (baseline < params_.lm_min_baseline_m) return false;

    Eigen::Matrix4d pose1_cw = pose1_wc.inverse();
    Eigen::Matrix4d pose2_cw = pose2_wc.inverse();

    Eigen::Matrix<double, 3, 4> P1 = K_ * pose1_cw.block<3, 4>(0, 0);
    Eigen::Matrix<double, 3, 4> P2 = K_ * pose2_cw.block<3, 4>(0, 0);

    Eigen::Matrix4d A;
    A.row(0) = uv1(0) * P1.row(2) - P1.row(0);
    A.row(1) = uv1(1) * P1.row(2) - P1.row(1);
    A.row(2) = uv2(0) * P2.row(2) - P2.row(0);
    A.row(3) = uv2(1) * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d S = svd.singularValues();
    if (S(3) < 1e-8 || S(0) / std::max(S(3), 1e-12) > 1e6) return false;

    Eigen::Vector4d homo = svd.matrixV().col(3);
    if (std::abs(homo(3)) < 1e-9) return false;
    Eigen::Vector3d X_w = homo.head<3>() / homo(3);

    double z1 = (pose1_cw.block<3, 3>(0, 0) * X_w + pose1_cw.block<3, 1>(0, 3))(2);
    double z2 = (pose2_cw.block<3, 3>(0, 0) * X_w + pose2_cw.block<3, 1>(0, 3))(2);
    const double zmin = params_.lm_tri_depth_min, zmax = params_.lm_tri_depth_max;
    if (!(z1 > zmin && z1 < zmax && z2 > zmin && z2 < zmax)) return false;

    // landmark.min_parallax_deg: the angle the baseline subtends at the point, a scale-free
    // version of the metric baseline gate above.
    if (params_.lm_min_parallax_deg > 0.0) {
        const Eigen::Vector3d r1 = X_w - pose1_wc.block<3, 1>(0, 3);
        const Eigen::Vector3d r2 = X_w - pose2_wc.block<3, 1>(0, 3);
        const double c = r1.dot(r2) / std::max(r1.norm() * r2.norm(), 1e-12);
        const double ang_deg = std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
        if (ang_deg < params_.lm_min_parallax_deg) return false;
    }

    X_w_out = X_w;
    return true;
}

// Smart-point path (point_backend=smart). Builds/extends SmartProjectionPoseFactors
// from this keyframe's observations, then runs ONE incremental update that:
//   - deletes the stale slot of every extended factor (factorsToRemove),
//   - adds the new/cloned smart factors FIRST, then graph_ (IMU/ZUPT/prior),
//   - reads each new smart factor's slot back from newFactorsIndices (1-to-1),
//   - prunes factors that marginalized out (slot freed or reused).
// Backup-and-restore wraps the update so a degenerate window cannot kill the run
// (basic recovery; full prior-injection recovery is Phase 3).
bool VioBackend::smart_point_update(
        const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
        const gtsam::FixedLagSmoother::KeyTimestampMap& ts) {
    std::vector<int64_t> staged_ids;        // push order -> maps 1-to-1 to newFactorsIndices
    gtsam::NonlinearFactorGraph smart_new;  // staged smart factors, same order as staged_ids
    gtsam::FactorIndices delete_slots;
    const gtsam::NonlinearFactorGraph& fg = isam_view_->getFactors();

    for (const auto& kv : observations) {
        const int64_t fid = kv.first;
        const gtsam::Point2 px(kv.second(0), kv.second(1));
        auto it = sf_old_.find(fid);
        if (it == sf_old_.end()) {
            // Not yet a graph factor: buffer until we have >= 2 views (a 1-view
            // smart factor is uninformative), then create with the full track.
            auto& buf = sf_track_[fid];
            buf.emplace_back(state_idx_, kv.second);
            if (buf.size() < 2) continue;
            auto f = boost::make_shared<SmartFactor>(smart_noise_, calib_, body_P_cam_, smart_params_);
            for (const auto& o : buf) f->add(gtsam::Point2(o.second(0), o.second(1)), X(o.first));
            sf_old_[fid] = {f, -1};
            sf_track_.erase(fid);
            smart_new.push_back(f);
            staged_ids.push_back(fid);
        } else {
            // Existing factor: clone, append this observation, delete the old slot.
            auto old  = it->second.first;
            const int slot = it->second.second;
            auto f = boost::make_shared<SmartFactor>(*old);
            f->add(px, X(state_idx_));
            if (slot >= 0 && fg.exists(slot) && fg.at(slot).get() == old.get())
                delete_slots.push_back(slot);
            it->second.first = f;          // slot refreshed after readback
            smart_new.push_back(f);
            staged_ids.push_back(fid);
        }
    }

    // SMART FACTORS FIRST so the first staged_ids.size() newFactorsIndices are theirs.
    gtsam::NonlinearFactorGraph nf;
    nf.reserve(smart_new.size() + graph_.size());
    nf.push_back(smart_new.begin(), smart_new.end());
    nf.push_back(graph_.begin(), graph_.end());

    gtsam::IncrementalFixedLagSmoother backup(*isam_view_);   // for recovery
    try {
        smoother_->update(nf, initial_estimates_, ts, delete_slots);
    } catch (const std::exception& e) {
        std::cerr << "[smart] update failed (" << e.what()
                  << "); restoring and retrying without smart factors this KF.\n";
        *isam_view_ = backup;
        for (int64_t fid : staged_ids) sf_old_.erase(fid);   // drop uncommitted staging
        try { smoother_->update(graph_, initial_estimates_, ts); }
        catch (const std::exception& e2) {
            std::cerr << "[smart] fallback update also failed: " << e2.what() << "\n";
            return false;
        }
        return true;
    }

    // Slot readback (1-to-1 with the factors we pushed, in order).
    const gtsam::ISAM2Result& res = isam_view_->getISAM2Result();
    for (size_t i = 0; i < staged_ids.size(); ++i)
        if (i < res.newFactorsIndices.size())
            sf_old_[staged_ids[i]].second = (int)res.newFactorsIndices[i];

    // Prune factors that marginalized out (slot freed, or reused by another factor).
    const gtsam::NonlinearFactorGraph& fg2 = isam_view_->getFactors();
    for (auto it = sf_old_.begin(); it != sf_old_.end(); ) {
        const int slot = it->second.second;
        if (slot < 0 || slot >= (int)fg2.size() ||
            fg2.at(slot) == nullptr || fg2.at(slot).get() != it->second.first.get())
            it = sf_old_.erase(it);
        else ++it;
    }
    return true;
}

// World-frame point backend (point_backend=world) — the L4 control arm.
//
// Landmarks are world-frame Point3 variables observed through
// GenericProjectionFactor, the OKVIS / SVO-pro homogeneous-point design (see
// svo_ceres_backend/homogeneous_point_parameter_block.hpp). Because a landmark is
// not expressed in any keyframe's camera, marginalizing a pose cannot orphan it:
// the Schur complement absorbs that pose's projection factors into the marginal and
// the landmark variable survives, carrying its accumulated information forward with
// its correlations intact. That deletes the re-anchoring problem outright, instead
// of paying for it the way VINS-Mono's removeBackShiftDepth does (which re-emits
// measurements already inside the marginal — measured here as a 4.7x NEES cost).
//
// Consequence to watch: landmarks now persist for the whole life of a track, so the
// marginal densifies over live landmarks rather than staying pose-sized. Cost is
// per-keyframe solve time, which is the thing to measure against inverse depth.
void VioBackend::world_point_update(
        const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
        const gtsam::Values& current,
        gtsam::FixedLagSmoother::KeyTimestampMap& ts,
        double w,
        std::vector<int64_t>& observed_points) {

    const gtsam::Pose3 wTc_cur = get_pose_cached(state_idx_, current).compose(body_P_cam_);

    auto pose_live = [&](int idx) {
        auto it = state_time_.find(idx);
        return it != state_time_.end()
            && (w - it->second) < lag_
            && current.exists(X(idx));
    };

    gtsam::Vector3 wp_sig; wp_sig.setConstant(params_.lm_world_prior_sigma_m);
    auto wp_noise = gtsam::noiseModel::Diagonal::Sigmas(wp_sig);

    for (const auto& kv : observations) {
        const int64_t fid = kv.first;
        const Eigen::Vector2d& uv = kv.second;

        gtsam::Key lkey = 0;
        bool usable = false;
        if (landmarks_in_graph_.count(fid)) {
            auto ik = lm_key_.find(fid);
            auto is = lm_last_seen_.find(fid);
            if (ik != lm_key_.end()) lkey = L(ik->second);
            // The ONLY death condition is last-seen staleness (the smoother evicts
            // the key on the same rule). There is no anchor to age out.
            usable = ik != lm_key_.end()
                  && latest_estimate_.exists(lkey)
                  && is != lm_last_seen_.end() && (w - is->second) < lag_;
            if (!usable) {
                landmarks_in_graph_.erase(fid);
                lm_last_seen_.erase(fid);
                lm_key_.erase(fid);
                feature_obs_.erase(fid);
            }
        }

        if (usable) {
            // Per-observation dynamic rejection (Kimera outlierRejection analog),
            // straight off the world point — no anchor lookup needed.
            if (params_.pt_obs_reject_px > 0.0) {
                const gtsam::Point3 Pc =
                    wTc_cur.transformTo(latest_estimate_.at<gtsam::Point3>(lkey));
                bool reject = Pc.z() <= 0.0;   // behind camera
                if (!reject) {
                    const gtsam::Point2 pred = calib_->uncalibrate(
                        gtsam::Point2(Pc.x() / Pc.z(), Pc.y() / Pc.z()));
                    reject = (pred - gtsam::Point2(uv(0), uv(1))).norm()
                             > params_.pt_obs_reject_px;
                }
                if (reject) { ++pt_obs_rejected_; continue; }
            }
            graph_.emplace_shared<ProjFactor>(gtsam::Point2(uv(0), uv(1)), cam_noise_,
                                              X(state_idx_), lkey, calib_, body_P_cam_);
            ts[lkey] = w;
            lm_last_seen_[fid] = w;
            observed_points.push_back(fid);
            continue;
        }

        // --- delayed initialization (same gates as the inverse-depth path) ---
        feature_obs_[fid].emplace_back(state_idx_, uv);
        auto& vec = feature_obs_[fid];

        if (vec.size() >= 2) {
            std::vector<std::pair<int, Eigen::Vector2d>> live;
            live.reserve(vec.size());
            for (const auto& o : vec)
                if (pose_live(o.first)) live.push_back(o);

            if (live.size() >= 2) {
                Eigen::Vector3d X_w;
                // baseline gate inside triangulate_dlt self-delays under low parallax
                if (triangulate_dlt(live.front().first, live.back().first,
                                    live.front().second, live.back().second,
                                    current, X_w)) {
                    bool all_ok = true;
                    for (const auto& o : live) {
                        const double z = get_pose_cached(o.first, current)
                                             .compose(body_P_cam_)
                                             .transformTo(gtsam::Point3(X_w)).z();
                        if (!(z > params_.lm_depth_min && z < params_.lm_depth_max)) {
                            all_ok = false; break;
                        }
                    }
                    if (all_ok) {
                        const gtsam::Point3 Pw(X_w);
                        const gtsam::Key lk = L(next_lm_key_++);
                        lm_key_[fid]       = gtsam::Symbol(lk).index();
                        landmarks_in_graph_.insert(fid);
                        lm_last_seen_[fid] = w;
                        initial_estimates_.insert(lk, Pw);
                        ts[lk]             = w;
                        observed_points.push_back(fid);

                        // Weak world-frame prior: the XYZ analog of the inverse-depth
                        // floor prior. World points have no anchor to regularize them,
                        // so this is what stands between a low-parallax track and the
                        // IndeterminantLinearSystem that killed the smart-factor path.
                        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
                            lk, Pw, wp_noise);
                        for (const auto& o : live)
                            graph_.emplace_shared<ProjFactor>(
                                gtsam::Point2(o.second(0), o.second(1)), cam_noise_,
                                X(o.first), lk, calib_, body_P_cam_);
                        feature_obs_.erase(fid);
                    }
                }
            }
        } else if (vec.size() > 15) {
            feature_obs_.erase(fid);
        }
    }
}

bool VioBackend::add_frame(const std::unordered_map<int64_t, Eigen::Vector2d>& observations,
                           double t_s, bool apply_zupt) {
    if (!initialized_) return false;

    // Keyframe gate. Non-keyframes add no graph state; add_imu() (called every
    // frame from the driver) keeps integrating pim_ across the gap, so the next
    // keyframe's CombinedImuFactor spans all skipped frames.
    if (!decide_keyframe(observations)) return false;

    frames_since_kf_ = 0;
    state_idx_++;
    int prev_idx = state_idx_ - 1;

    // Windowing coordinate for this keyframe: the keyframe index (Phase 2b, so the
    // window spans N_kf keyframes). Every window
    // stamp below (smoother KeyTimestampMap, state_time_, lm_last_seen_,
    // current_time_) and every in-window test uses `w` so the back-end's
    // re-anchoring stays locked to the smoother's eviction.
    const double w = win_coord(state_idx_);

    // Timestamps for every key touched this step. Keys whose timestamp falls
    // out of [w - lag_, w] get marginalized inside smoother_.update().
    gtsam::FixedLagSmoother::KeyTimestampMap ts;

    // ---------------------------------------------------------------
    // 1. IMU + bias factors (connect prev keyframe, still in-window)
    // ---------------------------------------------------------------
    graph_.emplace_shared<gtsam::CombinedImuFactor>(
        X(prev_idx), V(prev_idx),
        X(state_idx_), V(state_idx_),
        B(prev_idx), B(state_idx_), *pim_);

    // ---------------------------------------------------------------
    // 2. Snapshot the windowed estimate ONCE (prediction + triangulation).
    // ---------------------------------------------------------------
    const gtsam::Values current = latest_estimate_;

    // ---------------------------------------------------------------
    // 3. State prediction (uses cached `current`)
    // ---------------------------------------------------------------
    gtsam::NavState nav_prev(current.at<gtsam::Pose3>(X(prev_idx)),
                             current.at<gtsam::Vector3>(V(prev_idx)));
    gtsam::NavState nav_pred = pim_->predict(nav_prev, current_bias_);

    initial_estimates_.insert(X(state_idx_), nav_pred.pose());
    initial_estimates_.insert(V(state_idx_), nav_pred.v());
    initial_estimates_.insert(B(state_idx_), current_bias_);
    ts[X(state_idx_)] = w;
    ts[V(state_idx_)] = w;
    ts[B(state_idx_)] = w;

    // ZUPT: while stationary, monocular has no parallax so vision cannot pin
    // position; the IMU then integrates residual accel into a growing drift.
    // Pin velocity to zero and lock relative translation to the previous
    // stationary keyframe to stop it.
    if (apply_zupt) {
        const gtsam::Vector3 zero = gtsam::Vector3::Zero();
        initial_estimates_.update(V(state_idx_), zero);
        graph_.addPrior(V(state_idx_), zero,
                        gtsam::noiseModel::Isotropic::Sigma(3, params_.zupt_vel_sigma));
        if (prev_kf_stationary_) {
            Eigen::Matrix<double, 6, 1> btw_sig;
            btw_sig << 0.1, 0.1, 0.1, 0.001, 0.001, 0.001;   // rot loose, trans tight (1 mm)
            graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                X(prev_idx), X(state_idx_), gtsam::Pose3(),
                gtsam::noiseModel::Diagonal::Sigmas(btw_sig));
        }
    }
    prev_kf_stationary_ = apply_zupt;

    // A buffered observation can only become a factor if its pose is still in
    // the window AND present in `current` (else its key is/about to be gone).
    auto pose_live = [&](int idx) {
        auto it = state_time_.find(idx);
        return it != state_time_.end()
            && (w - it->second) < lag_
            && current.exists(X(idx));
    };

    // ---------------------------------------------------------------
    // 4. Vision constraints (inverse-depth, window-aware delayed init)
    //    Landmark L(fid) = (theta,phi,rho) in its ANCHOR keyframe camera.
    //    A landmark lives only while its anchor pose is in the window; when the
    //    anchor ages out it is dropped and (if still tracked) re-anchored by
    //    re-initialization at a current keyframe.
    // ---------------------------------------------------------------
    // point_backend=smart handles points via smart_point_update() below; skip the
    // explicit inverse-depth construction entirely in that mode.
    // In-graph point landmarks touched this keyframe (for the post-update depth cull).
    std::vector<int64_t> observed_points;
    // point_backend=world runs its own (much simpler) landmark path: no anchor, so
    // no re-anchoring, no depth transfer, no abandon-on-anchor-death.
    if (world_mode_) world_point_update(observations, current, ts, w, observed_points);
    // Live re-anchor chains (landmark.reanchor_live_max). A chain is live while its track is
    // fresh. A track that simply ends keeps its entry, so stale entries are pruned here, which
    // changes nothing else: a stale track can only be abandoned.
    int live_chains = 0;
    if (reanchor_exact_) {
        for (auto it = lm_chain_reanchors_.begin(); it != lm_chain_reanchors_.end(); ) {
            const auto is = lm_last_seen_.find(it->first);
            if (is == lm_last_seen_.end() || (w - is->second) >= lag_) it = lm_chain_reanchors_.erase(it);
            else { ++live_chains; ++it; }
        }
        lm_live_sum_ += live_chains;
        ++lm_live_n_;
        lm_live_peak_ = std::max(lm_live_peak_, live_chains);
    }
    if (!smart_mode_ && !world_mode_) for (const auto& kv : observations) {
        int64_t fid = kv.first;
        const Eigen::Vector2d& uv = kv.second;

        bool known = landmarks_in_graph_.count(fid) > 0;
        bool usable = false;
        int anchor = -1;
        gtsam::Key lkey = 0;
        if (known) {
            auto ik = lm_key_.find(fid);
            auto ia = lm_anchor_.find(fid);
            auto is = lm_last_seen_.find(fid);
            anchor = (ia != lm_anchor_.end()) ? ia->second : -1;
            if (ik != lm_key_.end()) lkey = L(ik->second);
            usable = ik != lm_key_.end()
                  && latest_estimate_.exists(lkey)
                  && anchor >= 0
                  && is != lm_last_seen_.end() && (w - is->second) < lag_
                  && pose_live(anchor);                 // anchor must survive this update
            if (!usable) {
                // EXACT RE-ANCHORING (landmark.reanchor: exact). The anchor is leaving the
                // window while the track is still observed: give the landmark a new
                // variable anchored at THIS keyframe and tie it to the old one by a change of
                // variables (InvDepthReanchorFactor). The old variable is never observed
                // again and marginalizes by lag through the tie, so all of its information,
                // including the part already in the marginal, reaches the new one, and no
                // pixel is used twice. Only for an anchor exit with a fresh last-seen: a
                // landmark that went stale through repeated rejection is abandoned as
                // before. The outgoing anchor is still in `current` (the back-end stops
                // using a pose one keyframe before the smoother evicts it).
                // landmark.reanchor_max: a chain that already carries the cap of re-anchors
                // falls back to abandon, and the track starts a fresh chain when re-triangulated.
                // landmark.reanchor_live_max: when the live chains fill the cap, an exit that
                // would start a chain falls back to abandon. Running chains continue.
                const auto ich = lm_chain_reanchors_.find(fid);
                const int chain_n = (ich != lm_chain_reanchors_.end()) ? ich->second : 0;
                const bool capped = params_.lm_reanchor_max > 0 && chain_n >= params_.lm_reanchor_max;
                const bool full = params_.lm_reanchor_live_max > 0 && chain_n == 0
                                  && live_chains >= params_.lm_reanchor_live_max;
                const bool exit_fresh = reanchor_exact_ && anchor >= 0 && lkey != 0
                    && latest_estimate_.exists(lkey) && current.exists(X(anchor))
                    && is != lm_last_seen_.end() && (w - is->second) < lag_;
                if (exit_fresh && capped) ++lm_reanchor_capped_;   // falls through to abandon
                if (exit_fresh && full)   ++lm_reanchor_full_;     // falls through to abandon
                if (exit_fresh && !capped && !full) {
                    const gtsam::Vector3 l_old = latest_estimate_.at<gtsam::Vector3>(lkey);
                    const gtsam::Pose3 wTc_a =
                        current.at<gtsam::Pose3>(X(anchor)).compose(body_P_cam_);
                    const gtsam::Pose3 wTc_b =
                        get_pose_cached(state_idx_, current).compose(body_P_cam_);
                    bool in_front = false;
                    const gtsam::Vector3 l_new = (l_old(2) > 0.0)
                        ? reanchor_invdepth(l_old, wTc_a, wTc_b, &in_front)
                        : gtsam::Vector3(gtsam::Vector3::Zero());
                    bool ok = l_old(2) > 0.0 && in_front && l_new.allFinite() && l_new(2) > 0.0
                              && 1.0 / l_new(2) > params_.lm_depth_min
                              && 1.0 / l_new(2) < params_.lm_depth_max;
                    // Same gate as a re-observation: this keyframe's pixel against the
                    // landmark's prediction, which in the new anchor camera is its bearing.
                    if (ok && params_.pt_obs_reject_px > 0.0) {
                        const gtsam::Point3 Pc = invdepth_to_cam_point(l_new);
                        const gtsam::Point2 pred = calib_->uncalibrate(
                            gtsam::Point2(Pc.x() / Pc.z(), Pc.y() / Pc.z()));
                        ok = (pred - gtsam::Point2(uv(0), uv(1))).norm()
                             <= params_.pt_obs_reject_px;
                    }
                    if (ok) {
                        const gtsam::Key lk = L(next_lm_key_++);   // fresh key, new anchor
                        lm_key_[fid]       = gtsam::Symbol(lk).index();
                        lm_anchor_[fid]    = state_idx_;
                        lm_last_seen_[fid] = w;
                        initial_estimates_.insert(lk, l_new);
                        ts[lk] = w;
                        observed_points.push_back(fid);
                        graph_.emplace_shared<InvDepthReanchorFactor>(
                            lkey, X(anchor), X(state_idx_), lk, body_P_cam_, reanchor_noise_);
                        // landmark.reanchor_retire: the old variable leaves with its anchor.
                        if (params_.lm_reanchor_retire) ts[lkey] = win_coord(anchor);
                        // the new anchor's own measurement: a new pixel, used once
                        graph_.emplace_shared<InvDepthAnchorFactor>(
                            X(state_idx_), lk, gtsam::Point2(uv(0), uv(1)),
                            body_P_cam_, calib_, cam_noise_for(fid));
                        ++lm_reanchored_;
                        if (chain_n == 0) ++live_chains;
                        lm_chain_reanchors_[fid] = chain_n + 1;
                        continue;
                    }
                    ++lm_reanchor_declined_;   // falls through to abandon
                }

                // DEPTH TRANSFER (VINS-Mono FeatureManager::removeBackShiftDepth
                // analog). The anchor is leaving the window, but the landmark's depth
                // has been refined over the whole window — re-express it in the NEW
                // anchor camera instead of throwing it away and waiting for a fresh
                // 2-view triangulation (which also leaves the track contributing
                // nothing in the meantime). The outgoing anchor is still present in
                // `current`: the back-end stops using a pose one keyframe BEFORE the
                // smoother evicts it, so its pose is available for the transfer.
                bool transferred = false;
                if (params_.pt_depth_transfer && anchor >= 0 && lkey != 0
                    && latest_estimate_.exists(lkey) && current.exists(X(anchor))) {
                    // Surviving views of this landmark. The retained history holds
                    // past keyframes; this keyframe's own observation is appended
                    // explicitly because X(state_idx_) is not in `current` yet and
                    // so pose_live() rejects it.
                    std::vector<std::pair<int, Eigen::Vector2d>> live;
                    auto ih = feature_obs_.find(fid);
                    if (ih != feature_obs_.end())
                        for (const auto& o : ih->second)
                            if (pose_live(o.first)) live.push_back(o);
                    live.emplace_back(state_idx_, uv);

                    // New host: oldest surviving view (VINS shifts frame 1 -> 0),
                    // i.e. the largest remaining baseline. `live` always holds this
                    // keyframe, so front()/back() are safe here.
                    const auto& a_obs =
                        params_.pt_depth_transfer_oldest ? live.front() : live.back();
                    const int a_new = a_obs.first;

                    // Two ways to decline the transfer:
                    //  <2 views  — VINS erases such a feature (feature_per_frame.size()
                    //    < 2). One view leaves rho UNOBSERVED, since
                    //    InvDepthAnchorFactor constrains only (theta,phi); the depth
                    //    would hang on the weak floor prior alone.
                    //  a_new == anchor — pure churn: the old key marginalizes and the
                    //    new one re-asserts the same estimate on the same pose, with no
                    //    geometry gained. Reachable only when !usable came from
                    //    last-seen staleness rather than the anchor dying.
                    if (live.size() >= 2 && a_new != anchor) {
                        const gtsam::Vector3 l_old = latest_estimate_.at<gtsam::Vector3>(lkey);
                        const gtsam::Pose3 wTc_old =
                            current.at<gtsam::Pose3>(X(anchor)).compose(body_P_cam_);
                        const gtsam::Pose3 wTc_new =
                            get_pose_cached(a_new, current).compose(body_P_cam_);
                        const gtsam::Point3 Pw =
                            wTc_old.transformFrom(invdepth_to_cam_point(l_old));
                        const gtsam::Point3 Pc_new = wTc_new.transformTo(Pw);
                        if (Pc_new.z() > params_.lm_depth_min && Pc_new.z() < params_.lm_depth_max) {
                            const gtsam::Vector3 l_new = cam_point_to_invdepth(Pc_new);
                            const gtsam::Key lk = L(next_lm_key_++);   // fresh key, new anchor frame
                            lm_key_[fid]       = gtsam::Symbol(lk).index();
                            lm_anchor_[fid]    = a_new;
                            lm_last_seen_[fid] = w;
                            initial_estimates_.insert(lk, l_new);
                            ts[lk] = w;
                            observed_points.push_back(fid);
                            feature_obs_[fid] = live;   // carry history to the next re-host

                            gtsam::Vector3 floor_sig;
                            floor_sig.setConstant(params_.lm_floor_prior_sigma);
                            graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
                                lk, l_new, gtsam::noiseModel::Diagonal::Sigmas(floor_sig));
                            // new anchor's own measurement constrains (theta,phi)
                            graph_.emplace_shared<InvDepthAnchorFactor>(
                                X(a_new), lk, gtsam::Point2(a_obs.second(0), a_obs.second(1)),
                                body_P_cam_, calib_, cam_noise_);
                            // every OTHER surviving view constrains rho — VINS
                            // re-emits all remaining residuals against the new host.
                            for (const auto& o : live) {
                                if (o.first == a_new) continue;
                                graph_.emplace_shared<InvDepthProjFactor>(
                                    X(a_new), X(o.first), lk,
                                    gtsam::Point2(o.second(0), o.second(1)),
                                    body_P_cam_, calib_, cam_noise_);
                            }
                            ++pt_depth_transferred_;
                            transferred = true;
                        }
                    }
                }
                if (transferred) continue;   // re-anchored in place; keep the track alive

                // anchor (or landmark) aged out -> ABANDON the old key (it will
                // marginalize on its own) and re-anchor via re-initialization
                // with a FRESH key. Reusing L(fid) would collide with the still-
                // live variable in the smoother.
                landmarks_in_graph_.erase(fid);
                lm_anchor_.erase(fid);
                lm_last_seen_.erase(fid);
                lm_key_.erase(fid);
                feature_obs_.erase(fid);
                lm_chain_reanchors_.erase(fid);
            }
        }

        if (usable) {
            // Per-observation dynamic rejection (Kimera outlierRejection analog):
            // predict the landmark's pixel from the current estimate; a grossly
            // inconsistent (or behind-camera) measurement is DROPPED — no factor,
            // no last-seen refresh, so a persistently-outlier landmark decays out
            // by lag instead of feeding bad constraints.
            if (params_.pt_obs_reject_px > 0.0) {
                const gtsam::Vector3 l = latest_estimate_.at<gtsam::Vector3>(lkey);
                const gtsam::Pose3 wTc_a =
                    get_pose_cached(anchor, current).compose(body_P_cam_);
                const gtsam::Point3 Pw = wTc_a.transformFrom(invdepth_to_cam_point(l));
                const gtsam::Pose3 wTc_o =
                    get_pose_cached(state_idx_, current).compose(body_P_cam_);
                const gtsam::Point3 Pc = wTc_o.transformTo(Pw);
                bool reject = Pc.z() <= 0.0;   // behind camera
                if (!reject) {
                    const gtsam::Point2 pred = calib_->uncalibrate(
                        gtsam::Point2(Pc.x() / Pc.z(), Pc.y() / Pc.z()));
                    reject = (pred - gtsam::Point2(uv(0), uv(1))).norm()
                             > params_.pt_obs_reject_px;
                }
                if (reject) { ++pt_obs_rejected_; continue; }
            }
            // cross-view: anchor + current keyframe observe the same inv-depth lm
            gtsam::SharedNoiseModel nm = cam_noise_for(fid);
            if (params_.pt_hetero_cov && params_.pt_hetero_metric == "parallax") {
                const gtsam::Vector3 l  = latest_estimate_.at<gtsam::Vector3>(lkey);
                const gtsam::Pose3 wTc_a = get_pose_cached(anchor, current).compose(body_P_cam_);
                const gtsam::Point3 Pw  = wTc_a.transformFrom(invdepth_to_cam_point(l));
                const gtsam::Point3 c1  = get_pose_cached(state_idx_, current)
                                              .compose(body_P_cam_).translation();
                nm = cam_noise_parallax(parallax_angle_deg(Pw, wTc_a.translation(), c1));
            }
            graph_.emplace_shared<InvDepthProjFactor>(
                X(anchor), X(state_idx_), lkey,
                gtsam::Point2(uv(0), uv(1)), body_P_cam_, calib_, nm);
            ts[lkey] = w;
            lm_last_seen_[fid] = w;
            observed_points.push_back(fid);
            // Retain the observation history of an IN-GRAPH landmark (pruned to
            // live poses, so it stays window-bounded). VINS's FeatureManager keeps
            // every view of every feature; we normally discard them at delayed init,
            // which is what left the old transfer with a single, rho-blind view.
            // Only while the transfer is enabled -> the flag-off path is unchanged.
            if (params_.pt_depth_transfer) {
                auto& hist = feature_obs_[fid];
                hist.erase(std::remove_if(hist.begin(), hist.end(),
                                          [&](const std::pair<int, Eigen::Vector2d>& o) {
                                              return !pose_live(o.first);
                                          }),
                           hist.end());
                hist.emplace_back(state_idx_, uv);
            }
            continue;
        }

        // --- delayed initialization (also the re-anchor path) ---
        feature_obs_[fid].emplace_back(state_idx_, uv);
        auto& vec = feature_obs_[fid];

        if (vec.size() >= 2) {
            std::vector<std::pair<int, Eigen::Vector2d>> live;
            live.reserve(vec.size());
            for (const auto& o : vec)
                if (pose_live(o.first)) live.push_back(o);

            if (live.size() >= 2) {
                Eigen::Vector3d X_w;
                // baseline gate inside triangulate_dlt self-delays under low parallax
                if (triangulate_dlt(live.front().first, live.back().first,
                                    live.front().second, live.back().second,
                                    current, X_w)) {
                    // depth validity across all live views
                    bool all_ok = true;
                    for (const auto& obs : live) {
                        double z = get_pose_cached(obs.first, current)
                                       .compose(body_P_cam_)
                                       .transformTo(gtsam::Point3(X_w)).z();
                        if (!(z > params_.lm_depth_min && z < params_.lm_depth_max)) { all_ok = false; break; }
                    }

                    // anchor = NEWEST in-window view -> longest remaining lifetime
                    int a_idx = live.back().first;
                    const Eigen::Vector2d& a_uv = live.back().second;
                    gtsam::Pose3 wTc_a = get_pose_cached(a_idx, current).compose(body_P_cam_);
                    gtsam::Point3 Pc = wTc_a.transformTo(gtsam::Point3(X_w));  // world -> anchor cam

                    if (all_ok && Pc.z() > 0.1) {
                        gtsam::Vector3 l = cam_point_to_invdepth(Pc);
                        gtsam::Key lk = L(next_lm_key_++);   // FRESH key (no re-insert collision)
                        lm_key_[fid]       = gtsam::Symbol(lk).index();
                        initial_estimates_.insert(lk, l);
                        landmarks_in_graph_.insert(fid);
                        lm_anchor_[fid]    = a_idx;
                        lm_last_seen_[fid] = w;
                        ts[lk]             = w;
                        observed_points.push_back(fid);   // fresh lm joins the post-update cull

                        // Weak per-component FLOOR prior in inverse-depth coords
                        // (theta,phi in rad, rho in 1/m) — only a numerical
                        // regularizer; the anchor + cross-view factors carry the
                        // real information.
                        gtsam::Vector3 floor_sig; floor_sig.setConstant(params_.lm_floor_prior_sigma);
                        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
                            lk, l, gtsam::noiseModel::Diagonal::Sigmas(floor_sig));

                        // anchor's own measurement constrains (theta,phi)
                        graph_.emplace_shared<InvDepthAnchorFactor>(
                            X(a_idx), lk, gtsam::Point2(a_uv(0), a_uv(1)),
                            body_P_cam_, calib_, cam_noise_for(fid));

                        // cross-view factors (these constrain rho)
                        for (const auto& obs : live) {
                            if (obs.first == a_idx) continue;
                            gtsam::SharedNoiseModel nm = cam_noise_for(fid);
                            if (params_.pt_hetero_cov && params_.pt_hetero_metric == "parallax") {
                                const gtsam::Point3 c1 = get_pose_cached(obs.first, current)
                                                             .compose(body_P_cam_).translation();
                                nm = cam_noise_parallax(parallax_angle_deg(
                                        gtsam::Point3(X_w), wTc_a.translation(), c1));
                            }
                            graph_.emplace_shared<InvDepthProjFactor>(
                                X(a_idx), X(obs.first), lk,
                                gtsam::Point2(obs.second(0), obs.second(1)),
                                body_P_cam_, calib_, nm);
                        }
                        // landmark.init_current_obs: this keyframe's own view. It is not in
                        // `live` (X(state_idx_) is not in `current` yet) and is erased with
                        // the history below, so without this it is never used. The pose is
                        // inserted in this same update, so the factor is legal. Same gate
                        // as a re-observation.
                        if (params_.lm_init_current_obs && a_idx != state_idx_) {
                            const gtsam::Pose3 wTc_o =
                                get_pose_cached(state_idx_, current).compose(body_P_cam_);
                            const gtsam::Point3 Pc_o = wTc_o.transformTo(gtsam::Point3(X_w));
                            bool reject = Pc_o.z() <= 0.0;
                            if (!reject && params_.pt_obs_reject_px > 0.0) {
                                const gtsam::Point2 pred = calib_->uncalibrate(
                                    gtsam::Point2(Pc_o.x() / Pc_o.z(), Pc_o.y() / Pc_o.z()));
                                reject = (pred - gtsam::Point2(uv(0), uv(1))).norm()
                                         > params_.pt_obs_reject_px;
                            }
                            if (reject) {
                                ++lm_init_obs_rejected_;
                            } else {
                                gtsam::SharedNoiseModel nm = cam_noise_for(fid);
                                if (params_.pt_hetero_cov && params_.pt_hetero_metric == "parallax")
                                    nm = cam_noise_parallax(parallax_angle_deg(
                                            gtsam::Point3(X_w), wTc_a.translation(),
                                            wTc_o.translation()));
                                graph_.emplace_shared<InvDepthProjFactor>(
                                    X(a_idx), X(state_idx_), lk,
                                    gtsam::Point2(uv(0), uv(1)), body_P_cam_, calib_, nm);
                                ++lm_init_obs_added_;
                            }
                        }
                        // Keep the views that built this landmark when the transfer
                        // is on, so its first re-host has a full history.
                        if (params_.pt_depth_transfer) feature_obs_[fid] = live;
                        else                           feature_obs_.erase(fid);
                    }
                }
            }
        } else if (vec.size() > 15) {
            feature_obs_.erase(fid);
        }
    }

    // ---------------------------------------------------------------
    // 4b. NIS metric A -- reprojection PRE-FIT innovation (GT-free consistency).
    //     Evaluate each existing-landmark observation of THIS keyframe at the PRIOR:
    //     obs pose = IMU prediction (nav_pred), anchor + landmark at their window
    //     estimates. graph_ here holds ONLY this KF's new factors (cleared post-update),
    //     so an InvDepthProjFactor with obs pose == X(state_idx_) and anchor != X(state_idx_)
    //     is exactly a re-observation of a landmark that already had a prior.
    //     S = H_obs P- H_obs^T + sigma_px^2 I -- POSE-ONLY innovation covariance (anchor and
    //     inverse-depth uncertainty omitted -> S understated -> NIS_A biased high). P- is the
    //     previous pose marginal propagated through the IMU relative motion (EKF predict).
    //     invdepth mode only (smart/world excluded); gated on dump_covariance.
    if (params_.dump_covariance && !smart_mode_ && !world_mode_) {
        Eigen::Matrix<double, 6, 6> P_prev;
        if (get_pose_covariance(prev_idx, P_prev)) {
            const gtsam::Pose3 T_prev = current.at<gtsam::Pose3>(X(prev_idx));
            const gtsam::Pose3 T_pred = nav_pred.pose();
            const gtsam::Pose3 dT = T_prev.inverse().compose(T_pred);
            Eigen::Matrix<double, 6, 6> Hc1, Hc2;
            T_prev.compose(dT, Hc1, Hc2);                          // T_pred = T_prev o dT
            const Eigen::Matrix<double, 6, 6> Q_dT =
                pim_->preintMeasCov().block<6, 6>(0, 0);           // [rot,trans] rel-pose cov
            const Eigen::Matrix<double, 6, 6> P_minus =
                Hc1 * P_prev * Hc1.transpose() + Hc2 * Q_dT * Hc2.transpose();
            const double R = params_.pt_sigma_px * params_.pt_sigma_px;

            gtsam::Values prior_vals = current;
            prior_vals.insert(X(state_idx_), T_pred);

            double nis_sum = 0.0; int dof = 0;
            for (const auto& f : graph_) {
                auto* pf = dynamic_cast<const InvDepthProjFactor*>(f.get());
                if (!pf) continue;
                const gtsam::Key ka = pf->keys()[0], ko = pf->keys()[1], kl = pf->keys()[2];
                if (ko != X(state_idx_) || ka == X(state_idx_)) continue;  // existing landmarks only
                if (!prior_vals.exists(ka) || !prior_vals.exists(kl)) continue;
                gtsam::Matrix H1, H2, H3;
                const gtsam::Vector r = pf->evaluateError(
                    prior_vals.at<gtsam::Pose3>(ka), prior_vals.at<gtsam::Pose3>(ko),
                    prior_vals.at<gtsam::Vector3>(kl), H1, H2, H3);
                const Eigen::Matrix2d S =
                    H2 * P_minus * H2.transpose() + R * Eigen::Matrix2d::Identity();
                nis_sum += r.dot(S.inverse() * r);
                dof += 2;
            }
            if (dof > 0) {
                nis_a_ndof_.push_back(nis_sum / dof);
                nis_a_gate_.push_back(nis_sum < chi2_p95(dof) ? 1.0 : 0.0);
            }
        }
    }

    // ---------------------------------------------------------------
    // 5. Fixed-lag update (LM/ISAM2 over the window; marginalizes old keys)
    //    smart mode: smart_point_update() prepends the smart factors and runs the
    //    delete-old/add-new update with slot bookkeeping; else the plain update of
    //    graph_ (IMU/ZUPT/invdepth factors).
    // ---------------------------------------------------------------
    if (smart_mode_) {
        smart_point_update(observations, ts);
    } else {
        run_update(graph_, initial_estimates_, ts);
    }

    // ---------------------------------------------------------------
    // 6. Refresh bias from the new estimate, reset preintegrator.
    // ---------------------------------------------------------------
    const gtsam::Values post_update = run_estimate();
    const gtsam::imuBias::ConstantBias new_bias =
        post_update.at<gtsam::imuBias::ConstantBias>(B(state_idx_));

    // --- IMU preintegration-length diagnostic (read-only; pim_ still holds [prev,cur]) ---
    // nav_pred (computed above) is the prediction at biasHat = current_bias_; re-predict
    // from the SAME nav_prev at the optimized bias and measure the endpoint shift. That
    // shift IS the first-order bias correction the CombinedImuFactor applies over this
    // interval -- the only term whose accuracy degrades as the interval lengthens.
    {
        const gtsam::NavState pred_opt = pim_->predict(nav_prev, new_bias);
        imu_dt_.push_back(pim_->deltaTij());
        imu_bcorr_pos_.push_back((pred_opt.position() - nav_pred.position()).norm());
        imu_bcorr_rot_.push_back(
            gtsam::Rot3::Logmap(nav_pred.attitude().between(pred_opt.attitude())).norm()
            * 180.0 / M_PI);
        const Eigen::Matrix<double, 15, 15> C = pim_->preintMeasCov();  // [theta,pos,vel,bias]
        imu_cov_rot_.push_back(std::sqrt(C.block<3, 3>(0, 0).trace()) * 180.0 / M_PI);  // deg
        imu_cov_pos_.push_back(std::sqrt(C.block<3, 3>(3, 3).trace()));                 // m

        // NIS metric B: IMU-prediction pose innovation (dof 6). nu = Log(T_pred^-1 T_opt)
        // between the IMU-predicted prior pose and the optimized posterior; S = preintMeasCov
        // pose block. Caveat: prior and posterior share the graph -> correlated -> optimistic
        // (an IMU-consistency indicator, not an independent innovation).
        if (params_.dump_covariance) {
            const gtsam::Pose3 T_opt = post_update.at<gtsam::Pose3>(X(state_idx_));
            const gtsam::Vector6 nu =
                gtsam::Pose3::Logmap(nav_pred.pose().inverse().compose(T_opt));
            const Eigen::Matrix<double, 6, 6> S = C.block<6, 6>(0, 0);
            const double nis_b = nu.dot(S.inverse() * nu);
            nis_b_ndof_.push_back(nis_b / 6.0);
            nis_b_gate_.push_back(nis_b < chi2_p95(6.0) ? 1.0 : 0.0);
        }
    }
    // Bias sanity clamp (VINS failureDetection-style, but non-destructive). A
    // physical MEMS bias is tiny; if the incremental solve returns a non-physical
    // bias it is diverging (accel bias absorbs real acceleration -> pim predicts a
    // huge accel next step -> velocity runaway). Do NOT reseed the preintegrator
    // with garbage: keep the last sane bias so the feedback loop is broken. The bad
    // B node ages out of the lag window on its own. (Guards the incremental back-end on
    // MH_04's fast segment, seen under the removed time window.)
    const double ba_n = new_bias.accelerometer().norm();
    const double bg_n = new_bias.gyroscope().norm();
    const bool clamp = (params_.bias_accel_max > 0.0 && ba_n > params_.bias_accel_max)
                    || (params_.bias_gyro_max  > 0.0 && bg_n > params_.bias_gyro_max);
    if (clamp) {
        std::cerr << "[bias-clamp] kf=" << state_idx_ << " reject |ba|=" << ba_n
                  << " |bg|=" << bg_n << " (keep prev |ba|="
                  << current_bias_.accelerometer().norm() << ")\n";
    } else {
        current_bias_ = new_bias;
    }
    pim_->resetIntegrationAndSetBias(current_bias_);
    latest_estimate_ = post_update;

    // --- Post-fit reprojection residual diagnostic (vision-side covariance) ---
    // Evaluate THIS KF's reprojection factors at the OPTIMIZED pose. graph_ still holds
    // them (cleared later). raw RMS[px] tells whether sigma_px matches the data; the
    // R-normalized chi2 (target ~1) separates sigma_px mis-specification from visual
    // information over-counting. invdepth mode only; gated on dump_covariance.
    if (params_.dump_covariance && !smart_mode_ && !world_mode_) {
        double sse = 0.0; int nobs = 0;
        for (const auto& f : graph_) {
            auto* pf = dynamic_cast<const InvDepthProjFactor*>(f.get());
            if (!pf) continue;
            const gtsam::Key ka = pf->keys()[0], ko = pf->keys()[1], kl = pf->keys()[2];
            if (!post_update.exists(ka) || !post_update.exists(ko) || !post_update.exists(kl)) continue;
            const gtsam::Vector r = pf->evaluateError(post_update.at<gtsam::Pose3>(ka),
                post_update.at<gtsam::Pose3>(ko), post_update.at<gtsam::Vector3>(kl));
            sse += r.squaredNorm(); nobs += 1;
        }
        if (nobs > 0) {
            const double sp = params_.pt_sigma_px;
            reproj_rms_.push_back(std::sqrt(sse / (2.0 * nobs)));          // px RMS / component
            reproj_chi2_.push_back(sse / (sp * sp) / (2.0 * nobs));        // ~1 if sigma_px calibrated
        }
    }

    // ---------------------------------------------------------------
    // 6a. Post-solve point-landmark depth cull (VINS-Mono setDepth/removeFailures
    //     analog): entry gates catch bad TRIANGULATIONS; this catches landmarks
    //     the OPTIMIZER drove bad. A landmark whose inverse depth went
    //     non-positive, or whose depth left [min,max] m, stops receiving factors
    //     and marginalizes out by lag (the fixed-lag analog of VINS erasing it).
    if ((params_.pt_cull_depth_min > 0.0 || params_.pt_cull_depth_max > 0.0)
        && !observed_points.empty()) {
        for (int64_t fid : observed_points) {
            auto ik = lm_key_.find(fid);
            if (ik == lm_key_.end()) continue;
            gtsam::Key lk = L(ik->second);
            if (!post_update.exists(lk)) continue;
            // Depth means different things per parametrization: inverse depth is
            // 1/rho in the ANCHOR camera; a world point has to be projected into the
            // CURRENT camera. gtsam::Point3 and gtsam::Vector3 are the same Eigen
            // type, so reading the wrong one silently yields garbage — branch.
            double depth;
            if (world_mode_) {
                const gtsam::Pose3 wTc =
                    post_update.at<gtsam::Pose3>(X(state_idx_)).compose(body_P_cam_);
                depth = wTc.transformTo(post_update.at<gtsam::Point3>(lk)).z();
            } else {
                const gtsam::Vector3 l = post_update.at<gtsam::Vector3>(lk);
                const double rho = l(2);
                depth = (rho > 1e-9) ? 1.0 / rho : -1.0;
            }
            bool bad = depth <= 0.0;   // behind anchor / at-infinity wrong side
            if (!bad && params_.pt_cull_depth_min > 0.0
                     && depth < params_.pt_cull_depth_min) bad = true;
            if (!bad && params_.pt_cull_depth_max > 0.0
                     && depth > params_.pt_cull_depth_max) bad = true;
            if (bad) {
                landmarks_in_graph_.erase(fid);
                lm_anchor_.erase(fid);
                lm_last_seen_.erase(fid);
                lm_key_.erase(fid);
                feature_obs_.erase(fid);
                lm_chain_reanchors_.erase(fid);
                ++pt_cull_removed_;
            }
        }
    }

    // ---------------------------------------------------------------
    // 7. Window bookkeeping + cleanup
    // ---------------------------------------------------------------
    current_time_ = w;
    state_time_[state_idx_] = w;
    for (auto it = state_time_.begin(); it != state_time_.end(); ) {
        if (it->first != state_idx_ && (current_time_ - it->second) >= lag_)
            it = state_time_.erase(it);
        else ++it;
    }

    graph_.resize(0);
    initial_estimates_.clear();

    last_kf_obs_ = observations;   // reference for the next parallax test
    return true;
}

bool VioBackend::decide_keyframe(const std::unordered_map<int64_t, Eigen::Vector2d>& obs) {
    frames_since_kf_++;

    if (last_kf_obs_.empty())                 { kf_first_++;    return true; }
    if (frames_since_kf_ >= kf_max_interval_) { kf_interval_++; return true; }

    // ---- DSO three-term flow score (FullSystem::needNewKf) ----
    // The IMU-predicted rotation splits each feature's optical flow into
    //   flow_T  = ||p_cur - p_pred||   translation-induced (rotation warped out)
    //   flow_R  = ||p_pred - p_ref||   rotation-induced
    //   flow_RT = ||p_cur - p_ref||    the full displacement (what we used to threshold)
    // and the gate fires when w_t*flow_T + w_r*flow_R + w_rt*flow_RT >= kf_parallax_px_.
    //
    // Keyframes serve TWO purposes and the three terms separate them. Only translation
    // buys triangulation baseline, so DSO gives rotation-only flow weight exactly zero
    // (setting_maxShiftWeightR = 0.0f). But rotation still sweeps features out of view and
    // destroys co-visibility between the window and the current frame, so DSO keeps the
    // COMBINED term at half the translation weight (maxShiftWeightT/RT = 0.04/0.02) rather
    // than ignoring rotation outright. Measured here: dropping the combined term (w=1,0,0)
    // is worse on ATE and NEES at every threshold including matched keyframe rate --
    // exactly the co-visibility failure the RT term exists to prevent.
    //
    // pim_ integrates from the LAST KEYFRAME (reset per keyframe at the end of add_frame)
    // and last_kf_obs_ is that same keyframe's observations, so the preintegrated rotation
    // is exactly the delta between the two views. The rotation warp is exact only for
    // points at infinity -- the standard approximation, and the one VINS-Mono's disabled
    // compensatedParallax2 path uses too.
    const bool use_rot = (kf_w_t_ != 0.0 || kf_w_r_ != 0.0) && pim_ && pim_->deltaTij() > 0.0;
    gtsam::Rot3 cRc_inv;                      // rotates a last-KF bearing into the current cam
    if (use_rot) {
        const gtsam::Rot3& bRc = body_P_cam_.rotation();
        cRc_inv = (bRc.inverse() * pim_->deltaRij() * bRc).inverse();
    }

    int n_common = 0;
    double sum_t = 0.0, sum_r = 0.0, sum_rt = 0.0;
    for (const auto& kv : obs) {
        auto it = last_kf_obs_.find(kv.first);
        if (it == last_kf_obs_.end()) continue;
        const gtsam::Point2 p_ref(it->second(0), it->second(1));
        const gtsam::Point2 p_cur(kv.second(0), kv.second(1));
        const double raw = (p_cur - p_ref).norm();
        sum_rt += raw;

        double f_t = raw, f_r = 0.0;   // no compensation -> all flow counts as translation
        if (use_rot) {
            const gtsam::Point2 xy = calib_->calibrate(p_ref);
            const gtsam::Point3 b = cRc_inv.rotate(gtsam::Point3(xy.x(), xy.y(), 1.0));
            if (b.z() > 1e-6) {        // rotated behind the camera -> leave the raw split
                const gtsam::Point2 p_pred =
                    calib_->uncalibrate(gtsam::Point2(b.x() / b.z(), b.y() / b.z()));
                f_t = (p_cur - p_pred).norm();
                f_r = (p_pred - p_ref).norm();
            }
        }
        sum_t += f_t;
        sum_r += f_r;
        ++n_common;
    }

    if (n_common > 0) {   // diagnostics: how the flow splits translation / rotation
        kf_par_raw_sum_  += sum_rt / n_common;
        kf_par_comp_sum_ += sum_t  / n_common;
        kf_par_rot_sum_  += sum_r  / n_common;
        ++kf_par_n_;
    }

    if (n_common < kf_min_tracked_) { kf_lowtrack_++; return true; }   // view changed fast
    const double score = (kf_w_t_  * sum_t + kf_w_r_ * sum_r + kf_w_rt_ * sum_rt)
                       / (double)n_common;
    if (score >= kf_parallax_px_) { kf_parallax_n_++; return true; }
    return false;
}

void VioBackend::print_kf_stats() const {
    long tot = kf_first_ + kf_interval_ + kf_lowtrack_ + kf_parallax_n_;
    std::cout << "[KF stats] total=" << tot
              << "  first=" << kf_first_
              << "  interval(>=" << kf_max_interval_ << "f)=" << kf_interval_
              << "  lowtrack(<" << kf_min_tracked_ << ")=" << kf_lowtrack_
              << "  parallax(>=" << kf_parallax_px_ << "px)=" << kf_parallax_n_ << "\n";

    // Mean per-decision flow, split three ways. With the rotation warp OFF (w_t = w_r = 0)
    // flow_T degenerates to flow_RT and flow_R is 0 by construction; with it ON,
    // flow_R / flow_RT is the fraction of our measured "parallax" that was rotation
    // carrying no triangulation baseline.
    if (kf_par_n_ > 0) {
        const double f_rt = kf_par_raw_sum_  / (double)kf_par_n_;
        const double f_t  = kf_par_comp_sum_ / (double)kf_par_n_;
        const double f_r  = kf_par_rot_sum_  / (double)kf_par_n_;
        std::cout << "[KF flow] mean T=" << f_t << "px  R=" << f_r << "px  RT=" << f_rt
                  << "px  rotation share=" << (f_rt > 1e-9 ? 100.0 * f_r / f_rt : 0.0)
                  << "%  score=" << (kf_w_t_ * f_t + kf_w_r_ * f_r + kf_w_rt_ * f_rt)
                  << "px  w(T,R,RT)=(" << kf_w_t_ << "," << kf_w_r_ << "," << kf_w_rt_ << ")\n";
    }

    // IMU preintegration-length loss. If the bias-correction displacement is sub-mm and
    // the factor's claimed pos-sigma is small at our deltaTij, collapsing the
    // between-keyframe IMU into one long CombinedImuFactor is NOT lossy at this rate.
    if (!imu_dt_.empty()) {
        auto q = [](std::vector<double> v, double f) {
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
        };
        std::cout << "[IMU preint] N=" << imu_dt_.size()
                  << "  dt[s] med=" << q(imu_dt_, 0.5) << " p90=" << q(imu_dt_, 0.9)
                  << " max=" << q(imu_dt_, 0.999)
                  << "  |  bias-corr pos[mm] med=" << 1e3 * q(imu_bcorr_pos_, 0.5)
                  << " p90=" << 1e3 * q(imu_bcorr_pos_, 0.9)
                  << " max=" << 1e3 * q(imu_bcorr_pos_, 0.999)
                  << "  rot[deg] med=" << q(imu_bcorr_rot_, 0.5)
                  << " p90=" << q(imu_bcorr_rot_, 0.9)
                  << "  |  cov pos-sig[m] med=" << q(imu_cov_pos_, 0.5)
                  << " p90=" << q(imu_cov_pos_, 0.9)
                  << "  cov rot-sig[deg] med=" << q(imu_cov_rot_, 0.5) << "\n";
    }

    // NIS consistency (GT-free). A = reprojection pre-fit (dof 2n, pose-only S -> conservative);
    // B = IMU-prediction pose (dof 6, optimistic -- prior/posterior correlated). Both as
    // NIS/dof (consistent estimator -> ~1) plus the within-95% chi-square gate.
    if (!nis_a_ndof_.empty() || !nis_b_ndof_.empty()) {
        auto q = [](std::vector<double> v, double f) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
        };
        auto pct = [](const std::vector<double>& v) {
            if (v.empty()) return 0.0;
            double s = 0.0; for (double x : v) s += x; return 100.0 * s / v.size();
        };
        std::cout << "[NIS] reproj(A): N=" << nis_a_ndof_.size()
                  << " med(NIS/dof)=" << q(nis_a_ndof_, 0.5)
                  << " p90=" << q(nis_a_ndof_, 0.9)
                  << " within95=" << pct(nis_a_gate_) << "%"
                  << "  |  imu(B): N=" << nis_b_ndof_.size()
                  << " med(NIS/6)=" << q(nis_b_ndof_, 0.5)
                  << " p90=" << q(nis_b_ndof_, 0.9)
                  << " within95=" << pct(nis_b_gate_) << "%\n";
    }

    // Post-fit reprojection: RMS[px] vs sigma_px, and R-normalized chi2/dof (~1 if
    // sigma_px matches the data). Diagnoses whether the vision-side over-confidence is
    // sigma_px mis-specification or visual information over-counting.
    if (!reproj_rms_.empty()) {
        auto q = [](std::vector<double> v, double f) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
        };
        std::cout << "[reproj] post-fit residual: N=" << reproj_rms_.size()
                  << "  RMS[px] med=" << q(reproj_rms_, 0.5) << " p90=" << q(reproj_rms_, 0.9)
                  << "  chi2/dof(sigma_px=" << params_.pt_sigma_px << ") med="
                  << q(reproj_chi2_, 0.5) << " p90=" << q(reproj_chi2_, 0.9) << "\n";
    }

    // Point-landmark bulletproofing counters (only when either mechanism is on).
    if (params_.pt_obs_reject_px > 0.0 || params_.pt_cull_depth_min > 0.0
        || params_.pt_cull_depth_max > 0.0) {
        std::cout << "[pointgate] obs-rejected=" << pt_obs_rejected_
                  << "  depth-culled=" << pt_cull_removed_
                  << "  depth-transferred=" << pt_depth_transferred_
                  << "  (reject_px=" << params_.pt_obs_reject_px
                  << ", depth=[" << params_.pt_cull_depth_min
                  << "," << params_.pt_cull_depth_max << "])\n";
    }
    // Landmark continuity counters (only when either switch is on, so the committed
    // system's output is unchanged).
    if (reanchor_exact_ || params_.lm_init_current_obs) {
        std::cout << "[continuity] init-current-obs added=" << lm_init_obs_added_
                  << " rejected=" << lm_init_obs_rejected_
                  << "  reanchored=" << lm_reanchored_
                  << " declined=" << lm_reanchor_declined_;
        if (params_.lm_reanchor_max > 0)
            std::cout << " capped=" << lm_reanchor_capped_ << " (reanchor_max " << params_.lm_reanchor_max << ")";
        if (params_.lm_reanchor_live_max > 0)
            std::cout << " full=" << lm_reanchor_full_ << " (reanchor_live_max " << params_.lm_reanchor_live_max << ")";
        if (reanchor_exact_ && lm_live_n_ > 0)
            std::cout << " live_mean=" << static_cast<double>(lm_live_sum_) / lm_live_n_
                      << " live_peak=" << lm_live_peak_;
        if (params_.lm_reanchor_retire) std::cout << " retire=on";
        std::cout << "\n";
    }
}


Eigen::Matrix4d VioBackend::get_current_pose() {
    gtsam::Pose3 p = latest_estimate_.at<gtsam::Pose3>(X(state_idx_));
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = p.rotation().matrix();
    T.block<3, 1>(0, 3) = p.translation();
    return T;
}

// voxl-gtsam-vio patch 0001 (read-only). See the declaration.
VioBackend::NavOut VioBackend::get_nav_out() const {
    NavOut o;
    if (!latest_estimate_.exists(X(state_idx_)) || !latest_estimate_.exists(V(state_idx_)))
        return o;
    const gtsam::Pose3   pose = latest_estimate_.at<gtsam::Pose3>(X(state_idx_));
    const gtsam::Vector3 vel  = latest_estimate_.at<gtsam::Vector3>(V(state_idx_));
    o.T_kf = pose.matrix();
    o.v_kf = vel;
    o.bg   = current_bias_.gyroscope();
    o.ba   = current_bias_.accelerometer();
    if (pim_ && pim_->deltaTij() > 0.0) {
        const gtsam::NavState s = pim_->predict(gtsam::NavState(pose, vel), current_bias_);
        o.T_now = s.pose().matrix();
        o.v_now = s.v();
    } else {
        o.T_now = o.T_kf;
        o.v_now = o.v_kf;
    }
    o.ok = true;
    return o;
}

} // namespace vio