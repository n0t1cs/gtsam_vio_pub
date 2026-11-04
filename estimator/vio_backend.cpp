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

// ISAM2 parameters for the incremental smoother. findUnusedFactorSlots: the slots of the factors
// that marginalization removes are reused, so the factor list stays as long as the window (every
// result since the smart-factor study ran with it). relinearize_threshold / skip come from config.
// The threshold does not act as an FEJ (threshold 0 changes nothing). iSAM2's
// FEJ-like behaviour is its built-in marginal freeze (fixedVariables_). CHOLESKY (GTSAM's default):
// at the keyframe window it gives an estimate identical to QR on 11/11 EuRoC sequences at about a
// quarter of the latency.
static gtsam::ISAM2Params make_isam2_params(double relin_threshold, int relin_skip) {
    gtsam::ISAM2Params p;
    p.findUnusedFactorSlots = true;
    p.relinearizeThreshold  = relin_threshold;
    p.relinearizeSkip       = relin_skip;
    p.factorization         = gtsam::ISAM2Params::CHOLESKY;
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

    // Backend smoother: incremental (ISAM2, the committed default) or the KeyframeSmoother.
    if (params_.backend_mode == "incremental") {
        auto isam = std::make_unique<gtsam::IncrementalFixedLagSmoother>(
            lag_, make_isam2_params(params_.isam_relin_threshold, params_.isam_relin_skip));
        isam_view_ = isam.get();          // non-owning view for the covariances
        smoother_  = std::move(isam);
        std::cout << "[VIO Backend] smoother = IncrementalFixedLagSmoother (ISAM2), factorization = CHOLESKY\n";
    } else if (params_.backend_mode == "keyframe") {
        // The SLKS. STANDALONE keyframe-indexed fixed-lag smoother with
        // explicit Schur marginalization (keyframe_smoother.h) — NOT a
        // gtsam::FixedLagSmoother, so it is owned in its own pointer (kf_smoother_)
        // and smoother_ stays null; VioBackend drives both through run_update()/
        // run_estimate(). The window is N_kf keyframes, owned inside the smoother.
        // lag_ remains the back-end's OWN re-anchoring horizon (keyframe count), which
        // matches the smoother's window by construction.

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

        // Solve-guard: only LANDMARKS may be dropped when the window's linear system goes
        // indeterminate. A pose/velocity/bias degeneracy is a real failure and must NOT be
        // masked, so the predicate excludes them — the guard rethrows in that case.
        KeyframeSmoother::CullKeyFilter cull_filter =
            [](gtsam::Key k) { return gtsam::Symbol(k).chr() == 'l'; };
        kf_smoother_ = std::make_unique<KeyframeSmoother>(params_.kf_window_size,
                                                          params_.kf_use_fej,
                                                          std::move(fej_filter),
                                                          params_.kf_max_iterations,
                                                          std::move(cull_filter));
        std::cout << "[VIO Backend] smoother = KeyframeSmoother (standalone, keyframe window N_kf="
                  << params_.kf_window_size
                  << ", FEJ=" << (params_.kf_use_fej ? "on" : "off")
                  << ", scope=" << params_.kf_fej_scope << ", GN)\n";
    } else {
        throw std::invalid_argument("[VIO Backend] backend_mode='" + params_.backend_mode +
                                    "' unknown (incremental | keyframe)");
    }
    // Echo the window actually built, so a campaign can grep back that its config took.
    if (smoother_)
        std::cout << "[VIO Backend] fixed-lag window = keyframe index, lag=" << lag_ << " KF\n";

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

    // Exact re-anchoring (landmark.reanchor: exact; VioParams::load rejects unsupported
    // combinations).
    reanchor_exact_ = (params_.lm_reanchor == "exact");
    if (reanchor_exact_) {
        reanchor_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, params_.lm_reanchor_sigma);
        std::cout << "[VIO Backend] landmark continuity: reanchor=" << params_.lm_reanchor
                  << " (sigma " << params_.lm_reanchor_sigma << ")"
                  << "  retire=" << (params_.lm_reanchor_retire ? "on" : "off")
                  << "  live_max=" << params_.lm_reanchor_live_max << "\n";
    }
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
    // A degenerate window (too short / inconsistent) seeds a single state instead.
    if (N < 3 || (int)vels.size() != N || (int)frames.size() != N) {
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
    for (auto& kv : obs_of) {
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

// 12x12 joint of X(ia), X(ib), block order [ia, ib].
static gtsam::Matrix joint12(const gtsam::Marginals& marg, int ia, int ib) {
    gtsam::JointMarginal jm = marg.jointMarginalCovariance(gtsam::KeyVector{X(ia), X(ib)});
    gtsam::Matrix J12(12, 12);
    J12.block<6,6>(0,0) = jm.at(X(ia), X(ia));  J12.block<6,6>(0,6) = jm.at(X(ia), X(ib));
    J12.block<6,6>(6,0) = jm.at(X(ib), X(ia));  J12.block<6,6>(6,6) = jm.at(X(ib), X(ib));
    return J12;
}

// d(Ta^-1 Tb)/d(Ta), d(...)/d(Tb) -> J = [H1 H2] (6x12); Sigma_rel = J S J^T.
static void between_covariance(const gtsam::Matrix& J12, const gtsam::Pose3& Ta, const gtsam::Pose3& Tb,
                               Eigen::Matrix<double, 6, 6>& out) {
    Eigen::Matrix<double, 6, 6> H1, H2;
    Ta.between(Tb, H1, H2);
    Eigen::Matrix<double, 6, 12> J;
    J.block<6,6>(0,0) = H1;
    J.block<6,6>(0,6) = H2;
    out = J * J12 * J.transpose();
    out = 0.5 * (out + out.transpose());   // re-symmetrize against round-off
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
            J12 = joint12(gtsam::Marginals(isam_view_->getFactors(), isam_view_->getLinearizationPoint()), ia, ib);
        } else {
            return false;
        }
        between_covariance(J12, latest_estimate_.at<gtsam::Pose3>(X(ia)),
                           latest_estimate_.at<gtsam::Pose3>(X(ib)), out);
    } catch (const std::exception& e) {
        static int n = 0;
        if (n++ < 3) std::cerr << "[cov] jointMarginalCovariance(X(" << ia << "),X(" << ib
                               << ")) failed: " << e.what() << "\n";
        return false;
    }
    return out.allFinite();
}

bool VioBackend::covariance_snapshot(CovSnapshot& s) const {
    if (isam_view_) {
        s.graph = isam_view_->getFactors();
        s.theta = isam_view_->getLinearizationPoint();
    } else if (kf_smoother_) {
        s.graph = kf_smoother_->getFactors();
        s.theta = kf_smoother_->getLinearizationPoint();
    } else {
        return false;
    }
    s.poses.clear();
    for (const auto kv : latest_estimate_)
        if (gtsam::Symbol(kv.key).chr() == 'x') s.poses.insert(kv.key, kv.value);
    return true;
}

bool VioBackend::pose_covariance(const CovSnapshot& s, const gtsam::Marginals& m, int idx,
                                 Eigen::Matrix<double, 6, 6>& out) {
    if (!s.poses.exists(X(idx))) return false;
    try {
        out = m.marginalCovariance(X(idx));
    } catch (const std::exception& e) {
        static int n = 0;
        if (n++ < 3) std::cerr << "[cov] marginalCovariance(X(" << idx << ")) failed: " << e.what() << "\n";
        return false;
    }
    return true;
}

bool VioBackend::relative_pose_covariance(const CovSnapshot& s, const gtsam::Marginals& m, int ia, int ib,
                                          Eigen::Matrix<double, 6, 6>& out) {
    if (ia == ib) return false;
    if (!s.poses.exists(X(ia)) || !s.poses.exists(X(ib))) return false;
    try {
        between_covariance(joint12(m, ia, ib), s.poses.at<gtsam::Pose3>(X(ia)), s.poses.at<gtsam::Pose3>(X(ib)), out);
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

    X_w_out = X_w;
    return true;
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
    }
    for (const auto& kv : observations) {
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
                // landmark.reanchor_live_max: when the live chains fill the cap, an exit that
                // would start a chain falls back to abandon. Running chains continue.
                const auto ich = lm_chain_reanchors_.find(fid);
                const int chain_n = (ich != lm_chain_reanchors_.end()) ? ich->second : 0;
                const bool full = params_.lm_reanchor_live_max > 0 && chain_n == 0
                                  && live_chains >= params_.lm_reanchor_live_max;
                const bool exit_fresh = reanchor_exact_ && anchor >= 0 && lkey != 0
                    && latest_estimate_.exists(lkey) && current.exists(X(anchor))
                    && is != lm_last_seen_.end() && (w - is->second) < lag_;
                if (exit_fresh && !full) {
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
                        graph_.emplace_shared<InvDepthReanchorFactor>(
                            lkey, X(anchor), X(state_idx_), lk, body_P_cam_, reanchor_noise_);
                        // landmark.reanchor_retire: the old variable leaves with its anchor.
                        if (params_.lm_reanchor_retire) ts[lkey] = win_coord(anchor);
                        // the new anchor's own measurement: a new pixel, used once
                        graph_.emplace_shared<InvDepthAnchorFactor>(
                            X(state_idx_), lk, gtsam::Point2(uv(0), uv(1)),
                            body_P_cam_, calib_, cam_noise_);
                        if (chain_n == 0) ++live_chains;
                        lm_chain_reanchors_[fid] = chain_n + 1;
                        continue;
                    }
                    // otherwise falls through to abandon
                }

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
                if (reject) continue;
            }
            // cross-view: anchor + current keyframe observe the same inv-depth lm
            graph_.emplace_shared<InvDepthProjFactor>(
                X(anchor), X(state_idx_), lkey,
                gtsam::Point2(uv(0), uv(1)), body_P_cam_, calib_, cam_noise_);
            ts[lkey] = w;
            lm_last_seen_[fid] = w;
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
                            body_P_cam_, calib_, cam_noise_);

                        // cross-view factors (these constrain rho)
                        for (const auto& obs : live) {
                            if (obs.first == a_idx) continue;
                            graph_.emplace_shared<InvDepthProjFactor>(
                                X(a_idx), X(obs.first), lk,
                                gtsam::Point2(obs.second(0), obs.second(1)),
                                body_P_cam_, calib_, cam_noise_);
                        }
                        feature_obs_.erase(fid);
                    }
                }
            }
        } else if (vec.size() > 15) {
            feature_obs_.erase(fid);
        }
    }

    // ---------------------------------------------------------------
    // 5. Fixed-lag update (GN/ISAM2 over the window; marginalizes old keys)
    // ---------------------------------------------------------------
    run_update(graph_, initial_estimates_, ts);

    // ---------------------------------------------------------------
    // 6. Refresh bias from the new estimate, reset preintegrator.
    // ---------------------------------------------------------------
    const gtsam::Values post_update = run_estimate();
    const gtsam::imuBias::ConstantBias new_bias =
        post_update.at<gtsam::imuBias::ConstantBias>(B(state_idx_));

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

    if (last_kf_obs_.empty())                         return true;   // first
    if (frames_since_kf_ >= params_.kf_max_interval) return true;   // bounds pim_

    // Mean raw pixel displacement of the features shared with the last keyframe. (A DSO-style
    // split into rotation and translation flow was measured and removed: dropping the combined
    // term is worse on ATE and NEES at every threshold, KF_FEJ_smoother.)
    int n_common = 0;
    double sum = 0.0;
    for (const auto& kv : obs) {
        auto it = last_kf_obs_.find(kv.first);
        if (it == last_kf_obs_.end()) continue;
        sum += (gtsam::Point2(kv.second(0), kv.second(1)) - gtsam::Point2(it->second(0), it->second(1))).norm();
        ++n_common;
    }
    if (n_common < params_.kf_min_tracked) return true;                // view changed fast
    return sum / (double)n_common >= params_.kf_parallax_px;
}

Eigen::Matrix4d VioBackend::get_current_pose() {
    gtsam::Pose3 p = latest_estimate_.at<gtsam::Pose3>(X(state_idx_));
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = p.rotation().matrix();
    T.block<3, 1>(0, 3) = p.translation();
    return T;
}

// Read-only, see the declaration.
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