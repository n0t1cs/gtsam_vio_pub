#include "engine.h"

#include "bundle_adjust.h"
#include "init_sfm.h"
#include "offline_helpers.h"
#include "vins_realign.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <opencv2/core.hpp>
#ifdef GVIO_WITH_TBB
#include <tbb/global_control.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace gvio {

using namespace vio;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return 1e3 * std::chrono::duration<double>(Clock::now() - t0).count();
}

VioEngine::VioEngine(EngineConfig cfg)
    : cfg_(std::move(cfg)),
      log_(cfg_.log_dir),
      imu_filter_(cfg_.imu_filter),
      img_q_(cfg_.lockstep ? 2 : std::max(1, cfg_.img_queue)),
      obs_q_(cfg_.lockstep ? 16 : std::max(1, cfg_.obs_queue)),
      fe_imu_q_(0),
      buf_(cfg_.P.init_window, cfg_.P.init_parallax_px, cfg_.P.init_min_common) {
    data_.cam_calib = cfg_.calib;
    const VioParams& P = cfg_.P;
    if (P.loop_closure)
        std::cerr << "[engine] runtime.loop_closure is ignored: the online system is the no-LC odometry\n";
    if (P.frontend_source != "images")
        std::cerr << "[engine] frontend_source '" << P.frontend_source << "' ignored: online tracks its own images\n";
    if (P.init_oracle_gravity)
        std::cerr << "[engine] init.oracle_gravity ignored: it needs ground truth\n";

    std::ostringstream m;
    m << "pinhole K fu " << cfg_.calib.K(0, 0) << " fv " << cfg_.calib.K(1, 1) << " cu " << cfg_.calib.K(0, 2)
      << " cv " << cfg_.calib.K(1, 2) << " size " << cfg_.calib.width << "x" << cfg_.calib.height
      << " undistort " << (cfg_.undistort.remaps() ? "fisheye" : "off");
    log_.meta(m.str());
    std::ostringstream t;
    t << "T_BC " << cfg_.calib.T_BC.row(0) << " | " << cfg_.calib.T_BC.row(1) << " | " << cfg_.calib.T_BC.row(2);
    log_.meta(t.str());
    log_.meta(std::string("imu_filter ") + (cfg_.imu_filter.enable ? "on" : "off") + " lockstep "
              + (cfg_.lockstep ? "on" : "off") + " auto_reset " + (cfg_.auto_reset ? "on" : "off"));
}

VioEngine::~VioEngine() { stop(); }

void VioEngine::start() {
    const VioParams& P = cfg_.P;
#ifdef GVIO_WITH_TBB
    // runtime.deterministic: GTSAM's TBB pinned to one thread, as the offline binary does.
    if (P.deterministic)
        tbb_pin_ = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, 1);
#endif
    FrontendThread::Params fp;                       // main.cpp, verbatim mapping
    fp.max_pts   = P.pt_max;
    fp.min_dist  = P.pt_min_dist_px;
    fp.fb_err    = P.pt_fb_err;
    fp.ransac_th = P.pt_ransac_th;
    fp.quality   = P.pt_quality;
    fp.grid_x    = P.pt_grid_x;
    fp.grid_y    = P.pt_grid_y;
    fp.histeq    = (P.pt_histeq == "clahe") ? 2 : (P.pt_histeq == "equalize") ? 1 : 0;
    fp.clahe_clip = P.pt_clahe_clip;
    fp.rng_seed  = P.deterministic ? P.rng_seed : -1;
    fp.ransac_method = (P.pt_ransac_method == "magsac") ? 1 : (P.pt_ransac_method == "twopt") ? 2 : 0;
    fp.ransac_conf   = P.pt_ransac_conf;
    fp.ransac_iters  = P.pt_ransac_iters;
    frontend_ = std::make_unique<FrontendThread>(data_.cam_calib, fp, img_q_, obs_q_,
                                                 P.pt_imu_aided_klt ? &fe_imu_q_ : nullptr);
    frontend_->start();
    sync_th_ = std::thread(&VioEngine::sync_loop, this);
    est_th_  = std::thread(&VioEngine::estimator_loop, this);
}

// ------------------------------------------------------------------ producers
void VioEngine::push_imu(const ImuSample& raw) {
    std::vector<ImuSample> out;
    bool ok;
    {
        std::lock_guard<std::mutex> lk(imu_filter_m_);
        ok = imu_filter_.push(raw, out);
    }
    {
        std::lock_guard<std::mutex> lk(stats_m_);
        ++stats_.imu_in;
        if (!ok) ++stats_.imu_rejected;
    }
    if (!ok) { input_err_.fetch_or(err::BAD_TIMESTAMP); return; }
    if (out.empty()) return;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (const auto& s : out) imu_fifo_.push_back(s);
        last_imu_t_ = out.back().t_ns;
        // Without frames nothing drains the fifo. Keep the newest ~10 s: after a camera gap above
        // max_frame_gap_s the estimator resets anyway.
        if (imu_fifo_.size() > 20000) imu_fifo_.erase(imu_fifo_.begin(), imu_fifo_.end() - 10000);
    }
    cv_.notify_all();
}

void VioEngine::push_image(int64_t t_ns, int64_t frame_id, const cv::Mat& gray) {
    if (gray.type() != CV_8UC1) { input_err_.fetch_or(err::CAM_BAD_FORMAT); return; }
    if (gray.cols != cfg_.undistort.width() || gray.rows != cfg_.undistort.height()) {
        input_err_.fetch_or(err::CAM_BAD_RES);
        return;
    }
    Pending p{t_ns, frame_id, gray.clone(), Clock::now()};
    bool dropped = false;
    {
        std::unique_lock<std::mutex> lk(m_);
        if (last_frame_t_ >= 0 && t_ns <= last_frame_t_) {
            lk.unlock();
            input_err_.fetch_or(err::BAD_TIMESTAMP);
            std::lock_guard<std::mutex> ls(stats_m_);
            ++stats_.frames_rejected;
            return;
        }
        last_frame_t_ = t_ns;
        if (cfg_.lockstep) {
            cv_.wait(lk, [&] { return stop_ || (int)pending_.size() < std::max(1, cfg_.max_pending_frames); });
            if (stop_) return;
        } else if ((int)pending_.size() >= std::max(1, cfg_.max_pending_frames)) {
            pending_.pop_front();                    // keep latency bounded: the oldest goes
            dropped = true;
        }
        pending_.push_back(std::move(p));
    }
    cv_.notify_all();
    std::lock_guard<std::mutex> ls(stats_m_);
    ++stats_.frames_in;
    if (dropped) { ++stats_.dropped_backlog; input_err_.fetch_or(err::DROPPED_CAM); }
}

// ------------------------------------------------------------------ sync
void VioEngine::sync_loop() {
    for (;;) {
        Pending f;
        std::vector<ImuSample> batch;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || input_done_ || !pending_.empty(); });
            if (stop_ || (pending_.empty() && input_done_)) break;
            const int64_t t = pending_.front().t_ns;
            const auto deadline = Clock::now() + std::chrono::duration<double>(cfg_.imu_wait_timeout_s);
            while (!stop_ && !input_done_ && !imu_done_ && last_imu_t_ < t && !pending_.empty()
                   && pending_.front().t_ns == t) {
                if (cfg_.lockstep) cv_.wait(lk);
                else if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) break;
            }
            if (stop_) break;
            if (pending_.empty() || pending_.front().t_ns != t) continue;   // dropped while waiting
            f = std::move(pending_.front());
            pending_.pop_front();
            // Never covered: drop it, its IMU stays queued. At the end of the input (or of the IMU)
            // no more IMU can come, so the frame goes on with what there is, as the offline driver
            // does with the frames past a dataset's last IMU sample (the _lpf IMU ends 16.6 ms early).
            if (last_imu_t_ < f.t_ns && !input_done_ && !imu_done_) {
                lk.unlock();
                cv_.notify_all();
                input_err_.fetch_or(err::DROPPED_CAM);
                std::lock_guard<std::mutex> ls(stats_m_);
                ++stats_.dropped_no_imu;
                continue;
            }
            while (!imu_fifo_.empty() && imu_fifo_.front().t_ns <= f.t_ns) {
                batch.push_back(imu_fifo_.front());
                imu_fifo_.pop_front();
            }
        }
        cv_.notify_all();                            // a lockstep producer may wait for room
        // The offline producer's order: every IMU sample up to the frame stamp, then the frame.
        {
            std::lock_guard<std::mutex> lk(est_imu_m_);
            for (const auto& s : batch) est_imu_.push_back(s);
        }
        for (const auto& s : batch) fe_imu_q_.push(s);
        {
            std::lock_guard<std::mutex> lk(meta_m_);
            meta_[f.t_ns] = FrameMeta{f.frame_id, f.entry};
        }
        StampedImage si;
        si.t_ns = f.t_ns;
        si.img = cfg_.undistort.apply(f.img);
        img_q_.push(std::move(si));
    }
    img_q_.close();
    fe_imu_q_.close();
}

// ------------------------------------------------------------------ estimator
void VioEngine::estimator_loop() {
    // main.cpp seeds the thread that runs the initialisation and the backend. cv::theRNG() is
    // thread-local, so this thread seeds its own.
    cv::setRNGSeed(0);
    std::srand(0);
    if (cfg_.P.rng_seed != 0) {
        cv::setRNGSeed(cfg_.P.rng_seed);
        std::srand(static_cast<unsigned>(cfg_.P.rng_seed));
    }
    log_.start_session(session_, cfg_.P.dump_covariance, cfg_.P.dump_relcov);
    StampedObservation so;
    while (obs_q_.pop(so)) process(so);
}

void VioEngine::process(const StampedObservation& so) {
    // IMU up to this frame (the sync thread queued it before the frame).
    {
        std::lock_guard<std::mutex> lk(est_imu_m_);
        while (!est_imu_.empty() && est_imu_.front().t_ns <= so.t_ns) {
            data_.imu.push_back(est_imu_.front());
            est_imu_.pop_front();
        }
    }
    FrameMeta meta{-1, Clock::now()};
    {
        std::lock_guard<std::mutex> lk(meta_m_);
        auto it = meta_.find(so.t_ns);
        if (it != meta_.end()) { meta = it->second; meta_.erase(it); }
    }
    if (t0_ns_ < 0) {
        t0_ns_ = so.t_ns;
        data_.frames.assign(1, Frame{t0_ns_, ""});   // init_align and the excitation check read front()
        log_.meta("t0_ns " + std::to_string(t0_ns_) + " (time base of every session's TUM and CSV)");
    }
    const double t_s = (so.t_ns - t0_ns_) * 1e-9;

    if (reset_hard_.exchange(false)) reset(true, so.t_ns, "control: hard reset");
    if (reset_soft_.exchange(false)) reset(false, so.t_ns, "control: soft reset");

    EngineOutput out;
    out.t_ns = so.t_ns;
    out.frame_id = meta.frame_id;
    out.n_features = (int)so.obs.ids.size();
    const auto& K = data_.cam_calib.K;
    for (size_t j = 0; j < so.obs.ids.size(); ++j)
        out.features.emplace_back(so.obs.ids[j],
                                  Eigen::Vector2d(K(0, 0) * so.obs.pts_norm(j, 0) + K(0, 2),
                                                  K(1, 1) * so.obs.pts_norm(j, 1) + K(1, 2)));
    if (!data_.imu.empty()) out.w_body = data_.imu.back().w;

    ++frame_idx_;
    if (latched_fail_) {
        out.state = STATE_FAILED;
    } else if (!running_) {
        run_init(t_s, so.obs, out);
    } else {
        run_online(t_s, so.obs, out);
    }
    prev_obs_ = so.obs;
    out.session = session_;
    out.session_hard = session_hard_;
    out.error_code |= input_err_.exchange(0) | sticky_err_;
    sticky_err_ = 0;
    out.latency_ms = ms_since(meta.entry);
    log_.frame(out);
    {
        std::lock_guard<std::mutex> lk(stats_m_);
        ++stats_.frames_out;
        if (out.is_keyframe) ++stats_.keyframes;
    }
    if (on_output_) on_output_(out);
    if (out.state == STATE_FAILED && running_) {     // lost: reset after reporting the failure
        if (cfg_.auto_reset) reset(false, so.t_ns, "auto reset");
        else { latched_fail_ = true; backend_.reset(); running_ = false; }
    }
    trim_imu();
}

// main.cpp Phase 1, one frame of the loop.
void VioEngine::run_init(double t_s, const Observation& obs, EngineOutput& out) {
    const VioParams& P = cfg_.P;
    out.state = STATE_INITIALIZING;
    const int i = frame_idx_;
    if (!buf_.try_add(i, t_s, obs) || !buf_.is_full()) return;
    bool done = false;
    if (check_imu_excitation(buf_.frames, data_, P.init_excitation_var)) {
        { std::lock_guard<std::mutex> lk(stats_m_); ++stats_.init_attempts; }
        SfmResult sfm_try = init_sfm(buf_.frames, data_.cam_calib);
        if (sfm_try.success) {
            SfmResult sfm_ba = sfm_try;
            bundle_adjust(sfm_ba, 20);
            double pre_ba_rms = sfm_ba.ba_n_obs > 0
                ? std::sqrt(2.0 * sfm_ba.ba_err_before / (double)sfm_ba.ba_n_obs)
                : 1e9;
            if (pre_ba_rms > P.init_max_pre_ba_rms_px) {
                std::cout << "init: window rejected -- SfM ill-conditioned (pre-BA reproj RMS "
                          << pre_ba_rms << " px > " << P.init_max_pre_ba_rms_px << " px); sliding.\n";
            } else {
                AlignedInit align_try = init_align(sfm_ba, data_, P.init_est_ba);
                if (align_try.success) {
                    double vmax = 0.0;
                    for (const auto& vv : align_try.velocities)
                        if (vv.norm() > vmax) vmax = vv.norm();
                    if (vmax > P.init_max_vel_mps) {
                        std::cout << "init: window rejected -- VI-alignment implausible (max|v|="
                                  << vmax << " m/s > " << P.init_max_vel_mps << " m/s); sliding.\n";
                    } else {
                        AlignedInit aligned = align_try;
                        vins_realign(aligned);
                        std::cout << "\n[SUCCESS] Initialization Window Resolved (pre-BA reproj RMS "
                                  << pre_ba_rms << " px).\n";
                        std::ostringstream e;
                        e << "session " << session_ << " initialised at frame " << i << " t_s " << t_s
                          << " scale " << aligned.scale << " max|v| " << vmax << " |b_g| " << aligned.b_g.norm();
                        log_.event(out.t_ns, e.str());
                        start_backend(aligned);
                        done = true;
                    }
                }
            }
        }
    }
    if (!done) { buf_.shift(); return; }
    // The init window's newest keyframe is this frame: report it as the first OK output.
    const auto nav = backend_->get_nav_out();
    out.state = STATE_OK;
    out.is_keyframe = true;
    out.T_wb = nav.T_now; out.v_w = nav.v_now; out.bg = nav.bg; out.ba = nav.ba;
    out.w_body -= nav.bg;
}

// main.cpp Phase 2.
void VioEngine::start_backend(AlignedInit& aligned) {
    const VioParams& P = cfg_.P;
    auto imu_params = gtsam::PreintegrationCombinedParams::MakeSharedU(aligned.g_w.norm());
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    imu_params->setAccelerometerCovariance(I3 * std::pow(P.imu_acc_n, 2) * P.imu_acc_cov_mult);
    imu_params->setGyroscopeCovariance    (I3 * std::pow(P.imu_gyr_n, 2) * P.imu_gyr_cov_mult);
    imu_params->setIntegrationCovariance  (I3 * P.imu_integration_cov);
    imu_params->setBiasAccCovariance      (I3 * std::pow(P.imu_acc_w, 2) * P.imu_acc_bias_cov_mult);
    imu_params->setBiasOmegaCovariance    (I3 * std::pow(P.imu_gyr_w, 2) * P.imu_gyr_bias_cov_mult);
    imu_params->biasAccOmegaInt = Eigen::Matrix<double, 6, 6>::Identity() * P.imu_bias_acc_omega_int;

    backend_ = std::make_unique<VioBackend>(data_.cam_calib.K, data_.cam_calib.T_BC, imu_params, P);
    backend_->initialize_window(aligned.T_wb, aligned.velocities, aligned.sfm.frames,
                                aligned.sfm.points, aligned.b_g, aligned.b_a, data_.imu, t0_ns_);
    // The init-window block of the trajectory, exactly as main.cpp writes it: the backend's poses
    // right after initialize_window, and their covariance when runtime.dump_covariance.
    const bool iw_in_smoother = P.init_continuous_window;
    for (size_t k = 0; k < aligned.sfm.frames.size(); ++k) {
        Eigen::Matrix4d refined;
        const Eigen::Matrix4d T = (iw_in_smoother && backend_->get_pose_at((int)k, refined))
                                      ? refined : aligned.T_wb[k];
        log_.kf_pose(aligned.sfm.frames[k].t, T);
        if (iw_in_smoother) {
            push_cov((int)k, aligned.sfm.frames[k].t, T, nullptr);
        } else if (P.dump_covariance) {
            Eigen::Matrix<double, 6, 6> C;
            C.setConstant(std::numeric_limits<double>::quiet_NaN());
            log_.kf_cov(aligned.sfm.frames[k].t, T, C);
        }
    }
    prev_t_s_ = aligned.sfm.frames.back().t;
    kf_hist_.clear();
    low_feat_since_s_ = -1.0;
    running_ = true;
}

// main.cpp Phase 3, one frame, plus the online health checks.
void VioEngine::run_online(double t_s, const Observation& obs, EngineOutput& out) {
    const VioParams& P = cfg_.P;
    out.state = STATE_OK;

    // Health of the inputs over (prev_t_s_, t_s].
    const int64_t ta = t0_ns_ + (int64_t)(prev_t_s_ * 1e9), tb = t0_ns_ + (int64_t)(t_s * 1e9);
    auto lo = std::upper_bound(data_.imu.begin(), data_.imu.end(), ta,
                               [](int64_t v, const ImuSample& s) { return v < s.t_ns; });
    auto hi = std::upper_bound(data_.imu.begin(), data_.imu.end(), tb,
                               [](int64_t v, const ImuSample& s) { return v < s.t_ns; });
    // Largest IMU gap seen from this interval: from the last sample before it, between its samples,
    // and from its last sample to the frame stamp (so a long outage cannot hide in one interval).
    int64_t last = (lo != data_.imu.begin()) ? std::prev(lo)->t_ns : ta;
    double max_gap = 0.0;
    for (auto it = lo; it != hi; ++it) { max_gap = std::max(max_gap, (it->t_ns - last) * 1e-9); last = it->t_ns; }
    max_gap = std::max(max_gap, (tb - last) * 1e-9);
    if (t_s - prev_t_s_ > cfg_.max_frame_gap_s) {
        out.state = STATE_FAILED; out.error_code |= err::STALLED;
        log_.event(out.t_ns, "camera gap " + std::to_string(t_s - prev_t_s_) + " s");
    }
    if (max_gap > cfg_.max_imu_gap_s) {
        out.state = STATE_FAILED; out.error_code |= (lo == hi ? err::IMU_MISSING : err::DROPPED_IMU);
        log_.event(out.t_ns, "IMU gap " + std::to_string(max_gap) + " s inside a frame interval");
    }
    if (out.state == STATE_FAILED) { prev_t_s_ = t_s; return; }

    try {
        integrate_imu_batch(*backend_, data_, prev_t_s_, t_s, t0_ns_);
        const bool stationary = detect_stationary(data_, prev_t_s_, t_s, t0_ns_, prev_obs_, obs,
                                                  P.zupt_acc_var, P.zupt_gyr_var, P.zupt_disp_px);
        const double fx = data_.cam_calib.K(0, 0), fy = data_.cam_calib.K(1, 1);
        const double cx = data_.cam_calib.K(0, 2), cy = data_.cam_calib.K(1, 2);
        std::unordered_map<int64_t, Eigen::Vector2d> obs_dict;
        obs_dict.reserve(obs.ids.size());
        std::unordered_map<int64_t, double> obs_q;
        const bool use_fb = (P.pt_hetero_metric == "fb");
        const bool have_q = use_fb ? (obs.fb_res.size() == obs.ids.size())
                                   : (obs.track_cnt.size() == obs.ids.size());
        for (size_t j = 0; j < obs.ids.size(); ++j) {
            double xn = obs.pts_norm(j, 0);
            double yn = obs.pts_norm(j, 1);
            obs_dict[obs.ids[j]] = Eigen::Vector2d(fx * xn + cx, fy * yn + cy);
            if (have_q) obs_q[obs.ids[j]] = use_fb ? obs.fb_res[j]
                                                   : static_cast<double>(obs.track_cnt[j]);
        }
        backend_->set_obs_quality(std::move(obs_q));

        const auto t_bk = Clock::now();
        const bool is_kf = backend_->add_frame(obs_dict, t_s, stationary);
        if (is_kf) {
            out.kf_update_ms = ms_since(t_bk);
            out.is_keyframe = true;
            const Eigen::Matrix4d Twb = backend_->get_current_pose();
            log_.kf_pose(t_s, Twb);
            push_cov(backend_->get_state_idx(), t_s, Twb, &out);
            if (P.dump_relcov) push_relcov(backend_->get_state_idx(), t_s);
        }
        const auto nav = backend_->get_nav_out();
        if (!nav.ok) throw std::runtime_error("no navigation state");
        out.T_wb = nav.T_now; out.v_w = nav.v_now; out.bg = nav.bg; out.ba = nav.ba;
        out.w_body -= nav.bg;
    } catch (const std::exception& e) {
        out.state = STATE_FAILED; out.error_code |= err::COVARIANCE;
        log_.event(out.t_ns, std::string("estimator exception: ") + e.what());
        prev_t_s_ = t_s;
        return;
    }
    prev_t_s_ = t_s;

    if (!out.T_wb.allFinite() || !out.v_w.allFinite()) {
        out.state = STATE_FAILED; out.error_code |= err::COVARIANCE;
        log_.event(out.t_ns, "non-finite state");
    } else if (out.v_w.norm() > cfg_.max_velocity_mps) {
        out.state = STATE_FAILED; out.error_code |= err::VEL_INST_CERT;
        log_.event(out.t_ns, "velocity " + std::to_string(out.v_w.norm()) + " m/s");
    }
    if ((int)obs.ids.size() < cfg_.min_features) {
        if (low_feat_since_s_ < 0) low_feat_since_s_ = t_s;
        if (t_s - low_feat_since_s_ > cfg_.min_features_timeout_s) {
            out.state = STATE_FAILED; out.error_code |= err::NO_FEATURES;
            log_.event(out.t_ns, "too few features for " + std::to_string(t_s - low_feat_since_s_) + " s");
        }
    } else {
        low_feat_since_s_ = -1.0;
    }
}

void VioEngine::push_cov(int idx, double t_s, const Eigen::Matrix4d& T, EngineOutput* out) {
    if (!cfg_.P.dump_covariance) return;
    Eigen::Matrix<double, 6, 6> C;
    if (!backend_->get_pose_covariance(idx, C)) C.setConstant(std::numeric_limits<double>::quiet_NaN());
    log_.kf_cov(t_s, T, C);
    if (out && C.allFinite()) { out->has_cov = true; out->cov = C; }
}

// main.cpp push_relcov.
void VioEngine::push_relcov(int ib, double tb) {
    kf_hist_.emplace_back(ib, tb);
    for (int span : {1, 8}) {
        if ((int)kf_hist_.size() <= span) continue;
        const auto& a = kf_hist_[kf_hist_.size() - 1 - span];
        Eigen::Matrix4d Ta, Tb;
        Eigen::Matrix<double, 6, 6> S;
        if (!backend_->get_pose_at(a.first, Ta) || !backend_->get_pose_at(ib, Tb)
            || !backend_->get_relative_pose_covariance(a.first, ib, S)) continue;
        log_.relcov(a.second, tb, span, Ta.inverse() * Tb, S);
    }
}

void VioEngine::reset(bool hard, int64_t t_ns, const std::string& why) {
    backend_.reset();
    buf_.reset();
    running_ = false;
    latched_fail_ = false;
    kf_hist_.clear();
    low_feat_since_s_ = -1.0;
    ++session_;
    session_hard_ = hard;
    log_.event(t_ns, why + (hard ? " (hard)" : " (soft)") + ", session " + std::to_string(session_));
    log_.start_session(session_, cfg_.P.dump_covariance, cfg_.P.dump_relcov);
    std::lock_guard<std::mutex> lk(stats_m_);
    ++stats_.resets;
}

// Keep the IMU the next steps can still read: the init buffer's span, or the last frame interval.
void VioEngine::trim_imu() {
    if (data_.imu.size() < 20000) return;
    double keep_s = prev_t_s_;
    if (!running_) keep_s = buf_.frames.empty() ? (data_.imu.back().t_ns - t0_ns_) * 1e-9 : buf_.frames.front().t;
    const int64_t keep_ns = t0_ns_ + (int64_t)((keep_s - 2.0) * 1e9);
    auto it = std::lower_bound(data_.imu.begin(), data_.imu.end(), keep_ns,
                               [](const ImuSample& s, int64_t v) { return s.t_ns < v; });
    if (it - data_.imu.begin() > 5000) data_.imu.erase(data_.imu.begin(), it);
}

// ------------------------------------------------------------------ shutdown
void VioEngine::imu_finished() {
    {
        std::lock_guard<std::mutex> lk(m_);
        imu_done_ = true;
    }
    cv_.notify_all();
}

void VioEngine::finish() {
    {
        std::lock_guard<std::mutex> lk(m_);
        input_done_ = true;
    }
    cv_.notify_all();
    if (sync_th_.joinable()) sync_th_.join();
    if (frontend_) frontend_->join();
    if (est_th_.joinable()) est_th_.join();
}

void VioEngine::stop() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    img_q_.close();
    fe_imu_q_.close();
    if (frontend_) frontend_->request_stop();
    obs_q_.close();
    if (sync_th_.joinable()) sync_th_.join();
    if (frontend_) frontend_->join();
    if (est_th_.joinable()) est_th_.join();
}

VioEngine::Stats VioEngine::stats() const {
    Stats s;
    {
        std::lock_guard<std::mutex> lk(stats_m_);
        s = stats_;
    }
    std::lock_guard<std::mutex> lk(imu_filter_m_);
    s.imu_rate_hz = imu_filter_.rate_hz();
    s.imu_tau_ms = 1e3 * imu_filter_.tau_s();
    s.imu_gaps = imu_filter_.n_gaps();
    return s;
}

} // namespace gvio
