#include "params.h"

#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <iostream>

namespace vio {

// Read sec[key] into out if present; leave out (the default) otherwise.
template <typename T>
static void rd(const YAML::Node& sec, const char* key, T& out) {
    if (sec && sec[key]) {
        try { out = sec[key].as<T>(); } catch (const std::exception&) {}
    }
}

VioParams VioParams::load(const std::string& path) {
    VioParams p;
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        std::cout << "[params] could not load '" << path << "' (" << e.what()
                  << "); using built-in defaults.\n";
        return p;
    }

    auto rt = root["runtime"];
    rd(rt, "dump_frontend",  p.dump_frontend);
    rd(rt, "diag",           p.diag);
    rd(rt, "dump_covariance", p.dump_covariance);
    rd(rt, "dump_relcov",     p.dump_relcov);
    rd(rt, "deterministic",   p.deterministic);
    rd(rt, "rng_seed",        p.rng_seed);
    rd(rt, "kf_window_size", p.kf_window_size);
    rd(rt, "kf_use_fej",     p.kf_use_fej);
    rd(rt, "kf_fej_scope",   p.kf_fej_scope);
    rd(rt, "kf_max_iterations", p.kf_max_iterations);
    rd(rt, "kf_optimizer",   p.kf_optimizer);
    rd(rt, "kf_marg_elim",   p.kf_marg_elim);
    rd(rt, "kf_diag_cond",   p.kf_diag_cond);
    rd(rt, "loop_closure",   p.loop_closure);
    rd(rt, "frontend_source", p.frontend_source);
    rd(rt, "frontend_dir",    p.frontend_dir);
    rd(rt, "frontend_root",   p.frontend_root);
    rd(rt, "backend_mode",   p.backend_mode);
    rd(rt, "point_backend",  p.point_backend);
    rd(rt, "isam_relin_threshold", p.isam_relin_threshold);
    rd(rt, "isam_relin_skip",      p.isam_relin_skip);
    rd(rt, "isam_factorization",   p.isam_factorization);
    rd(rt, "bias_accel_max",       p.bias_accel_max);
    rd(rt, "bias_gyro_max",        p.bias_gyro_max);

    auto pf = root["point_frontend"];
    rd(pf, "max_points",    p.pt_max);
    rd(pf, "min_dist_px",   p.pt_min_dist_px);
    rd(pf, "fb_err",        p.pt_fb_err);
    rd(pf, "ransac_thresh", p.pt_ransac_th);
    rd(pf, "ransac_method", p.pt_ransac_method);
    rd(pf, "ransac_conf",   p.pt_ransac_conf);
    rd(pf, "ransac_iters",  p.pt_ransac_iters);
    rd(pf, "quality",       p.pt_quality);
    rd(pf, "imu_aided_klt", p.pt_imu_aided_klt);
    rd(pf, "grid_x",        p.pt_grid_x);
    rd(pf, "grid_y",        p.pt_grid_y);
    rd(pf, "histogram_eq",  p.pt_histeq);
    rd(pf, "clahe_clip",    p.pt_clahe_clip);

    auto im = root["imu_noise"];
    rd(im, "acc_n",              p.imu_acc_n);
    rd(im, "gyr_n",              p.imu_gyr_n);
    rd(im, "acc_w",              p.imu_acc_w);
    rd(im, "gyr_w",              p.imu_gyr_w);
    rd(im, "acc_cov_mult",       p.imu_acc_cov_mult);
    rd(im, "gyr_cov_mult",       p.imu_gyr_cov_mult);
    rd(im, "acc_bias_cov_mult",  p.imu_acc_bias_cov_mult);
    rd(im, "gyr_bias_cov_mult",  p.imu_gyr_bias_cov_mult);
    rd(im, "integration_cov",    p.imu_integration_cov);
    rd(im, "bias_acc_omega_int", p.imu_bias_acc_omega_int);

    auto in = root["init"];
    rd(in, "window",            p.init_window);
    rd(in, "parallax_px",       p.init_parallax_px);
    rd(in, "min_common",        p.init_min_common);
    rd(in, "excitation_var",    p.init_excitation_var);
    rd(in, "max_pre_ba_rms_px", p.init_max_pre_ba_rms_px);
    rd(in, "max_vel_mps",       p.init_max_vel_mps);
    rd(in, "est_accel_bias",    p.init_est_ba);
    rd(in, "oracle_gravity",    p.init_oracle_gravity);
    rd(in, "continuous_window", p.init_continuous_window);
    rd(in, "iw_pose0_sigma",    p.iw_pose0_sigma);
    rd(in, "iw_vel0_sigma",     p.iw_vel0_sigma);
    rd(in, "iw_bias0_acc_sigma", p.iw_bias0_acc_sigma);
    rd(in, "iw_bias0_gyr_sigma", p.iw_bias0_gyr_sigma);

    auto vp = root["point_factor"];
    rd(vp, "huber_k",  p.pt_huber_k);
    rd(vp, "sigma_px", p.pt_sigma_px);
    rd(vp, "hetero_cov",       p.pt_hetero_cov);
    rd(vp, "hetero_metric",    p.pt_hetero_metric);
    rd(vp, "hetero_age_full",  p.pt_hetero_age_full);
    rd(vp, "hetero_fb_ref",    p.pt_hetero_fb_ref);
    rd(vp, "hetero_parallax_ref", p.pt_hetero_parallax_ref);
    rd(vp, "hetero_sigma_max", p.pt_hetero_sigma_max);

    auto sf = root["smart_factor"];
    rd(sf, "sigma_px",      p.smart_sigma_px);
    rd(sf, "rank_tol",      p.smart_rank_tol);
    rd(sf, "landmark_dist", p.smart_landmark_dist);
    rd(sf, "retriang_th",   p.smart_retriang_th);
    rd(sf, "outlier_px",    p.smart_outlier_px);

    auto kf = root["keyframe"];
    rd(kf, "parallax_px",  p.kf_parallax_px);
    rd(kf, "min_tracked",  p.kf_min_tracked);
    rd(kf, "max_interval", p.kf_max_interval);
    rd(kf, "shift_weight_t",  p.kf_shift_weight_t);
    rd(kf, "shift_weight_r",  p.kf_shift_weight_r);
    rd(kf, "shift_weight_rt", p.kf_shift_weight_rt);

    auto lm = root["landmark"];
    rd(lm, "min_baseline_m",    p.lm_min_baseline_m);
    rd(lm, "min_parallax_deg",  p.lm_min_parallax_deg);
    rd(lm, "tri_depth_min",     p.lm_tri_depth_min);
    rd(lm, "tri_depth_max",     p.lm_tri_depth_max);
    rd(lm, "depth_min",         p.lm_depth_min);
    rd(lm, "depth_max",         p.lm_depth_max);
    rd(lm, "floor_prior_sigma", p.lm_floor_prior_sigma);
    rd(lm, "world_prior_sigma_m", p.lm_world_prior_sigma_m);
    rd(lm, "point_obs_reject_px",      p.pt_obs_reject_px);
    rd(lm, "point_cull_depth_min",     p.pt_cull_depth_min);
    rd(lm, "point_cull_depth_max",     p.pt_cull_depth_max);
    rd(lm, "point_depth_transfer",     p.pt_depth_transfer);
    rd(lm, "point_depth_transfer_oldest", p.pt_depth_transfer_oldest);
    rd(lm, "init_current_obs",  p.lm_init_current_obs);
    rd(lm, "reanchor",          p.lm_reanchor);
    rd(lm, "reanchor_sigma",    p.lm_reanchor_sigma);
    rd(lm, "reanchor_max",      p.lm_reanchor_max);
    rd(lm, "reanchor_retire",   p.lm_reanchor_retire);
    rd(lm, "reanchor_live_max", p.lm_reanchor_live_max);

    auto lc = root["loop"];
    rd(lc, "skip_recent",      p.lc_skip_recent);
    rd(lc, "fast_threshold",   p.lc_fast_threshold);
    rd(lc, "min_inliers",      p.lc_min_inliers);
    rd(lc, "hamming_th",       p.lc_hamming_th);
    rd(lc, "score_best",       p.lc_score_best);
    rd(lc, "score_cand",       p.lc_score_cand);
    rd(lc, "fund_thresh_px",   p.lc_fund_thresh_px);
    rd(lc, "max_trans_m",      p.lc_max_trans_m);
    rd(lc, "max_yaw_deg",      p.lc_max_yaw_deg);
    rd(lc, "backbone_hops",    p.lc_backbone_hops);
    rd(lc, "loop_trans_sigma", p.lc_loop_trans_sigma);
    rd(lc, "loop_rot_sigma",   p.lc_loop_rot_sigma);
    rd(lc, "odo_trans_sigma",  p.lc_odo_trans_sigma);
    rd(lc, "odo_rot_sigma",    p.lc_odo_rot_sigma);
    rd(lc, "odo_from_estimator",  p.lc_odo_from_estimator);
    rd(lc, "odo_sigma_inflation", p.lc_odo_sigma_inflation);
    rd(lc, "odo_sigma_floor_m",   p.lc_odo_sigma_floor_m);
    rd(lc, "odo_sigma_floor_rad", p.lc_odo_sigma_floor_rad);
    rd(lc, "cauchy_c",         p.lc_cauchy_c);
    rd(lc, "inlier_weighting", p.lc_inlier_weighting);
    rd(lc, "min_correction_m", p.lc_min_correction_m);
    rd(lc, "loop_drift_rate",  p.lc_loop_drift_rate);

    // Extrinsics override. Accepts either a bare 16-element list or a EuRoC-style
    // {rows, cols, data} block, so a sensor.yaml T_BS can be pasted in verbatim.
    auto ext = root["extrinsics"];
    if (ext) {
        auto rd_T = [&](const char* key, std::vector<double>& out) {
            if (!ext[key]) return;
            const YAML::Node& n = ext[key];
            try {
                out = (n.IsMap() && n["data"]) ? n["data"].as<std::vector<double>>()
                                               : n.as<std::vector<double>>();
            } catch (const std::exception&) { return; }
            if (out.size() != 16) {
                std::cout << "[params] extrinsics." << key << " has " << out.size()
                          << " entries, expected 16 (row-major 4x4); IGNORED.\n";
                out.clear();
            }
        };
        rd_T("T_BC", p.ext_T_BC);
        rd_T("T_BI", p.ext_T_BI);
    }

    auto zu = root["zupt"];
    rd(zu, "acc_var",   p.zupt_acc_var);
    rd(zu, "gyr_var",   p.zupt_gyr_var);
    rd(zu, "disp_px",   p.zupt_disp_px);
    rd(zu, "vel_sigma", p.zupt_vel_sigma);

    // The batch backend and the time window were removed on 2026-10-08. Unknown keys
    // are otherwise ignored, so a config that relied on either would silently run as
    // something else. Reject it instead. Accepted on purpose: window_coord: keyframe
    // (now the only window), and smoother_lag_s in a config that never used it (the
    // SLKS ignored it, and so did iSAM2 under window_coord: keyframe).
    {
        const YAML::Node& crt = rt;   // const access: never inserts keys
        std::string wc;
        rd(crt, "window_coord", wc);
        const bool has_wc  = crt && crt["window_coord"];
        const bool has_lag = crt && crt["smoother_lag_s"];
        std::string why;
        if (p.backend_mode == "batch")
            why = "backend_mode: batch (BatchFixedLagSmoother) was removed";
        else if (has_wc && wc != "keyframe")
            why = "window_coord: " + wc + " was removed (every window is keyframe-indexed)";
        else if (p.backend_mode == "incremental" && has_lag && !has_wc)
            why = "this incremental config sets smoother_lag_s without window_coord: keyframe, "
                  "which meant the removed time window";
        if (!why.empty()) {
            std::cerr << "[params] FATAL in '" << path << "': " << why
                      << ". The option is not part of this release.\n";
            std::exit(2);
        }
    }
    // Landmark continuity: a typo or an unsupported combination would otherwise run
    // silently as the committed system.
    {
        std::string why;
        const bool cont = p.lm_reanchor == "exact" || p.lm_init_current_obs;
        if (p.lm_reanchor != "abandon" && p.lm_reanchor != "exact")
            why = "landmark.reanchor: " + p.lm_reanchor + " is not abandon or exact";
        else if (cont && p.point_backend != "invdepth")
            why = "landmark.reanchor: exact and landmark.init_current_obs need point_backend: invdepth";
        else if (cont && p.pt_depth_transfer)
            why = "landmark.reanchor: exact and landmark.init_current_obs do not combine with "
                  "point_depth_transfer, which double counts";
        else if (p.lm_reanchor == "exact" && !(p.lm_reanchor_sigma > 0.0))
            why = "landmark.reanchor_sigma must be positive";
        else if (p.lm_reanchor_max < 0)
            why = "landmark.reanchor_max must be 0 (unlimited) or positive";
        else if (p.lm_reanchor_max > 0 && p.lm_reanchor != "exact")
            why = "landmark.reanchor_max needs landmark.reanchor: exact";
        else if (p.lm_reanchor_live_max < 0)
            why = "landmark.reanchor_live_max must be 0 (unlimited) or positive";
        else if ((p.lm_reanchor_live_max > 0 || p.lm_reanchor_retire) && p.lm_reanchor != "exact")
            why = "landmark.reanchor_live_max and landmark.reanchor_retire need landmark.reanchor: exact";
        else if (p.rng_seed < 0)
            why = "runtime.rng_seed must be 0 or positive";
        else if (p.pt_ransac_method != "ransac" && p.pt_ransac_method != "magsac" && p.pt_ransac_method != "twopt")
            why = "point_frontend.ransac_method: " + p.pt_ransac_method + " is not ransac, magsac or twopt";
        else if (!(p.pt_ransac_conf > 0.0 && p.pt_ransac_conf < 1.0) || p.pt_ransac_iters < 1)
            why = "point_frontend.ransac_conf must lie in (0, 1) and ransac_iters be positive";
        else if (p.lm_min_parallax_deg < 0.0)
            why = "landmark.min_parallax_deg must be 0 (off) or positive";
        if (!why.empty()) {
            std::cerr << "[params] FATAL in '" << path << "': " << why << ".\n";
            std::exit(2);
        }
    }

    std::cout << "[params] loaded '" << path << "'\n";
    return p;
}

} // namespace vio
