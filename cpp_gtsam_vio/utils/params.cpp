#include "params.h"

#include <yaml-cpp/yaml.h>
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
    rd(rt, "deterministic",   p.deterministic);
    rd(rt, "smoother_lag_s", p.smoother_lag_s);
    rd(rt, "loop_closure",   p.loop_closure);
    rd(rt, "frontend_source", p.frontend_source);
    rd(rt, "frontend_dir",    p.frontend_dir);
    rd(rt, "frontend_root",   p.frontend_root);
    rd(rt, "backend_mode",   p.backend_mode);
    rd(rt, "point_backend",  p.point_backend);
    rd(rt, "isam_relin_threshold", p.isam_relin_threshold);
    rd(rt, "isam_relin_skip",      p.isam_relin_skip);
    rd(rt, "bias_accel_max",       p.bias_accel_max);
    rd(rt, "bias_gyro_max",        p.bias_gyro_max);

    auto pf = root["point_frontend"];
    rd(pf, "max_points",    p.pt_max);
    rd(pf, "min_dist_px",   p.pt_min_dist_px);
    rd(pf, "fb_err",        p.pt_fb_err);
    rd(pf, "ransac_thresh", p.pt_ransac_th);
    rd(pf, "quality",       p.pt_quality);
    rd(pf, "imu_aided_klt", p.pt_imu_aided_klt);

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
    rd(in, "continuous_window", p.init_continuous_window);
    rd(in, "iw_pose0_sigma",    p.iw_pose0_sigma);
    rd(in, "iw_vel0_sigma",     p.iw_vel0_sigma);
    rd(in, "iw_bias0_acc_sigma", p.iw_bias0_acc_sigma);
    rd(in, "iw_bias0_gyr_sigma", p.iw_bias0_gyr_sigma);

    auto vp = root["point_factor"];
    rd(vp, "huber_k",  p.pt_huber_k);
    rd(vp, "sigma_px", p.pt_sigma_px);

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
    rd(lc, "cauchy_c",         p.lc_cauchy_c);
    rd(lc, "inlier_weighting", p.lc_inlier_weighting);
    rd(lc, "min_correction_m", p.lc_min_correction_m);
    rd(lc, "loop_drift_rate",  p.lc_loop_drift_rate);

    auto zu = root["zupt"];
    rd(zu, "acc_var",   p.zupt_acc_var);
    rd(zu, "gyr_var",   p.zupt_gyr_var);
    rd(zu, "disp_px",   p.zupt_disp_px);
    rd(zu, "vel_sigma", p.zupt_vel_sigma);

    std::cout << "[params] loaded '" << path << "'\n";
    return p;
}

} // namespace vio
