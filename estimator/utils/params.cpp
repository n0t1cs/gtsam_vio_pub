#include "params.h"

#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace vio {

// Read sec[key] into out if present; leave out (the default) otherwise.
template <typename T>
static void rd(const YAML::Node& sec, const char* key, T& out) {
    if (sec && sec[key]) {
        try { out = sec[key].as<T>(); } catch (const std::exception&) {}
    }
}

// A removed switch: accepted only at the value the system ran with. Any other value (or one that
// does not parse) is recorded in `why`, the first one wins.
template <typename T>
static void removed(const YAML::Node& sec, const char* name, const char* key, const T& kept, std::string& why) {
    if (!why.empty() || !sec || !sec[key]) return;
    T v{};
    bool ok = true;
    try { v = sec[key].as<T>(); } catch (const std::exception&) { ok = false; }
    if (ok && v == kept) return;
    std::ostringstream s;
    s << std::boolalpha << name << "." << key << " was removed, only " << kept << " is accepted";
    why = s.str();
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

    const YAML::Node rt = root["runtime"];
    rd(rt, "dump_covariance", p.dump_covariance);
    rd(rt, "dump_relcov",     p.dump_relcov);
    rd(rt, "deterministic",   p.deterministic);
    rd(rt, "rng_seed",        p.rng_seed);
    rd(rt, "kf_window_size", p.kf_window_size);
    rd(rt, "kf_use_fej",     p.kf_use_fej);
    rd(rt, "kf_fej_scope",   p.kf_fej_scope);
    rd(rt, "kf_max_iterations", p.kf_max_iterations);
    rd(rt, "backend_mode",   p.backend_mode);
    rd(rt, "isam_relin_threshold", p.isam_relin_threshold);
    rd(rt, "isam_relin_skip",      p.isam_relin_skip);
    rd(rt, "bias_accel_max",       p.bias_accel_max);
    rd(rt, "bias_gyro_max",        p.bias_gyro_max);

    const YAML::Node pf = root["point_frontend"];
    rd(pf, "max_points",    p.pt_max);
    rd(pf, "min_dist_px",   p.pt_min_dist_px);
    rd(pf, "fb_err",        p.pt_fb_err);
    rd(pf, "ransac_thresh", p.pt_ransac_th);
    rd(pf, "ransac_conf",   p.pt_ransac_conf);
    rd(pf, "ransac_iters",  p.pt_ransac_iters);
    rd(pf, "quality",       p.pt_quality);
    rd(pf, "histogram_eq",  p.pt_histeq);
    rd(pf, "clahe_clip",    p.pt_clahe_clip);

    const YAML::Node im = root["imu_noise"];
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

    const YAML::Node in = root["init"];
    rd(in, "window",            p.init_window);
    rd(in, "parallax_px",       p.init_parallax_px);
    rd(in, "min_common",        p.init_min_common);
    rd(in, "excitation_var",    p.init_excitation_var);
    rd(in, "max_pre_ba_rms_px", p.init_max_pre_ba_rms_px);
    rd(in, "max_vel_mps",       p.init_max_vel_mps);
    rd(in, "iw_pose0_sigma",    p.iw_pose0_sigma);
    rd(in, "iw_vel0_sigma",     p.iw_vel0_sigma);
    rd(in, "iw_bias0_acc_sigma", p.iw_bias0_acc_sigma);
    rd(in, "iw_bias0_gyr_sigma", p.iw_bias0_gyr_sigma);

    const YAML::Node vp = root["point_factor"];
    rd(vp, "huber_k",  p.pt_huber_k);
    rd(vp, "sigma_px", p.pt_sigma_px);

    const YAML::Node kf = root["keyframe"];
    rd(kf, "parallax_px",  p.kf_parallax_px);
    rd(kf, "min_tracked",  p.kf_min_tracked);
    rd(kf, "max_interval", p.kf_max_interval);

    const YAML::Node lm = root["landmark"];
    rd(lm, "min_baseline_m",    p.lm_min_baseline_m);
    rd(lm, "tri_depth_min",     p.lm_tri_depth_min);
    rd(lm, "tri_depth_max",     p.lm_tri_depth_max);
    rd(lm, "depth_min",         p.lm_depth_min);
    rd(lm, "depth_max",         p.lm_depth_max);
    rd(lm, "floor_prior_sigma", p.lm_floor_prior_sigma);
    rd(lm, "point_obs_reject_px", p.pt_obs_reject_px);
    rd(lm, "reanchor",          p.lm_reanchor);
    rd(lm, "reanchor_sigma",    p.lm_reanchor_sigma);
    rd(lm, "reanchor_retire",   p.lm_reanchor_retire);
    rd(lm, "reanchor_live_max", p.lm_reanchor_live_max);

    const YAML::Node zu = root["zupt"];
    rd(zu, "acc_var",   p.zupt_acc_var);
    rd(zu, "gyr_var",   p.zupt_gyr_var);
    rd(zu, "disp_px",   p.zupt_disp_px);
    rd(zu, "vel_sigma", p.zupt_vel_sigma);

    // Removed switches (2026-11-03: the online system keeps what its configs use). Their blocks
    // (smart_factor, loop, the hetero_* and smart keys) are otherwise ignored like any unknown key.
    // The batch backend and the time window went on 2026-10-08.
    std::string why;
    removed(rt, "runtime", "dump_frontend", false, why);
    removed(rt, "runtime", "nis_prefit", false, why);
    removed(rt, "runtime", "loop_closure", false, why);
    removed(rt, "runtime", "frontend_source", std::string("images"), why);
    removed(rt, "runtime", "point_backend", std::string("invdepth"), why);
    removed(rt, "runtime", "isam_factorization", std::string("cholesky"), why);
    removed(rt, "runtime", "kf_optimizer", std::string("gn"), why);
    removed(rt, "runtime", "kf_marg_elim", std::string("cholesky"), why);
    removed(rt, "runtime", "kf_diag_cond", false, why);
    removed(rt, "runtime", "window_coord", std::string("keyframe"), why);
    if (why.empty() && rt && rt["smoother_lag_s"])
        why = "runtime.smoother_lag_s was removed (every window is keyframe-indexed)";
    removed(pf, "point_frontend", "ransac_method", std::string("ransac"), why);
    removed(pf, "point_frontend", "imu_aided_klt", true, why);
    removed(pf, "point_frontend", "grid_x", 1, why);
    removed(pf, "point_frontend", "grid_y", 1, why);
    removed(in, "init", "est_accel_bias", false, why);
    removed(in, "init", "oracle_gravity", false, why);
    removed(in, "init", "continuous_window", true, why);
    removed(vp, "point_factor", "hetero_cov", false, why);
    removed(kf, "keyframe", "shift_weight_t", 0.0, why);
    removed(kf, "keyframe", "shift_weight_r", 0.0, why);
    removed(kf, "keyframe", "shift_weight_rt", 1.0, why);
    removed(lm, "landmark", "min_parallax_deg", 0.0, why);
    removed(lm, "landmark", "point_cull_depth_min", 0.0, why);
    removed(lm, "landmark", "point_cull_depth_max", 0.0, why);
    removed(lm, "landmark", "point_depth_transfer", false, why);
    removed(lm, "landmark", "init_current_obs", false, why);
    removed(lm, "landmark", "reanchor_max", 0, why);
    if (why.empty() && root["extrinsics"])
        why = "the extrinsics block was removed (T_BC comes from the vehicle's extrinsics.conf)";

    // Values a typo would otherwise turn into something else.
    if (why.empty()) {
        if (p.backend_mode != "incremental" && p.backend_mode != "keyframe")
            why = "runtime.backend_mode: " + p.backend_mode + " is not incremental or keyframe";
        else if (p.kf_fej_scope != "xv" && p.kf_fej_scope != "xvb" && p.kf_fej_scope != "all")
            why = "runtime.kf_fej_scope: " + p.kf_fej_scope + " is not xv, xvb or all";
        else if (p.pt_histeq != "none" && p.pt_histeq != "equalize" && p.pt_histeq != "clahe")
            why = "point_frontend.histogram_eq: " + p.pt_histeq + " is not none, equalize or clahe";
        else if (p.lm_reanchor != "abandon" && p.lm_reanchor != "exact")
            why = "landmark.reanchor: " + p.lm_reanchor + " is not abandon or exact";
        else if (p.lm_reanchor == "exact" && !(p.lm_reanchor_sigma > 0.0))
            why = "landmark.reanchor_sigma must be positive";
        else if (p.lm_reanchor_live_max < 0)
            why = "landmark.reanchor_live_max must be 0 (unlimited) or positive";
        else if ((p.lm_reanchor_live_max > 0 || p.lm_reanchor_retire) && p.lm_reanchor != "exact")
            why = "landmark.reanchor_live_max and landmark.reanchor_retire need landmark.reanchor: exact";
        else if (p.rng_seed < 0)
            why = "runtime.rng_seed must be 0 or positive";
        else if (!(p.pt_ransac_conf > 0.0 && p.pt_ransac_conf < 1.0) || p.pt_ransac_iters < 1)
            why = "point_frontend.ransac_conf must lie in (0, 1) and ransac_iters be positive";
    }
    if (!why.empty()) {
        std::cerr << "[params] FATAL in '" << path << "': " << why << ".\n";
        std::exit(2);
    }

    std::cout << "[params] loaded '" << path << "'\n";
    return p;
}

} // namespace vio
