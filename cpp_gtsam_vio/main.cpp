// main.cpp - VIO pipeline driver. Output is a CSV of timestamps + estimated
// and aligned ground-truth positions (vio_results.csv). Plot externally.

#include "load.h"
#include "frontend.h"
#include "frame_queue.h"      // ThreadSafeQueue + StampedImage/StampedObservation
#include "frontend_thread.h"  // FrontendThread: KLT + IMU-aided prediction on its own thread
#include "init_buffer.h"
//#include "imu_excitation.h"
#include "init_sfm.h"
#include "bundle_adjust.h"
#include "init_align.h"
#include "vins_realign.h"
#include "vio_backend.h"
#include "params.h"
#include "brief_extractor.h"
#include "loop_keyframe.h"
#include "loop_database.h"
#include "pose_graph.h"

#include <gtsam/navigation/PreintegrationParams.h>
#include <tbb/global_control.h>   // pin TBB threads for bit-reproducible runs (runtime.deterministic)
#include <chrono>                 // backend-update timing ([timing] speed metric)
#include <gtsam/navigation/CombinedImuFactor.h>
//#include <gtsam/navigation/PreintegrationCombinedParams.h> // Add this

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <Eigen/Geometry>
#include <unordered_map>
#include <map>
#include <sstream>

namespace fs = std::filesystem;
using namespace vio;


// ---------- IMU batch -> backend ----------
static void integrate_imu_batch(VioBackend& backend, const EurocData& data,
                                double t0_s, double t1_s, int64_t t0_ns_base) {
    int64_t t0_ns = t0_ns_base + (int64_t)(t0_s * 1e9);
    int64_t t1_ns = t0_ns_base + (int64_t)(t1_s * 1e9);

    auto lo = std::lower_bound(data.imu.begin(), data.imu.end(), t0_ns,
                               [](const ImuSample& s, int64_t v){ return s.t_ns < v; });
    auto hi = std::upper_bound(data.imu.begin(), data.imu.end(), t1_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });

    auto it_prev = lo;
    if (it_prev == hi) return;
    for (auto it = it_prev + 1; it != hi; ++it) {
        double dt = (it->t_ns - it_prev->t_ns) * 1e-9;
        if (dt > 0) backend.add_imu(dt, it->a, it->w);
        it_prev = it;
    }
}

// ---------- SE(3) trajectory alignment ----------
static std::vector<Eigen::Vector3d> align_se3(
        const std::vector<Eigen::Vector3d>& est, const std::vector<Eigen::Vector3d>& gt) {
    int N = (int)est.size();
    Eigen::Vector3d mu_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_gt = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) { mu_est += est[i]; mu_gt += gt[i]; }
    mu_est /= (double)N;
    mu_gt /= (double)N;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
        H += (est[i] - mu_est) * (gt[i] - mu_gt).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }
    Eigen::Vector3d t = mu_gt - R * mu_est;

    std::vector<Eigen::Vector3d> out(N);
    for (int i = 0; i < N; ++i) out[i] = R * est[i] + t;
    return out;
}

static void print_ate(const char* tag, const std::vector<Eigen::Vector3d>& aligned,
                      const std::vector<Eigen::Vector3d>& gt_pos) {
    Eigen::Vector3d rmse = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < aligned.size(); ++i) {
        Eigen::Vector3d e = aligned[i] - gt_pos[i];
        rmse += e.cwiseProduct(e);
    }
    rmse = (rmse / (double)aligned.size()).cwiseSqrt();
    std::cout << "\n========================================\n";
    std::cout << "      TRAJECTORY METRICS (ATE) " << tag << "\n";
    std::cout << "========================================\n";
    std::cout << "RMSE X:     " << rmse(0) << " m\n";
    std::cout << "RMSE Y:     " << rmse(1) << " m\n";
    std::cout << "RMSE Z:     " << rmse(2) << " m\n";
    std::cout << "Total RMSE: " << rmse.norm() << " m\n";
    std::cout << "========================================\n";
}

static void evaluate_and_write_csv(const std::vector<double>& ts,
                                   const std::vector<Eigen::Matrix4d>& poses,
                                   const EurocData& data, int64_t t0_ns,
                                   const std::string& out_dir,
                                   const std::string& csv_name,
                                   const std::vector<Eigen::Matrix4d>& lc_poses = {}) {
    const auto& gt = data.gt;
    std::vector<double> gt_t_s(gt.size());
    for (size_t i = 0; i < gt.size(); ++i) gt_t_s[i] = (gt[i].t_ns - t0_ns) * 1e-9;

    std::vector<Eigen::Vector3d> gt_pos, est_pos;
    gt_pos.reserve(ts.size());
    est_pos.reserve(ts.size());

    for (size_t k = 0; k < ts.size(); ++k) {
        double t = ts[k];
        auto it = std::lower_bound(gt_t_s.begin(), gt_t_s.end(), t);
        size_t idx = it - gt_t_s.begin();
        if (idx == 0) idx = 1;
        if (idx >= gt_t_s.size()) idx = gt_t_s.size() - 1;
        double dt1 = t - gt_t_s[idx - 1];
        double dt2 = gt_t_s[idx] - t;
        size_t chosen = (dt1 < dt2) ? (idx - 1) : idx;
        gt_pos.push_back(gt[chosen].p);
        est_pos.push_back(poses[k].block<3, 1>(0, 3));
    }

    auto est_aligned = align_se3(est_pos, gt_pos);
    const bool has_lc = (!lc_poses.empty() && lc_poses.size() == ts.size());
    // The pose graph is non-invasive, so est is always the clean no-LC VIO baseline.
    print_ate("(no loop closure)", est_aligned, gt_pos);

    std::vector<Eigen::Vector3d> lc_aligned;
    if (has_lc) {
        std::vector<Eigen::Vector3d> lc_pos;
        lc_pos.reserve(ts.size());
        for (size_t k = 0; k < ts.size(); ++k) lc_pos.push_back(lc_poses[k].block<3, 1>(0, 3));
        lc_aligned = align_se3(lc_pos, gt_pos);
        print_ate("(with loop closure -- pose graph)", lc_aligned, gt_pos);
    }

    std::string filepath = out_dir + "/" + csv_name;
    std::ofstream fout(filepath);
    fout << "t_s,gt_x,gt_y,gt_z,est_x,est_y,est_z" << (has_lc ? ",lc_x,lc_y,lc_z" : "") << "\n";
    for (size_t i = 0; i < ts.size(); ++i) {
        fout << ts[i] << ","
             << gt_pos[i](0) << "," << gt_pos[i](1) << "," << gt_pos[i](2) << ","
             << est_aligned[i](0) << "," << est_aligned[i](1) << "," << est_aligned[i](2);
        if (has_lc)
            fout << "," << lc_aligned[i](0) << "," << lc_aligned[i](1) << "," << lc_aligned[i](2);
        fout << "\n";
    }
    std::cout << "Wrote " << filepath << "\n";

    // ---- evo-compatible TUM trajectory dumps (raw, unaligned, full poses) ----
    // evo associates est<->gt by timestamp and runs its own Umeyama alignment,
    // so dump RAW world-frame poses (NOT est_aligned) on a shared clock (seconds
    // since t0_ns). TUM row: t tx ty tz qx qy qz qw.
    std::string stem = csv_name;
    if (stem.size() > 4 && stem.substr(stem.size() - 4) == ".csv")
        stem = stem.substr(0, stem.size() - 4);

    auto write_tum_pose = [](std::ofstream& os, double t, const Eigen::Matrix4d& T) {
        Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
        q.normalize();
        os << std::fixed << std::setprecision(9)
           << t << " " << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " "
           << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    };

    { std::ofstream os(out_dir + "/" + stem + "_est.tum");
      for (size_t k = 0; k < ts.size(); ++k) write_tum_pose(os, ts[k], poses[k]); }

    { std::ofstream os(out_dir + "/" + stem + "_gt.tum");
      for (const auto& g : gt) {
          Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
          Eigen::Quaterniond q(g.q_wxyz(0), g.q_wxyz(1), g.q_wxyz(2), g.q_wxyz(3));
          T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
          T.block<3, 1>(0, 3) = g.p;
          write_tum_pose(os, (g.t_ns - t0_ns) * 1e-9, T);
      } }

    if (has_lc) {
        std::ofstream os(out_dir + "/" + stem + "_lc.tum");
        for (size_t k = 0; k < ts.size(); ++k) write_tum_pose(os, ts[k], lc_poses[k]);
    }
    std::cout << "Wrote evo TUM: " << out_dir << "/" << stem << "_{est,gt"
              << (has_lc ? ",lc" : "") << "}.tum\n";
}


bool check_imu_excitation(const std::vector<KeyframeEntry>& frames,
                          const EurocData& data,
                          double min_acc_var) {
    if (frames.size() < 2) return false;

    const int64_t t0_ns = data.frames.front().t_ns;
    const auto& imu = data.imu;

    // Use binary search since imu is sorted by t_ns.
    auto find_lo = [&](int64_t t) {
        return std::upper_bound(imu.begin(), imu.end(), t,
                                [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    };
    auto find_hi = [&](int64_t t) {
        // upper_bound on t_b -> all with t_ns <= t_b are before this iterator.
        return std::upper_bound(imu.begin(), imu.end(), t,
                                [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    };

    // Collect all accel samples across all consecutive intervals.
    std::vector<Eigen::Vector3d> all_acc;
    all_acc.reserve(2048);

    for (size_t i = 0; i + 1 < frames.size(); ++i) {
        int64_t t_a = t0_ns + (int64_t)std::llround(frames[i].t * 1e9);
        int64_t t_b = t0_ns + (int64_t)std::llround(frames[i + 1].t * 1e9);

        // Mask: t_ns > t_a AND t_ns <= t_b.
        auto lo = find_lo(t_a);   // first with t_ns > t_a
        auto hi = find_hi(t_b);   // first with t_ns > t_b
        if (std::distance(lo, hi) < 2) return false;

        for (auto it = lo; it != hi; ++it) all_acc.push_back(it->a);
    }

    if (all_acc.size() < 2) return false;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& a : all_acc) mean += a;
    mean /= (double)all_acc.size();

    Eigen::Vector3d var = Eigen::Vector3d::Zero();
    for (const auto& a : all_acc) {
        Eigen::Vector3d d = a - mean;
        var += d.cwiseProduct(d);
    }
    var /= (double)all_acc.size();   // population variance (matches np.var default)

    double total_var = var.sum();
    return total_var > min_acc_var;
}

// Stationary detector: true only when IMU is quiet AND tracked features barely
// move. The vision gate is what rejects constant-velocity motion (low accel
// variance but real translation -> features still move).
static bool detect_stationary(const EurocData& data,
                              double t0_s, double t1_s, int64_t t0_ns_base,
                              const Observation& prev_obs,
                              const Observation& cur_obs,
                              double acc_var_th, double gyr_var_th, double disp_th) {
    int64_t t0_ns = t0_ns_base + (int64_t)(t0_s * 1e9);
    int64_t t1_ns = t0_ns_base + (int64_t)(t1_s * 1e9);
    auto lo = std::lower_bound(data.imu.begin(), data.imu.end(), t0_ns,
                               [](const ImuSample& s, int64_t v){ return s.t_ns < v; });
    auto hi = std::upper_bound(data.imu.begin(), data.imu.end(), t1_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    int n = (int)std::distance(lo, hi);
    if (n < 5) return false;

    Eigen::Vector3d a_mean = Eigen::Vector3d::Zero(), w_mean = Eigen::Vector3d::Zero();
    for (auto it = lo; it != hi; ++it) { a_mean += it->a; w_mean += it->w; }
    a_mean /= n;  w_mean /= n;
    double a_var = 0.0, w_var = 0.0;
    for (auto it = lo; it != hi; ++it) {
        a_var += (it->a - a_mean).squaredNorm();
        w_var += (it->w - w_mean).squaredNorm();
    }
    a_var /= n;  w_var /= n;

    std::unordered_map<int64_t, Eigen::Vector2d> prev_map;
    prev_map.reserve(prev_obs.ids.size());
    for (size_t i = 0; i < prev_obs.ids.size(); ++i)
        prev_map[prev_obs.ids[i]] = Eigen::Vector2d(prev_obs.pts_raw(i, 0),
                                                    prev_obs.pts_raw(i, 1));
    std::vector<double> disp;
    disp.reserve(cur_obs.ids.size());
    for (size_t i = 0; i < cur_obs.ids.size(); ++i) {
        auto it = prev_map.find(cur_obs.ids[i]);
        if (it == prev_map.end()) continue;
        double dx = cur_obs.pts_raw(i, 0) - it->second(0);
        double dy = cur_obs.pts_raw(i, 1) - it->second(1);
        disp.push_back(std::sqrt(dx * dx + dy * dy));
    }
    double median_disp = 1e9;
    if (!disp.empty()) {
        std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
        median_disp = disp[disp.size() / 2];
    }

    return a_var < acc_var_th && w_var < gyr_var_th && median_disp < disp_th;
}

// ---------------------------------------------------------------------------
// Folder frontend source (first cut). Read pre-computed point features and
// IMU from <dir>/{points,imu}.csv (extract_frontend_features.py output) and
// fill the same structures the image tracker would: replace data.imu, rebuild
// data.frames from the feature-message timestamps (~10 Hz), and populate the
// per-frame observations. cam_calib + GT stay from
// load_dataset; the bag features share the EuRoC clock so GT alignment by
// timestamp is coherent. Points carry undistorted-normalized coords (pts_norm,
// re-projected to pixels by the backend) + raw pixels (pts_raw).
// ---------------------------------------------------------------------------
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

static void load_folder_frontend(
        const std::string& dir, EurocData& data,
        std::vector<std::tuple<int, double, Observation>>& observations) {
    auto fail = [](const std::string& f) {
        std::cerr << "FATAL: folder frontend cannot open " << f << "\n"; std::exit(1); };

    // --- IMU: t_ns,wx,wy,wz,ax,ay,az ---
    {
        std::ifstream f(dir + "/imu.csv");
        if (!f) fail(dir + "/imu.csv");
        std::vector<ImuSample> imu;
        std::string ln; std::getline(f, ln);                  // header
        while (std::getline(f, ln)) {
            if (ln.empty()) continue;
            auto c = split_csv(ln);
            if (c.size() < 7) continue;
            ImuSample s;
            s.t_ns = std::stoll(c[0]);
            s.w = Eigen::Vector3d(std::stod(c[1]), std::stod(c[2]), std::stod(c[3]));
            s.a = Eigen::Vector3d(std::stod(c[4]), std::stod(c[5]), std::stod(c[6]));
            imu.push_back(s);
        }
        std::sort(imu.begin(), imu.end(),
                  [](const ImuSample& a, const ImuSample& b) { return a.t_ns < b.t_ns; });
        data.imu = std::move(imu);
    }

    // --- points grouped by timestamp (ordered) -> frames + observations ---
    struct PtRow { int64_t id; double xn, yn, u, v; };
    std::map<int64_t, std::vector<PtRow>> pts_by_t;
    {
        std::ifstream f(dir + "/points.csv");
        if (!f) fail(dir + "/points.csv");
        std::string ln; std::getline(f, ln);                  // header
        while (std::getline(f, ln)) {
            if (ln.empty()) continue;
            auto c = split_csv(ln);
            if (c.size() < 8) continue;
            pts_by_t[std::stoll(c[1])].push_back(
                PtRow{ std::stoll(c[3]), std::stod(c[4]), std::stod(c[5]),
                       std::stod(c[6]), std::stod(c[7]) });
        }
    }
    if (pts_by_t.empty()) fail(dir + "/points.csv (no rows)");

    // EuRoC images (20Hz) were loaded into data.frames by load_dataset; the folder
    // features are a 10Hz subset stamped ~84ns off the image clock. Snapshot the
    // image (stamp -> path) index BEFORE we clear data.frames, so each feature frame
    // can be matched to its nearest image -> Frame.path. The loop-closure harvest
    // imreads that path; without it folder mode has no images for BRIEF.
    std::vector<std::pair<int64_t, std::string>> img_index;
    img_index.reserve(data.frames.size());
    for (const auto& fr : data.frames) img_index.emplace_back(fr.t_ns, fr.path);
    std::sort(img_index.begin(), img_index.end());

    data.frames.clear();
    observations.clear();
    const int64_t t0_ns = pts_by_t.begin()->first;

    std::unordered_map<int64_t, int> prev_age;
    int frame_idx = 0;
    for (const auto& kv : pts_by_t) {
        const int64_t t_ns = kv.first;
        const auto& rows = kv.second;
        const int N = (int)rows.size();

        Observation obs;
        obs.ids.resize(N);
        obs.track_cnt.resize(N);
        obs.pts_raw.resize(N, 2);
        obs.pts_norm.resize(N, 2);
        std::unordered_map<int64_t, int> cur_age;
        for (int j = 0; j < N; ++j) {
            const int64_t id = rows[j].id;
            obs.ids[j]        = id;
            obs.pts_norm(j, 0) = rows[j].xn;
            obs.pts_norm(j, 1) = rows[j].yn;
            obs.pts_raw(j, 0)  = rows[j].u;
            obs.pts_raw(j, 1)  = rows[j].v;
            const int age = prev_age.count(id) ? prev_age[id] + 1 : 1;  // consecutive track len
            obs.track_cnt[j] = age;
            cur_age[id] = age;
        }
        prev_age.swap(cur_age);

        // Nearest EuRoC image to this feature stamp (enables folder-mode loop
        // closure). Tolerance 1ms; the real offset is ~84ns so this always hits.
        std::string img_path;
        if (!img_index.empty()) {
            auto hi = std::lower_bound(img_index.begin(), img_index.end(),
                                       std::make_pair(t_ns, std::string()));
            int64_t best_d = INT64_MAX;
            if (hi != img_index.end()) { best_d = std::llabs(hi->first - t_ns); img_path = hi->second; }
            if (hi != img_index.begin()) {
                auto lo = std::prev(hi);
                if (std::llabs(lo->first - t_ns) < best_d) {
                    best_d = std::llabs(lo->first - t_ns); img_path = lo->second;
                }
            }
            if (best_d > 1000000) img_path.clear();   // >1ms: no matching image
        }
        data.frames.push_back(Frame{t_ns, img_path});
        observations.emplace_back(frame_idx, (t_ns - t0_ns) * 1e-9, std::move(obs));
        ++frame_idx;
    }
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;   // flush every <<, so a crash doesn't swallow the log

    // Determinism: every RANSAC in the pipeline (frontend F-matrix, init
    // Essential/PnP, loop F-matrix/PnP) draws from OpenCV's GLOBAL RNG. Unseeded,
    // it is seeded from wall-clock time -> the SAME config produces DIFFERENT ATE
    // run-to-run (observed: MH_05 +LC 0.233 vs 0.309). Seed it once here so every
    // run is reproducible and results are comparable / publishable.
    cv::setRNGSeed(0);
    std::srand(0);
    std::cout << "[determinism] cv::setRNGSeed(0) + srand(0)\n";

    std::string dataset_root = (argc > 1)
        ? std::string(argv[1])
        : (fs::current_path().parent_path() / "MH_01_easy").string();

    // Central config (2nd CLI arg overrides; missing file -> built-in defaults).
    std::string config_path = (argc > 2) ? std::string(argv[2]) : "config/vio_params.yaml";
    VioParams P = VioParams::load(config_path);

    // Determinism (NEES study): GTSAM's linear solver uses TBB; parallel
    // scheduling makes the same config yield float-different (bit-nonidentical)
    // runs — tolerable for ATE, fatal for a NEES curve. Pinning TBB to 1 thread
    // makes runs bit-reproducible (verified: serial runs are already identical;
    // only parallel scheduling flips them). Must outlive the whole run.
    std::unique_ptr<tbb::global_control> tbb_pin;
    if (P.deterministic) {
        tbb_pin = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, 1);
        std::cout << "[determinism] TBB pinned to 1 thread (runtime.deterministic)\n";
    }

    EurocData data = load_dataset(dataset_root);

    // Frontend source. DEFAULT "folder" reads the PL-VINS 20Hz feature+IMU bags and
    // pre-fills data.imu + data.frames + observations (loop closure still runs — each
    // feature frame is matched to its EuRoC image for BRIEF). "images" runs the legacy
    // KLT/LSD tracker below (ablation).
    const bool kFolderFrontend = (P.frontend_source == "folder");
    std::vector<std::tuple<int, double, Observation>> observations;
    if (kFolderFrontend) {
        // Auto-derive the per-sequence feature folder from the dataset path when the
        // (shared) config leaves frontend_dir empty: <dataset_parent>/<frontend_root>/<name>.
        if (P.frontend_dir.empty()) {
            fs::path dsp(dataset_root);
            std::string nm = dsp.filename().string();
            if (nm.empty()) { dsp = dsp.parent_path(); nm = dsp.filename().string(); }
            P.frontend_dir = (dsp.parent_path() / P.frontend_root / nm).string();
            std::cout << "[frontend] folder_dir auto-derived: " << P.frontend_dir << "\n";
        }
        load_folder_frontend(P.frontend_dir, data, observations);
        // Loop closure now works in folder mode: load_folder_frontend matched each
        // feature frame to its nearest EuRoC image (Frame.path) for BRIEF. Disable
        // only if no images resolved (e.g. dataset_root has no mav0/cam0/data).
        if (P.loop_closure) {
            size_t with_img = 0;
            for (const auto& fr : data.frames) if (!fr.path.empty()) ++with_img;
            if (with_img == 0) {
                std::cerr << "[frontend] folder mode: no images matched feature stamps "
                             "-- forcing loop_closure OFF.\n";
                P.loop_closure = false;
            } else {
                std::cout << "[frontend] folder mode: loop closure ON (" << with_img
                          << "/" << data.frames.size()
                          << " frames have an image for BRIEF)\n";
            }
        }
        std::cout << "[frontend] folder source: " << P.frontend_dir << "  ("
                  << data.frames.size() << " feature frames, "
                  << data.imu.size() << " imu samples)\n";
    }

    int64_t t0_ns = data.frames.front().t_ns;

    std::cout << "Dataset Loaded: " << data.frames.size()
              << " frames, " << data.imu.size() << " IMU samples.\n";

    // All outputs go under results/<dataset_name>/ so sequences don't overwrite
    // each other and can be compared (dataset name = last path component).
    fs::path ds_path(dataset_root);
    std::string ds_name = ds_path.filename().string();
    if (ds_name.empty()) ds_name = ds_path.parent_path().filename().string();  // trailing slash
    std::string out_dir = "results/" + ds_name;
    fs::create_directories(out_dir);
    std::cout << "Output dir: " << out_dir << "\n";

    // Point front-end: KLT + F-matrix RANSAC (config: point_frontend). In image mode
    // it runs on its OWN thread (FrontendThread); this thread produces images + IMU and
    // consumes tracked observations. Folder mode pre-fills observations and skips it.
    InitBuffer buf(P.init_window, P.init_parallax_px, P.init_min_common);

    // Dump per-frame front-end detections for offline inspection / view_frontend.py
    // (config: runtime.dump_frontend).
    const bool kDumpFrontend = P.dump_frontend;
    std::ofstream f_frames, f_points;
    if (kDumpFrontend) {
        // Front-end dumps live in their own tree, one folder per dataset, so they
        // never mix with the VIO results under results/.
        std::string fe_dir = "results_frontend/" + ds_name;
        fs::create_directories(fe_dir);
        std::cout << "Frontend dump dir: " << fe_dir << "\n";
        f_frames.open(fe_dir + "/frontend_frames.csv");
        f_points.open(fe_dir + "/frontend_points.csv");
        f_frames << "frame,t_s,image_path\n";
        f_points << "frame,id,u,v,track_cnt\n";
    }

    // Camera matrices for keypoint undistortion (radtan k1,k2,p1,p2), used by
    // the loop-closure harvest below.
    cv::Mat Kcv = (cv::Mat_<double>(3, 3) <<
        data.cam_calib.K(0, 0), 0, data.cam_calib.K(0, 2),
        0, data.cam_calib.K(1, 1), data.cam_calib.K(1, 2),
        0, 0, 1);
    cv::Mat Dcv = (cv::Mat_<double>(4, 1) << data.cam_calib.dist(0), data.cam_calib.dist(1),
                                             data.cam_calib.dist(2), data.cam_calib.dist(3));

    // Pre-pass: track features across all frames on a dedicated frontend thread.
    // Skipped entirely in folder mode (observations were pre-filled from CSV).
    //
    // Three-thread pipeline for this phase:
    //   producer (spawned) : load_image + fan IMU  -> img_q / imu_q
    //   FrontendThread      : KLT + IMU-aided pred  -> obs_q
    //   this thread         : drain obs_q IN ORDER  -> observations[]
    // A single frontend worker draining a FIFO image queue guarantees observation
    // order == frame order, so the downstream observations[] contract is unchanged.
    int n_frames = (int)data.frames.size();   // always process every frame
    if (!kFolderFrontend) {
    observations.reserve(n_frames);
    std::cout << "\n--- Detecting features across all frames (threaded frontend"
              << (P.pt_imu_aided_klt ? ", IMU-aided" : "") << ") ---\n";

    ThreadSafeQueue<StampedImage>       img_q(16);   // camera-rate images  (bounded backpressure)
    ThreadSafeQueue<StampedObservation> obs_q(16);   // tracked features out
    ThreadSafeQueue<ImuSample>          imu_q;       // gyro stream for IMU-aided KLT (unbounded)

    FrontendThread::Params fp;
    fp.max_pts   = P.pt_max;
    fp.min_dist  = P.pt_min_dist_px;
    fp.fb_err    = P.pt_fb_err;
    fp.ransac_th = P.pt_ransac_th;
    fp.quality   = P.pt_quality;
    fp.rng_seed  = P.deterministic ? 0 : -1;   // match the main thread's cv::setRNGSeed(0)
    FrontendThread fe(data.cam_calib, fp, img_q, obs_q,
                      P.pt_imu_aided_klt ? &imu_q : nullptr);
    fe.start();

    // Producer: push all IMU up to each image stamp BEFORE the image (so the frontend's
    // gyro integration up to that stamp is available), then the image; close when done.
    std::thread producer([&]{
        size_t imu_i = 0;
        for (int i = 0; i < n_frames; ++i) {
            const int64_t t_ns = data.frames[i].t_ns;
            while (imu_i < data.imu.size() && data.imu[imu_i].t_ns <= t_ns)
                imu_q.push(data.imu[imu_i++]);
            StampedImage si;
            si.t_ns = t_ns;
            si.img  = load_image(data.frames[i].path);
            img_q.push(std::move(si));
        }
        img_q.close();
        imu_q.close();
    });

    // Consumer (this thread): drain observations in order into observations[].
    StampedObservation so;
    int i = 0;
    while (obs_q.pop(so)) {
        double t_s = (data.frames[i].t_ns - t0_ns) * 1e-9;
        if (kDumpFrontend) {
            f_frames << i << ',' << t_s << ',' << data.frames[i].path << '\n';
            for (size_t j = 0; j < so.obs.ids.size(); ++j)
                f_points << i << ',' << so.obs.ids[j] << ','
                         << so.obs.pts_raw(j, 0) << ',' << so.obs.pts_raw(j, 1) << ','
                         << so.obs.track_cnt[j] << '\n';
        }
        observations.emplace_back(i, t_s, std::move(so.obs));
        if ((i + 1) % 100 == 0) std::cout << "  frame " << (i + 1) << "/" << n_frames << "\n";
        ++i;
    }
    producer.join();
    fe.join();
    if (kDumpFrontend) { f_frames.close(); f_points.close(); }
    }  // end image-frontend pre-pass (folder mode pre-filled observations)

    // PHASE 1: Bootstrap
    std::cout << "\n--- Phase 1: Hunting for Initialization Window ---\n";
    AlignedInit aligned;
    int last_processed_idx = -1;

    // VINS-Mono defer-until-well-conditioned: do NOT commit the first window that
    // merely passes the alignment sanity check. The static/pickup phase of
    // MH_03/MH_04 is rotation-dominant with a tiny translational baseline, so its
    // multi-view SfM is geometrically inconsistent. The post-BA residual hides
    // this (LM buries the error in far landmark depths), but the PRE-BA per-obs
    // reprojection RMS exposes it: MH_01 ~2.5px vs MH_03 ~12px vs MH_04 ~17px.
    // Reject ill-conditioned windows and keep sliding until a translation-rich
    // flight window is found -- the same window MH_01 gets for free.
    const double kInitMaxPreBaRmsPx = P.init_max_pre_ba_rms_px;
    const double kInitMaxVel = P.init_max_vel_mps;  // VI-consistency init gate (config init.max_vel_mps)
    const bool   kVioDiag    = P.diag;              // diagnostic prints (config runtime.diag)

    for (int i = 0; i < n_frames; ++i) {
        const auto& obs = std::get<2>(observations[i]);
        double t_s = std::get<1>(observations[i]);

        if (buf.try_add(i, t_s, obs)) {
            if (buf.is_full()) {
                // Bounded sliding window: shift whenever full (below), even before
                // IMU excitation passes, so the buffer never exceeds window_size.
                // At 10 Hz (folder frontend) it otherwise grew to 2x window before
                // excitation, and an over-long init window spans too much rotation
                // -> degenerate SfM (pre-BA RMS explodes, every window rejected).
                if (check_imu_excitation(buf.frames, data, P.init_excitation_var)) {
                SfmResult sfm_try = init_sfm(buf.frames, data.cam_calib);
                if (sfm_try.success) {
                    SfmResult sfm_ba = sfm_try;
                    bundle_adjust(sfm_ba, 20);
                    double pre_ba_rms = sfm_ba.ba_n_obs > 0
                        ? std::sqrt(2.0 * sfm_ba.ba_err_before / (double)sfm_ba.ba_n_obs)
                        : 1e9;
                    if (pre_ba_rms > kInitMaxPreBaRmsPx) {
                        std::cout << "init: window rejected -- SfM ill-conditioned "
                                  << "(pre-BA reproj RMS " << pre_ba_rms << " px > "
                                  << kInitMaxPreBaRmsPx << " px); sliding.\n";
                    } else {
                        AlignedInit align_try = init_align(sfm_ba, data);
                        if (align_try.success) {
                            // VI-consistency gate: a bad alignment inflates the init
                            // velocities (scale error) -> the online VI-BA reconciles it
                            // with a runaway accel bias -> divergence (V2_02: max|v|=34.6,
                            // ba->4.09). Reject the implausible window and slide to a later,
                            // better-conditioned one (mirrors VINS re-init intent).
                            double vmax = 0.0;
                            for (const auto& vv : align_try.velocities)
                                if (vv.norm() > vmax) vmax = vv.norm();
                            if (vmax > kInitMaxVel) {
                                std::cout << "init: window rejected -- VI-alignment implausible "
                                          << "(max|v|=" << vmax << " m/s > " << kInitMaxVel
                                          << " m/s); sliding.\n";
                            } else {
                                aligned = align_try;
                                vins_realign(aligned);
                                last_processed_idx = i;
                                std::cout << "\n[SUCCESS] Initialization Window Resolved "
                                          << "(pre-BA reproj RMS " << pre_ba_rms << " px).\n";
                                if (kVioDiag)
                                    std::cout << "[initdiag] window@frame=" << i
                                              << " scale=" << aligned.scale
                                              << " max|v|=" << vmax << " m/s"
                                              << " |b_g|=" << aligned.b_g.norm() << " rad/s\n";
                                break;
                            }
                        }
                    }
                }
                }   // close excitation gate (init only attempted when excited)
                buf.shift();   // ALWAYS shift when full -> bounded sliding window
            }
        }
    }

    if (!aligned.success) {
        std::cout << "\nFATAL: Failed to initialize system.\n";
        return 1;
    }

    if(kDumpFrontend)
    {
        return 0;   // skip backend and CSV output if we're just dumping frontend data
    }


    // PHASE 2: Configure backend
    // Use the gravity magnitude the initializer actually computed/constrained
    // (aligned.g_w was rotated to -Z by vins_realign; its norm == G_MAG). This
    // keeps the preintegration gravity identical to the init's, no second hardcode.
    auto imu_params =
        gtsam::PreintegrationCombinedParams::MakeSharedU(aligned.g_w.norm());

    // IMU noise from config (imu_noise): dataset densities x covariance
    // multipliers in the variance domain. See config/vio_params.yaml for values
    // and rationale (inflate IMU white noise -> trust vision more).
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    imu_params->setAccelerometerCovariance(I3 * std::pow(P.imu_acc_n, 2) * P.imu_acc_cov_mult);
    imu_params->setGyroscopeCovariance    (I3 * std::pow(P.imu_gyr_n, 2) * P.imu_gyr_cov_mult);
    imu_params->setIntegrationCovariance  (I3 * P.imu_integration_cov);
    imu_params->setBiasAccCovariance      (I3 * std::pow(P.imu_acc_w, 2) * P.imu_acc_bias_cov_mult);
    imu_params->setBiasOmegaCovariance    (I3 * std::pow(P.imu_gyr_w, 2) * P.imu_gyr_bias_cov_mult);
    imu_params->biasAccOmegaInt = Eigen::Matrix<double, 6, 6>::Identity() * P.imu_bias_acc_omega_int;

    VioBackend backend(data.cam_calib.K, data.cam_calib.T_BC, imu_params, P);

    // VINS-Mono solveOdometry(): seed the WHOLE bootstrap window into the smoother
    // (states + IMU factors + bootstrap landmarks) so it is re-optimized
    // continuously as online keyframes arrive, instead of a single frozen seed.
    backend.initialize_window(aligned.T_wb, aligned.velocities, aligned.sfm.frames,
                              aligned.sfm.points, aligned.b_g, data.imu, t0_ns);

    // Init-window CSV block = the backend's refined poses (falls back to the raw
    // aligned pose if a state already marginalized out of the lag window).
    std::vector<double> estimated_ts;
    std::vector<Eigen::Matrix4d> estimated_poses;
    // Per-KF pose marginal covariance for the NEES study (runtime.dump_covariance),
    // captured ONLINE while each state is still in the window; kept 1:1 aligned
    // with estimated_poses. NaN = unavailable (state already marginalized).
    std::vector<Eigen::Matrix<double, 6, 6>> est_covs;
    auto push_cov = [&](int idx) {
        Eigen::Matrix<double, 6, 6> C;
        if (!P.dump_covariance || !backend.get_pose_covariance(idx, C))
            C.setConstant(std::numeric_limits<double>::quiet_NaN());
        est_covs.push_back(C);
    };
    // ABLATION A5 (init.continuous_window=false): the bootstrap is NOT in the
    // smoother -- state 0 is the frozen seed carrying the LAST bootstrap keyframe's
    // pose, so get_pose_at(k) is meaningless for this block (k=0 would return the
    // last keyframe's pose stamped with the first keyframe's time). Emit the raw
    // aligned SfM poses instead, which IS the frozen-seed trajectory by definition.
    // With the default (true) this is the original code path exactly.
    const bool iw_in_smoother = P.init_continuous_window;
    for (size_t k = 0; k < aligned.sfm.frames.size(); ++k) {
        estimated_ts.push_back(aligned.sfm.frames[k].t);
        Eigen::Matrix4d refined;
        estimated_poses.push_back((iw_in_smoother && backend.get_pose_at((int)k, refined))
                                      ? refined
                                      : aligned.T_wb[k]);
        if (iw_in_smoother) {
            push_cov((int)k);
        } else {
            // No smoother state backs this row -> covariance is undefined, not zero.
            Eigen::Matrix<double, 6, 6> C;
            C.setConstant(std::numeric_limits<double>::quiet_NaN());
            est_covs.push_back(C);
        }
    }

    double fx = data.cam_calib.K(0, 0), fy = data.cam_calib.K(1, 1);
    double cx = data.cam_calib.K(0, 2), cy = data.cam_calib.K(1, 2);

    // --- Loop-closure harvest setup (online pose graph; see loop/) ----------
    // Every keyframe is stored with its VIO pose, tracked 3D points + BRIEF, and
    // image FAST-corner BRIEF for DBoW place recognition. Loops are detected and
    // the pose graph re-optimized ONLINE, per keyframe, inside the loop below.
    const bool harvest_loop = P.loop_closure;   // runtime.loop_closure (false = pure-VIO no-LC baseline)
    // Loop-closure tuning threaded from config (Fronts A.1 + B).
    vio::LoopParams lp;
    lp.skip_recent      = P.lc_skip_recent;
    lp.min_inliers      = P.lc_min_inliers;
    lp.hamming_th       = P.lc_hamming_th;
    lp.score_best       = P.lc_score_best;
    lp.score_cand       = P.lc_score_cand;
    lp.fund_thresh_px   = P.lc_fund_thresh_px;
    lp.max_trans_m      = P.lc_max_trans_m;
    lp.max_yaw_deg      = P.lc_max_yaw_deg;
    lp.backbone_hops    = P.lc_backbone_hops;
    lp.loop_trans_sigma = P.lc_loop_trans_sigma;
    lp.loop_rot_sigma   = P.lc_loop_rot_sigma;
    lp.cauchy_c         = P.lc_cauchy_c;
    lp.inlier_weighting = P.lc_inlier_weighting;
    lp.min_correction_m = P.lc_min_correction_m;
    lp.loop_drift_rate  = P.lc_loop_drift_rate;
    std::unique_ptr<vio::BriefExtractor> brief;
    std::unique_ptr<vio::LoopDatabase>   loop_db;
    if (harvest_loop) {
        try {
            brief.reset(new vio::BriefExtractor("support_files/brief_pattern.yml"));
            loop_db.reset(new vio::LoopDatabase("support_files/brief_k10L6.bin", lp));
        } catch (const std::exception& e) {
            std::cerr << "[loop] disabled (harvest setup failed): " << e.what() << "\n";
            brief.reset();  loop_db.reset();
        }
    }
    vio::PoseGraph pg(lp);   // persistent online pose graph (parallel to the VIO)

    // Online loop-closure state (VINS-Mono estimator/pose_graph split, run
    // synchronously here instead of as a second ROS node). The pose graph is
    // NON-INVASIVE: the estimator output (estimated_poses) is never touched, so it
    // stays a clean no-LC baseline. T_drift = w_T_vio maps a raw VIO body pose into
    // the latest loop-corrected frame; recomputed after every in-loop pose-graph
    // optimize and fed forward to later keyframes. lc_poses is the loop-corrected
    // trajectory grown alongside estimated_poses (T_drift = I until the first loop,
    // so lc == est there). [Faithful to VINS-Mono with FAST_RELOCALIZATION=0.]
    Eigen::Matrix4d T_drift = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Matrix4d> lc_poses = estimated_poses;

    // [Step-1 diagnostic] GT-label each loop: nearest-GT position at a KF's time.
    // Lets us tell a TRUE revisit (two KFs co-located in GT) from a FALSE loop
    // (perceptual aliasing -> KFs far apart in GT). Log-only; behaviour unchanged.
    const int64_t loopgt_t0 = data.frames.empty() ? 0 : data.frames.front().t_ns;
    auto gt_pos_at = [&data, loopgt_t0](double t_rel) {
        Eigen::Vector3d p = Eigen::Vector3d::Zero(); double best = 1e18;
        for (const auto& s : data.gt) {
            double d = std::fabs((double)(s.t_ns - loopgt_t0) * 1e-9 - t_rel);
            if (d < best) { best = d; p = s.p; }
        }
        return p;
    };

    // PHASE 3: Online VIO
    std::cout << "\n--- Phase 2: Online VIO Processing ---\n";
    double prev_t_s = estimated_ts.back();
    // Backend-update wall-clock (estimator only: the add_frame call, excluding frontend,
    // diagnostics, and covariance dump). Reported as the [timing] speed metric.
    double backend_update_s = 0.0;
    int    n_online_frames  = 0;
    for (int i = last_processed_idx + 1; i < n_frames; ++i) {
        double curr_t_s = (data.frames[i].t_ns - t0_ns) * 1e-9;
        integrate_imu_batch(backend, data, prev_t_s, curr_t_s, t0_ns);

        const auto& obs = std::get<2>(observations[i]);
        const auto& prev_obs = std::get<2>(observations[i - 1]);
        bool stationary = detect_stationary(data, prev_t_s, curr_t_s, t0_ns, prev_obs, obs,
                                            P.zupt_acc_var, P.zupt_gyr_var, P.zupt_disp_px);

        std::unordered_map<int64_t, Eigen::Vector2d> obs_dict;
        obs_dict.reserve(obs.ids.size());
        for (size_t j = 0; j < obs.ids.size(); ++j) {
            double xn = obs.pts_norm(j, 0);
            double yn = obs.pts_norm(j, 1);
            obs_dict[obs.ids[j]] = Eigen::Vector2d(fx * xn + cx, fy * yn + cy);
        }

        auto _t_bk = std::chrono::steady_clock::now();
        bool is_kf = backend.add_frame(obs_dict, curr_t_s, stationary);
        backend_update_s += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - _t_bk).count();
        ++n_online_frames;
        if (is_kf) {
            estimated_ts.push_back(curr_t_s);
            Eigen::Matrix4d Twb = backend.get_current_pose();
            estimated_poses.push_back(Twb);
            push_cov(backend.get_state_idx());   // online covariance of the just-created KF
            lc_poses.push_back(T_drift * Twb);   // loop-corrected (causal); est stays clean no-LC

            // --- TEMP per-keyframe diagnostic: nav-state runaway + map health -------
            // |p| explodes / vel|bias spike  -> divergence onset (V-room).
            // pts_obs vs pts_g -> tracking & triangulation yield.
            if (kVioDiag) {
                const auto dg = backend.get_diag();
                const Eigen::Vector3d p = Twb.block<3, 1>(0, 3);
                std::cout << "[viodiag] kf=" << backend.get_state_idx()
                          << " f=" << i << " t=" << curr_t_s
                          << " |p|=" << p.norm() << " vel=" << dg.vel_norm
                          << " ba=" << dg.ba_norm << " bg=" << dg.bg_norm
                          << " pts_obs=" << obs_dict.size() << " pts_g=" << dg.n_pts_graph
                          << "\n";
            }

            // --- harvest this keyframe for loop closure ---
            if (brief && loop_db) {
                cv::Mat img = cv::imread(data.frames[i].path, cv::IMREAD_GRAYSCALE);
                if (!img.empty()) {
                    std::unordered_map<int64_t, Eigen::Vector3d> lm_world;
                    backend.get_landmarks_world(lm_world);

                    vio::LoopKeyframe kf;
                    kf.index     = (int)loop_db->size();
                    kf.frame_idx = i;
                    kf.traj_idx  = (int)estimated_poses.size() - 1;  // row just pushed
                    kf.t_s       = curr_t_s;
                    kf.T_wb      = Twb;   // raw VIO pose: pose graph stays in the VIO frame

                    // tracked features that have a 3D estimate (for PnP)
                    std::vector<cv::KeyPoint> pt_keys;
                    for (size_t j = 0; j < obs.ids.size(); ++j) {
                        auto it = lm_world.find(obs.ids[j]);
                        if (it == lm_world.end()) continue;
                        double xn = obs.pts_norm(j, 0), yn = obs.pts_norm(j, 1);
                        float u = (float)(fx * xn + cx), v = (float)(fy * yn + cy);
                        kf.point_3d.emplace_back((float)it->second.x(),
                                                 (float)it->second.y(), (float)it->second.z());
                        kf.point_2d_uv.emplace_back(u, v);
                        kf.point_2d_norm.emplace_back((float)xn, (float)yn);
                        kf.point_id.push_back(obs.ids[j]);
                        // BRIEF must be sampled at the TRUE distorted-image pixel (where the
                        // feature physically is) so it matches old's FAST descriptors. Folder
                        // features carry the bag's raw u_px/v_px in pts_raw; the undistorted
                        // fx*xn+cx mis-samples and halves PnP inliers. Image mode keeps its
                        // validated KLT behaviour (fx*xn+cx) untouched.
                        float bu = u, bv = v;
                        if (kFolderFrontend) { bu = (float)obs.pts_raw(j, 0); bv = (float)obs.pts_raw(j, 1); }
                        pt_keys.emplace_back(bu, bv, 7.f);
                    }
                    (*brief)(img, pt_keys, kf.point_desc);   // 1:1 with point_* arrays

                    // FAST corners over the whole image (DBoW + matching)
                    cv::FAST(img, kf.keypoints, P.lc_fast_threshold, true);
                    (*brief)(img, kf.keypoints, kf.brief_desc);
                    if (!kf.keypoints.empty()) {
                        std::vector<cv::Point2f> px, norm;
                        px.reserve(kf.keypoints.size());
                        for (const auto& k : kf.keypoints) px.push_back(k.pt);
                        cv::undistortPoints(px, norm, Kcv, Dcv);   // -> normalized undistorted
                        for (size_t j = 0; j < kf.keypoints.size(); ++j) {
                            cv::KeyPoint kn = kf.keypoints[j];
                            kn.pt = norm[j];
                            kf.keypoints_norm.push_back(kn);
                        }
                    }
                    // --- ONLINE loop closure: query DB (before adding this KF),
                    //     grow the pose graph, re-optimize, and feed the drift back ---
                    auto loops = loop_db->query_loops(kf, data.cam_calib.T_BC);
                    // [Step-1 diagnostic] label each candidate loop against GT (diag-gated).
                    if (kVioDiag) {
                        for (const auto& e : loops) {
                            const auto& okf = loop_db->keyframes()[e.index_old];
                            double gtd  = (gt_pos_at(kf.t_s) - gt_pos_at(okf.t_s)).norm();
                            double viod = (kf.T_wb.block<3,1>(0,3) - okf.T_wb.block<3,1>(0,3)).norm();
                            std::cout << "[loopgt] KF " << e.index_cur << "<->" << e.index_old
                                      << " inliers=" << e.inliers << " gt_dist=" << gtd
                                      << " vio_dist=" << viod
                                      << (gtd < 1.0 ? "  near" : "  FAR") << "\n";
                        }
                    }
                    loop_db->add(kf);
                    pg.add_keyframe(kf.index, kf.T_wb);   // raw VIO pose: pg stays in the VIO frame
                    for (const auto& e : loops) pg.add_loop(e);
                    if (!loops.empty()) {
                        pg.optimize();
                        // Drift of the newest node (optimized vs raw VIO), fed forward to
                        // every later keyframe via T_drift. The estimator is NOT touched
                        // (no relocalization) -> est stays a clean no-LC baseline.
                        T_drift = pg.corrected(kf.index) * kf.T_wb.inverse();
                        // A loop moves earlier nodes too, so re-read the whole globally
                        // re-optimized path, not just the tail.
                        for (const auto& k : loop_db->keyframes())
                            if (k.traj_idx >= 0 && k.traj_idx < (int)lc_poses.size())
                                lc_poses[k.traj_idx] = pg.corrected(k.index);
                    }
                }
            }
        }
        prev_t_s = curr_t_s;

        if ((i - last_processed_idx) % 100 == 0)
            std::cout << "  ISAM2 frame " << i << "/" << n_frames << "\n";
    }

    std::cout << "Total keyframes: " << backend.get_state_idx()
              << " of " << (n_frames - last_processed_idx - 1) << " online frames\n";
    {
        const int n_kf = backend.get_state_idx();
        std::cout << "[timing] backend_update_total_s=" << backend_update_s
                  << " n_online_frames=" << n_online_frames << " n_kf=" << n_kf
                  << " mean_ms_per_frame="
                  << (n_online_frames ? 1e3 * backend_update_s / n_online_frames : 0.0)
                  << " mean_ms_per_kf="
                  << (n_kf ? 1e3 * backend_update_s / n_kf : 0.0) << "\n";
    }
    backend.print_kf_stats();

    if (loop_db) {
        size_t tot3d = 0, totfast = 0;
        for (const auto& kf : loop_db->keyframes()) { tot3d += kf.point_3d.size(); totfast += kf.keypoints.size(); }
        const size_t n = loop_db->size();
        std::cout << "[loop] harvested " << n << " keyframes"
                  << " | avg 3D pts/KF " << (n ? tot3d / n : 0)
                  << " | avg FAST/KF "   << (n ? totfast / n : 0) << "\n";
    }

    // PHASE 4: final flush. The pose graph was grown and re-optimized ONLINE in
    // the loop above and lc_poses already tracks it; one last optimize propagates
    // the final loop's correction onto the tail keyframes created after it.
    if (loop_db && pg.num_loops() > 0) {
        pg.optimize();
        for (const auto& kf : loop_db->keyframes())
            if (kf.traj_idx >= 0 && kf.traj_idx < (int)lc_poses.size())
                lc_poses[kf.traj_idx] = pg.corrected(kf.index);
        std::cout << "[loop] online pose graph: " << loop_db->size()
                  << " nodes, " << pg.num_loops() << " loop edges\n";
    }

    // PHASE 5: Evaluation
    // Pass lc_poses only when a loop actually fired (else: no LC columns written).
    // results/<dataset>/<loop|noloop>/points/vio_results.csv — the loop dimension
    // lives in the folder path, so the A/B runs never overwrite each other and
    // the file is simply vio_results.csv.
    std::string csv_dir = out_dir + "/" + (P.loop_closure ? "loop" : "noloop")
                                  + "/points";
    fs::create_directories(csv_dir);
    std::vector<Eigen::Matrix4d> no_lc;
    evaluate_and_write_csv(estimated_ts, estimated_poses, data, t0_ns, csv_dir, "vio_results.csv",
                           (loop_db && pg.num_loops() > 0) ? lc_poses : no_lc);

    // NEES study dump (runtime.dump_covariance): per-KF est pose + 6x6 marginal
    // covariance (tangent ordering [rot(3), trans(3)], GTSAM Pose3 convention),
    // self-contained for python/nees_eval.py (matched to GT by t_s). Aligned 1:1
    // with the no-LC est trajectory; rows with NaN covariance are skipped there.
    if (P.dump_covariance) {
        std::ofstream fc(csv_dir + "/vio_cov.csv");
        fc << "t_s,px,py,pz,qw,qx,qy,qz";
        for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) fc << ",c" << r << c;
        fc << "\n";
        fc << std::setprecision(12);
        for (size_t i = 0; i < estimated_ts.size() && i < est_covs.size(); ++i) {
            const Eigen::Matrix4d& T = estimated_poses[i];
            Eigen::Quaterniond q(T.block<3, 3>(0, 0));
            fc << estimated_ts[i]
               << ',' << T(0, 3) << ',' << T(1, 3) << ',' << T(2, 3)
               << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
            for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) fc << ',' << est_covs[i](r, c);
            fc << "\n";
        }
        std::cout << "[nees] wrote " << csv_dir << "/vio_cov.csv ("
                  << est_covs.size() << " keyframes)\n";
    }
    return 0;
}