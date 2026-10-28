// gvio-engine-replay: run the online engine on a recording, in process, without MPA.
//
// Two feeder threads stand in for the MPA callback threads of the server (IMU, camera) and push
// into the same VioEngine the server uses. This is the desktop test bench of the online system:
//   --dataset DIR   a EuRoC-layout dataset (EuRoC, or a mantis_qualisys_<log>[_lpf]): images are
//                   already pinhole, the dataset's sensor.yaml is the camera, IMU as written
//   --voxl-log DIR  a raw VOXL MPA log (voxl-logger): fisheye PNG of --cam and run/mpa/imu_apps,
//                   undistorted and low-passed online exactly as the server does on the vehicle
// Timing:
//   --rate 0        lockstep: nothing is dropped, every frame waits for the previous ones. With
//                   the IMU filter off this must reproduce the offline binary byte for byte.
//   --rate R > 0    real time scaled by R: stamps are released at their (scaled) time, the IMU in
//                   batches of --imu-batch-ms, decoding runs ahead in its own thread, the engine
//                   drops what it cannot keep up with.
#include "camera_model.h"
#include "engine.h"
#include "load.h"
#include "params.h"

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void usage() {
    std::cerr <<
        "usage: gvio-engine-replay (--dataset DIR | --voxl-log DIR) --config ESTIMATOR.yaml --out DIR\n"
        "         [--rate R] [--lpf on|off] [--cam NAME] [--intrinsics YML] [--extrinsics CONF]\n"
        "         [--balance B] [--pin-K fu,fv,cu,cv] [--imu-batch-ms MS] [--max-frames N]\n"
        "         [--no-auto-reset]\n"
        "  defaults: --rate 0 (lockstep), --lpf off for --dataset and on for --voxl-log,\n"
        "            --cam tracking_front_misp_norm, intrinsics and extrinsics from the log itself\n";
}

struct Frame { int64_t t_ns; int64_t id; std::string path; };

// voxl-logger camera channel: data.csv (i,timestamp(ns),...) + %05d.png
static std::vector<Frame> voxl_frames(const fs::path& dir) {
    std::ifstream f(dir / "data.csv");
    if (!f) throw std::runtime_error("no " + (dir / "data.csv").string());
    std::vector<Frame> out;
    std::string line;
    std::getline(f, line);                               // header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string i, t;
        std::getline(ss, i, ',');
        std::getline(ss, t, ',');
        char name[16];
        std::snprintf(name, sizeof(name), "%05d.png", std::stoi(i));
        if (fs::exists(dir / name)) out.push_back({std::stoll(t), std::stoll(i), (dir / name).string()});
    }
    return out;
}

// voxl-logger imu channel: i,timestamp(ns),batch_id,AX,AY,AZ,GX,GY,GZ,T
static std::vector<vio::ImuSample> voxl_imu(const fs::path& csv) {
    std::ifstream f(csv);
    if (!f) throw std::runtime_error("no " + csv.string());
    std::vector<vio::ImuSample> out;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::vector<std::string> tok;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, ',')) tok.push_back(x);
        if (tok.size() < 9) continue;
        vio::ImuSample s;
        s.t_ns = std::stoll(tok[1]);
        s.a << std::stod(tok[3]), std::stod(tok[4]), std::stod(tok[5]);
        s.w << std::stod(tok[6]), std::stod(tok[7]), std::stod(tok[8]);
        out.push_back(s);
    }
    return out;
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string dataset, voxl_log, config, out_dir, cam = "tracking_front_misp_norm", intr, extr, lpf, pin;
    double rate = 0.0, balance = 0.0, imu_batch_ms = 10.0;
    long max_frames = -1;
    bool auto_reset = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--dataset") dataset = next();
        else if (a == "--voxl-log") voxl_log = next();
        else if (a == "--config") config = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--rate") rate = std::stod(next());
        else if (a == "--lpf") lpf = next();
        else if (a == "--cam") cam = next();
        else if (a == "--intrinsics") intr = next();
        else if (a == "--extrinsics") extr = next();
        else if (a == "--balance") balance = std::stod(next());
        else if (a == "--pin-K") pin = next();
        else if (a == "--imu-batch-ms") imu_batch_ms = std::stod(next());
        else if (a == "--max-frames") max_frames = std::stol(next());
        else if (a == "--no-auto-reset") auto_reset = false;
        else { usage(); return 2; }
    }
    if ((dataset.empty() == voxl_log.empty()) || config.empty() || out_dir.empty()) { usage(); return 2; }
    if (fs::exists(out_dir) && !fs::is_empty(out_dir)) {
        std::cerr << "ERROR: --out " << out_dir << " exists and is not empty\n";
        return 2;
    }

    gvio::EngineConfig cfg;
    cfg.P = vio::VioParams::load(config);
    cfg.lockstep = (rate <= 0.0);
    cfg.auto_reset = auto_reset;
    cfg.log_dir = out_dir;

    std::vector<Frame> frames;
    std::vector<vio::ImuSample> imu;
    if (!dataset.empty()) {
        vio::EurocData d = vio::load_dataset(dataset);
        cfg.calib = d.cam_calib;                         // the offline binary's camera, distortion included
        cfg.undistort.init_passthrough(d.cam_calib.K, d.cam_calib.width, d.cam_calib.height);
        for (size_t k = 0; k < d.frames.size(); ++k) frames.push_back({d.frames[k].t_ns, (int64_t)k, d.frames[k].path});
        imu = std::move(d.imu);
        cfg.imu_filter.enable = (lpf == "on");
    } else {
        const fs::path log(voxl_log);
        std::string sensor = cam;
        const std::string suffix = "_misp_norm";
        if (sensor.size() > suffix.size() && sensor.compare(sensor.size() - suffix.size(), suffix.size(), suffix) == 0)
            sensor.resize(sensor.size() - suffix.size());
        const auto ic = gvio::load_opencv_intrinsics(
            intr.empty() ? (log / "data/modalai" / ("opencv_" + sensor + "_intrinsics.yml")).string() : intr);
        Eigen::Vector4d pk;
        if (!pin.empty()) {
            std::stringstream ss(pin);
            std::string x;
            for (int k = 0; k < 4 && std::getline(ss, x, ','); ++k) pk(k) = std::stod(x);
        }
        cfg.undistort.init_fisheye(ic, balance, pin.empty() ? nullptr : &pk);
        const Eigen::Matrix4d T_BC = gvio::load_voxl_extrinsic(
            extr.empty() ? (log / "etc/modalai/extrinsics.conf").string() : extr, "imu_apps", sensor);
        frames = voxl_frames(log / "run/mpa" / cam);
        imu = voxl_imu(log / "run/mpa/imu_apps/data.csv");
        cfg.calib = gvio::make_cam_calib(cfg.undistort, T_BC, 30);
        cfg.imu_filter.enable = (lpf != "off");
    }
    // main.cpp applies an `extrinsics:` block of the estimator config the same way.
    if (cfg.P.ext_T_BC.size() == 16) {
        cfg.calib.T_BC = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(cfg.P.ext_T_BC.data());
        std::cout << "[extrinsics] cam0 T_BC OVERRIDDEN from config\n";
    }
    if (max_frames > 0 && (long)frames.size() > max_frames) frames.resize(max_frames);
    if (frames.empty() || imu.empty()) { std::cerr << "ERROR: no frames or no IMU\n"; return 1; }
    std::cout << "[replay] " << frames.size() << " frames, " << imu.size() << " IMU samples, "
              << (cfg.lockstep ? "lockstep" : "real time x" + std::to_string(rate))
              << ", IMU filter " << (cfg.imu_filter.enable ? "on" : "off")
              << ", undistort " << (cfg.undistort.remaps() ? "fisheye" : "off") << "\n";

    gvio::VioEngine engine(cfg);
    std::atomic<int> n_ok{0}, n_kf{0};
    engine.set_output_callback([&](const gvio::EngineOutput& o) {
        if (o.state == gvio::STATE_OK) ++n_ok;
        if (o.is_keyframe) ++n_kf;
    });
    engine.start();
    const auto t_start = Clock::now();

    if (cfg.lockstep) {
        // One feeder: every IMU sample up to a lookahead past a frame, then the frame (the engine
        // blocks when its queues are full, so nothing is ever dropped). With the IMU filter on, the
        // lookahead must exceed its warm-up (1024 samples, 1 s) or the first frame is never covered
        // while the feeder waits for room: a deadlock. The sync thread hands out only the IMU up to
        // each frame's stamp, so a longer lookahead changes nothing else.
        const int64_t look = cfg.imu_filter.enable ? 2000000000LL : 100000000LL;
        size_t j = 0;
        bool imu_end = false;
        for (const auto& f : frames) {
            while (j < imu.size() && imu[j].t_ns <= f.t_ns + look) engine.push_imu(imu[j++]);
            // Frames past the last IMU sample (a VOXL log's camera runs ~0.4 s longer) would wait
            // for coverage forever while the feeder waits for room: say no more IMU is coming.
            if (j == imu.size() && !imu_end) { engine.imu_finished(); imu_end = true; }
            engine.push_image(f.t_ns, f.id, vio::load_image(f.path));
        }
        while (j < imu.size()) engine.push_imu(imu[j++]);
    } else {
        const int64_t s0 = std::min(frames.front().t_ns, imu.front().t_ns);
        auto at = [&](int64_t t_ns) {                  // wall time at which a stamp is "now"
            return t_start + std::chrono::nanoseconds((int64_t)((t_ns - s0) / rate));
        };
        vio::ThreadSafeQueue<std::pair<Frame, cv::Mat>> decoded(32);
        std::thread decoder([&] {
            for (const auto& f : frames) decoded.push({f, vio::load_image(f.path)});
            decoded.close();
        });
        std::thread imu_feeder([&] {
            const auto step = std::chrono::microseconds((int64_t)(1000 * imu_batch_ms));
            size_t j = 0;
            auto next = t_start;
            while (j < imu.size()) {
                next += step;
                std::this_thread::sleep_until(next);
                while (j < imu.size() && at(imu[j].t_ns) <= Clock::now()) engine.push_imu(imu[j++]);
            }
        });
        std::pair<Frame, cv::Mat> fr;
        while (decoded.pop(fr)) {
            std::this_thread::sleep_until(at(fr.first.t_ns));
            engine.push_image(fr.first.t_ns, fr.first.id, fr.second);
        }
        decoder.join();
        imu_feeder.join();
    }
    engine.finish();
    const double wall = std::chrono::duration<double>(Clock::now() - t_start).count();
    const auto s = engine.stats();
    std::ostringstream st;
    st << "wall_s " << wall << "\nframes_in " << s.frames_in << "\nframes_rejected " << s.frames_rejected
       << "\ndropped_backlog " << s.dropped_backlog << "\ndropped_no_imu " << s.dropped_no_imu
       << "\nframes_out " << s.frames_out << "\nframes_ok " << n_ok << "\nkeyframes " << n_kf
       << "\ninit_attempts " << s.init_attempts << "\nresets " << s.resets
       << "\nimu_in " << s.imu_in << "\nimu_rejected " << s.imu_rejected << "\nimu_gaps " << s.imu_gaps
       << "\nimu_rate_hz " << s.imu_rate_hz << "\nimu_tau_ms " << s.imu_tau_ms << "\n";
    std::ofstream(fs::path(out_dir) / "stats.txt") << st.str();
    std::cout << "[replay] done\n" << st.str();
    return 0;
}
