// gvio-replay: publish a recording as MPA pipes, the way voxl-imu-server and voxl-camera-server do,
// so voxl-gtsam-vio-server can be run on a desktop or on the vehicle without flying.
//   --voxl-log DIR   a raw VOXL log (voxl-logger): run/mpa/<cam>/ PNG + data.csv, run/mpa/imu_apps
//   --dataset DIR    a EuRoC-layout dataset: mav0/cam0 PNG, mav0/imu0/data.csv
// Pipes: --imu-pipe (default imu_apps) and --cam-pipe (default tracking_front_misp_norm), names under
// /run/mpa/ or full paths (a desktop uses /tmp/...). Timing:
//   --rate R             real time scaled by R, IMU in batches of --imu-batch-ms
//   --lockstep PIPE      publish a frame, then wait until the server's ext_vio_data_t PIPE reports it
//                        (last_cam_timestamp_ns), so nothing is ever dropped (parity tests)
//   --live-stamps        shift every stamp by (now - first stamp), as voxl-replay does, so the data
//                        looks live to consumers that check staleness (voxl-vision-hub)
// The IMU goes out as imu_data_t, i.e. float32: that rounding is the one difference from an
// in-process replay of the same data.
#include "load.h"

#include <modal_pipe.h>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Frame { int64_t t_ns; int64_t id; std::string path; int32_t exposure_ns; int16_t gain; };
struct Imu { int64_t t_ns; float a[3], w[3], temp; };

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) out.push_back(x);
    return out;
}

static std::atomic<int64_t> g_last_out_cam_t{-1};
static std::mutex g_m;
static std::condition_variable g_cv;

static void out_cb(int, char* data, int bytes, void*) {
    int n = 0;
    ext_vio_data_t* d = pipe_validate_ext_vio_data_t(data, bytes, &n);
    if (!d || n <= 0) return;
    g_last_out_cam_t = d[n - 1].last_cam_timestamp_ns;
    g_cv.notify_all();
}

static void make_info(pipe_info_t& info, const std::string& name_or_path, const char* type, int size) {
    std::string name = name_or_path, loc;
    if (!name_or_path.empty() && name_or_path[0] == '/') {
        loc = name_or_path.back() == '/' ? name_or_path : name_or_path + "/";
        const std::string t = loc.substr(0, loc.size() - 1);
        name = t.substr(t.find_last_of('/') + 1);
    } else {
        loc = std::string(MODAL_PIPE_DEFAULT_BASE_DIR) + name_or_path + "/";
    }
    std::memset(&info, 0, sizeof(info));
    std::strncpy(info.name, name.c_str(), MODAL_PIPE_MAX_NAME_LEN - 1);
    std::strncpy(info.location, loc.c_str(), MODAL_PIPE_MAX_DIR_LEN - 1);
    std::strncpy(info.type, type, MODAL_PIPE_MAX_TYPE_LEN - 1);
    std::strncpy(info.server_name, "gvio-replay", MODAL_PIPE_MAX_NAME_LEN - 1);
    info.size_bytes = size;
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string voxl_log, dataset, cam = "tracking_front_misp_norm", imu_pipe = "imu_apps", cam_pipe, lockstep;
    double rate = 1.0, imu_batch_ms = 10.0;
    bool live = false;
    long max_frames = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) std::exit(2); return argv[++i]; };
        if (a == "--voxl-log") voxl_log = next();
        else if (a == "--dataset") dataset = next();
        else if (a == "--cam") cam = next();
        else if (a == "--imu-pipe") imu_pipe = next();
        else if (a == "--cam-pipe") cam_pipe = next();
        else if (a == "--rate") rate = std::stod(next());
        else if (a == "--imu-batch-ms") imu_batch_ms = std::stod(next());
        else if (a == "--lockstep") lockstep = next();
        else if (a == "--live-stamps") live = true;
        else if (a == "--max-frames") max_frames = std::stol(next());
        else {
            std::cerr << "usage: gvio-replay (--voxl-log DIR [--cam NAME] | --dataset DIR) [--imu-pipe P]"
                         " [--cam-pipe P] [--rate R | --lockstep EXT_PIPE] [--imu-batch-ms MS]"
                         " [--live-stamps] [--max-frames N]\n";
            return 2;
        }
    }
    if (voxl_log.empty() == dataset.empty()) { std::cerr << "need exactly one of --voxl-log, --dataset\n"; return 2; }
    if (cam_pipe.empty()) cam_pipe = cam;

    std::vector<Frame> frames;
    std::vector<Imu> imu;
    if (!voxl_log.empty()) {
        const fs::path dir = fs::path(voxl_log) / "run/mpa" / cam;
        std::ifstream f(dir / "data.csv");
        std::string line;
        std::getline(f, line);                         // i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved
        while (std::getline(f, line)) {
            const auto t = split(line);
            if (t.size() < 8) continue;
            char name[16];
            std::snprintf(name, sizeof(name), "%05d.png", std::stoi(t[0]));
            if (fs::exists(dir / name))
                frames.push_back({std::stoll(t[1]), std::stoll(t[7]), (dir / name).string(),
                                  (int32_t)std::stol(t[3]), (int16_t)std::stoi(t[2])});
        }
        std::ifstream g(fs::path(voxl_log) / "run/mpa/imu_apps/data.csv");
        std::getline(g, line);                         // i,timestamp(ns),batch_id,AX..AZ,GX..GZ,T
        while (std::getline(g, line)) {
            const auto t = split(line);
            if (t.size() < 10) continue;
            imu.push_back({std::stoll(t[1]), {std::stof(t[3]), std::stof(t[4]), std::stof(t[5])},
                           {std::stof(t[6]), std::stof(t[7]), std::stof(t[8])}, std::stof(t[9])});
        }
    } else {
        const vio::EurocData d = vio::load_dataset(dataset);
        for (size_t k = 0; k < d.frames.size(); ++k) frames.push_back({d.frames[k].t_ns, (int64_t)k, d.frames[k].path, 0, 0});
        for (const auto& s : d.imu)
            imu.push_back({s.t_ns, {(float)s.a.x(), (float)s.a.y(), (float)s.a.z()},
                           {(float)s.w.x(), (float)s.w.y(), (float)s.w.z()}, 0.0f});
    }
    if (max_frames > 0 && (long)frames.size() > max_frames) frames.resize(max_frames);
    if (frames.empty() || imu.empty()) { std::cerr << "no frames or no IMU\n"; return 1; }
    const cv::Mat probe = cv::imread(frames.front().path, cv::IMREAD_GRAYSCALE);
    const int W = probe.cols, H = probe.rows;

    enable_signal_handler();
    main_running = 1;
    const int imu_ch = pipe_server_get_next_available_channel();
    pipe_info_t info;
    make_info(info, imu_pipe, "imu_data_t", IMU_RECOMMENDED_PIPE_SIZE);
    if (pipe_server_create(imu_ch, info, 0)) return 1;
    const int cam_ch = pipe_server_get_next_available_channel();
    make_info(info, cam_pipe, "camera_image_metadata_t", 64 * W * H);
    if (pipe_server_create(cam_ch, info, 0)) return 1;
    int out_ch = -1;
    if (!lockstep.empty()) {
        out_ch = pipe_client_get_next_available_channel();
        pipe_client_set_simple_helper_cb(out_ch, out_cb, nullptr);
        pipe_client_open(out_ch, lockstep.c_str(), "gvio-replay", CLIENT_FLAG_EN_SIMPLE_HELPER,
                         EXT_VIO_RECOMMENDED_READ_BUF_SIZE);
    }
    std::cout << "[gvio-replay] " << frames.size() << " frames " << W << "x" << H << ", " << imu.size()
              << " IMU samples, waiting for subscribers on " << imu_pipe << " and " << cam_pipe << "\n";
    while (main_running && (pipe_server_get_num_clients(imu_ch) < 1 || pipe_server_get_num_clients(cam_ch) < 1
                            || (out_ch >= 0 && !pipe_client_is_connected(out_ch))))
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    const int64_t s0 = std::min(frames.front().t_ns, imu.front().t_ns);
    const int64_t shift = live ? (mpa_time_monotonic_ns() - s0) : 0;
    auto send_imu = [&](size_t& j, int64_t up_to) {
        std::vector<imu_data_t> b;
        while (j < imu.size() && imu[j].t_ns <= up_to) {
            imu_data_t d;
            std::memset(&d, 0, sizeof(d));
            d.magic_number = IMU_MAGIC_NUMBER;
            std::memcpy(d.accl_ms2, imu[j].a, sizeof(d.accl_ms2));
            std::memcpy(d.gyro_rad, imu[j].w, sizeof(d.gyro_rad));
            d.temp_c = imu[j].temp;
            d.timestamp_ns = (uint64_t)(imu[j].t_ns + shift);
            b.push_back(d);
            ++j;
        }
        if (!b.empty()) pipe_server_write(imu_ch, b.data(), (int)(b.size() * sizeof(imu_data_t)));
    };
    auto send_frame = [&](const Frame& f, const cv::Mat& img) {
        camera_image_metadata_t m;
        std::memset(&m, 0, sizeof(m));
        m.magic_number = CAMERA_MAGIC_NUMBER;
        m.timestamp_ns = f.t_ns + shift;
        m.frame_id = (int32_t)f.id;
        m.width = (int16_t)img.cols;
        m.height = (int16_t)img.rows;
        m.size_bytes = img.cols * img.rows;
        m.stride = img.cols;
        m.exposure_ns = f.exposure_ns;
        m.gain = f.gain;
        m.format = IMAGE_FORMAT_RAW8;
        m.framerate = 30;
        pipe_server_write_camera_frame(cam_ch, m, img.data);
    };

    const auto t_start = Clock::now();
    size_t j = 0, timeouts = 0;
    if (!lockstep.empty()) {
        for (const auto& f : frames) {
            if (!main_running) break;
            send_imu(j, f.t_ns + 100000000LL);
            const cv::Mat img = cv::imread(f.path, cv::IMREAD_GRAYSCALE);
            send_frame(f, img);
            std::unique_lock<std::mutex> lk(g_m);
            if (!g_cv.wait_for(lk, std::chrono::seconds(5), [&] { return g_last_out_cam_t >= f.t_ns + shift || !main_running; }))
                ++timeouts;
        }
    } else {
        auto at = [&](int64_t t) { return t_start + std::chrono::nanoseconds((int64_t)((t - s0) / rate)); };
        std::thread imu_th([&] {
            size_t k = 0;
            auto next = t_start;
            while (k < imu.size() && main_running) {
                next += std::chrono::microseconds((int64_t)(1000 * imu_batch_ms));
                std::this_thread::sleep_until(next);
                const int64_t now_data = s0 + (int64_t)(std::chrono::duration<double>(Clock::now() - t_start).count() * 1e9 * rate);
                send_imu(k, now_data);
            }
        });
        for (const auto& f : frames) {
            if (!main_running) break;
            const cv::Mat img = cv::imread(f.path, cv::IMREAD_GRAYSCALE);
            std::this_thread::sleep_until(at(f.t_ns));
            send_frame(f, img);
        }
        imu_th.join();
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "[gvio-replay] done in " << std::chrono::duration<double>(Clock::now() - t_start).count()
              << " s, lockstep timeouts " << timeouts << "\n";
    pipe_client_close_all();
    pipe_server_close_all();
    return 0;
}
