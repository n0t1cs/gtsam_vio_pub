// voxl-gtsam-vio-server: the GTSAM VIO of cpp_gtsam_vio, online on the VOXL2, on MPA pipes.
//
//   voxl-gtsam-vio-server [-c CONFIG] [-v]
//     -c   server config (default /etc/modalai/voxl-gtsam-vio-server.yaml)
//     -v   status line every 10 s
//
// Subscribes to imu_pipe and camera_pipe, publishes output_pipe (vio_data_t) and
// output_pipe_extended (ext_vio_data_t). See README.md.
#include "camera_model.h"
#include "engine.h"
#include "load.h"
#include "mpa_bridge.h"
#include "params.h"
#include "server_config.h"

#include <modal_pipe.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>

#define PROCESS_NAME "voxl-gtsam-vio-server"

static void usage() {
    std::cout << "usage: " PROCESS_NAME " [-c CONFIG] [-v]\n"
                 "  -c CONFIG  server config (default /etc/modalai/voxl-gtsam-vio-server.yaml)\n"
                 "  -v         status line every 10 s\n";
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    std::string config = "/etc/modalai/voxl-gtsam-vio-server.yaml";
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-c" && i + 1 < argc) config = argv[++i];
        else if (a == "-v") verbose = true;
        else { usage(); return a == "-h" ? 0 : 2; }
    }

    gvio::ServerConfig sc;
    gvio::EngineConfig ec;
    gvio::OpenCvIntrinsics ic;
    try {
        sc = gvio::ServerConfig::load(config);
        verbose = verbose || sc.verbose;
        std::cout << "[config] " << config << "\n" << sc.summary() << "\n";
        ec.P = vio::VioParams::load(sc.estimator_config);

        if (!sc.sensor_yaml.empty()) {
            // Test mode: the frames are a dataset's pinhole images, the dataset's camera is the camera.
            const vio::CamCalib c = vio::load_cam_calib(sc.sensor_yaml);
            ec.undistort.init_passthrough(c.K, c.width, c.height);
            ec.calib = c;
            if (ec.P.ext_T_BC.size() == 16)
                ec.calib.T_BC = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(ec.P.ext_T_BC.data());
            ic.model = "pinhole";
            std::cout << "[camera] TEST MODE: " << sc.sensor_yaml << " (pinhole frames, no remap)\n";
        } else {
        ic = gvio::load_opencv_intrinsics(sc.intrinsics_file);
        Eigen::Vector4d pin;
        if (sc.pinhole_override.size() == 4)
            pin << sc.pinhole_override[0], sc.pinhole_override[1], sc.pinhole_override[2], sc.pinhole_override[3];
        Eigen::Vector4d dist = Eigen::Vector4d::Zero();
        if (ic.model == "fisheye") {
            ec.undistort.init_fisheye(ic, sc.undistort_balance, sc.pinhole_override.empty() ? nullptr : &pin);
        } else {
            // A radtan calibration: no remap, the tracker undistorts its points (k1, k2, p1, p2).
            if (ic.D.total() > 4 && std::abs(ic.D.at<double>(4)) > 0.0)
                throw std::runtime_error("radtan calibration with k3 != 0 is not supported");
            for (int k = 0; k < 4 && k < (int)ic.D.total(); ++k) dist(k) = ic.D.at<double>(k);
            Eigen::Matrix3d K;
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) K(r, c) = ic.K.at<double>(r, c);
            ec.undistort.init_passthrough(K, ic.width, ic.height);
        }
        Eigen::Matrix4d T_BC = gvio::load_voxl_extrinsic(sc.extrinsics_file, sc.imu_frame, sc.camera_name);
        if (ec.P.ext_T_BC.size() == 16) {
            T_BC = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(ec.P.ext_T_BC.data());
            std::cout << "[extrinsics] T_BC OVERRIDDEN by the estimator config's extrinsics block\n";
        }
        ec.calib = gvio::make_cam_calib(ec.undistort, T_BC, 30);
        ec.calib.dist = dist;
        std::cout << "[camera] " << sc.camera_name << " " << ic.model << " " << ic.width << "x" << ic.height
                  << " -> pinhole fu " << ec.calib.K(0, 0) << " fv " << ec.calib.K(1, 1) << " cu "
                  << ec.calib.K(0, 2) << " cv " << ec.calib.K(1, 2) << "\n[camera] T_BC (camera in "
                  << sc.imu_frame << ")\n" << T_BC << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    ec.imu_filter.enable = sc.imu_lowpass;
    ec.imu_filter.order = sc.imu_lowpass_order;
    ec.imu_filter.cutoff_hz = sc.imu_lowpass_cutoff_hz;
    ec.imu_filter.rate_hz = sc.imu_rate_hz;
    ec.max_pending_frames = sc.max_pending_frames;
    ec.imu_wait_timeout_s = sc.imu_wait_timeout_s;
    ec.auto_reset = sc.auto_reset;
    ec.max_velocity_mps = sc.max_velocity_mps;
    ec.min_features = sc.min_features;
    ec.min_features_timeout_s = sc.min_features_timeout_s;
    ec.max_imu_gap_s = sc.max_imu_gap_s;
    ec.max_frame_gap_s = sc.max_frame_gap_s;
    if (!sc.log_dir.empty()) {
        char stamp[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
        ec.log_dir = sc.log_dir + "/" + stamp;
        std::cout << "[log] " << ec.log_dir << "\n";
    }

    if (sc.pid_file && kill_existing_process(PROCESS_NAME, 2.0) < -2) return 1;
    if (enable_signal_handler() == -1) { std::cerr << "ERROR: signal handler\n"; return 1; }
    if (sc.pid_file) make_pid_file(PROCESS_NAME);

    int rc = 0;
    {
        gvio::VioEngine engine(ec);
        gvio::MpaBridge bridge(sc, engine, ic);
        engine.start();
        main_running = 1;
        if (!bridge.open()) {
            std::cerr << "ERROR: cannot open the pipes\n";
            rc = 1;
            main_running = 0;
        } else {
            std::cout << "[" PROCESS_NAME "] running, publishing " << sc.output_pipe << " and "
                      << sc.output_pipe_extended << "\n";
        }
        int tick = 0;
        while (main_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (verbose && ++tick % 10 == 0) bridge.print_status();
        }
        std::cout << "[" PROCESS_NAME "] stopping\n";
        bridge.close();
        engine.stop();
        bridge.print_status();
    }
    pipe_client_close_all();
    pipe_server_close_all();
    if (sc.pid_file) remove_pid_file(PROCESS_NAME);
    return rc;
}
