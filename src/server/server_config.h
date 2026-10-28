#pragma once
// voxl-gtsam-vio-server configuration (default /etc/modalai/voxl-gtsam-vio-server.yaml). Everything
// the estimator itself needs is in the estimator config it points to.
#include <string>
#include <vector>

namespace gvio {

struct ServerConfig {
    std::string estimator_config = "/usr/share/voxl-gtsam-vio/estimator_mantis2.yaml";

    // inputs: a pipe name (under /run/mpa/) or a full path (any location starting with '/')
    std::string imu_pipe    = "imu_apps";
    std::string camera_pipe = "tracking_front_misp_norm";
    // calibration, read from the vehicle's own files
    std::string camera_name     = "tracking_front";   // child in extrinsics.conf
    std::string imu_frame       = "imu_apps";         // parent in extrinsics.conf, the body frame
    std::string intrinsics_file = "/data/modalai/opencv_tracking_front_intrinsics.yml";
    std::string extrinsics_file = "/etc/modalai/extrinsics.conf";
    double undistort_balance = 0.0;
    std::vector<double> pinhole_override;   // [fu, fv, cu, cv], empty = computed from the balance
    long long camera_time_offset_ns = 0;     // added to every frame stamp. 0 = the metadata stamp
                                             // (start of exposure), as in the offline datasets
    std::string sensor_yaml;                 // TESTS: a EuRoC cam0/sensor.yaml replaces both calibration
                                             // files (pinhole frames, no remap), for dataset replays

    // IMU low-pass (the _lpf datasets' filter)
    bool   imu_lowpass = true;
    int    imu_lowpass_order = 4;
    double imu_lowpass_cutoff_hz = 25.0;
    double imu_rate_hz = 0.0;                // 0 = measured at start

    // outputs
    std::string output_pipe          = "gvio";            // vio_data_t
    std::string output_pipe_extended = "gvio_extended";   // ext_vio_data_t
    bool soft_reset_keeps_frame = true;      // after an automatic reset, continue the published frame

    // real time and health
    int    max_pending_frames = 3;
    double imu_wait_timeout_s = 0.3;
    bool   auto_reset = true;
    double max_velocity_mps = 20.0;
    int    min_features = 1;
    double min_features_timeout_s = 3.0;
    double max_imu_gap_s = 0.1;
    double max_frame_gap_s = 1.0;
    double input_timeout_s = 1.0;            // no output for this long: publish FAILED (missing input)

    // housekeeping
    std::string log_dir = "/data/gvio";      // a timestamped run folder per start; empty = no log
    bool pid_file = true;                    // /run/voxl-gtsam-vio-server.pid, kill a running copy
    bool verbose = false;

    static ServerConfig load(const std::string& path);
    std::string summary() const;
};

} // namespace gvio
