#include "server_config.h"

#include <yaml-cpp/yaml.h>

#include <sstream>
#include <stdexcept>

namespace gvio {

template <typename T>
static void rd(const YAML::Node& n, const char* key, T& v) {
    if (n && n[key]) v = n[key].as<T>();
}

ServerConfig ServerConfig::load(const std::string& path) {
    ServerConfig c;
    YAML::Node r;
    try {
        r = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("cannot read server config " + path + ": " + e.what());
    }
    rd(r, "estimator_config", c.estimator_config);
    rd(r, "imu_pipe", c.imu_pipe);
    rd(r, "camera_pipe", c.camera_pipe);
    rd(r, "camera_name", c.camera_name);
    rd(r, "imu_frame", c.imu_frame);
    rd(r, "intrinsics_file", c.intrinsics_file);
    rd(r, "extrinsics_file", c.extrinsics_file);
    rd(r, "undistort_balance", c.undistort_balance);
    rd(r, "pinhole_override", c.pinhole_override);
    rd(r, "camera_time_offset_ns", c.camera_time_offset_ns);
    // Removed: the IMU low-pass (the raw IMU since V11) and the sensor_yaml test mode. A config
    // that still enables either would run as something else, so it is refused.
    bool lpf = false;
    rd(r["imu_lowpass"], "enable", lpf);
    if (lpf) throw std::runtime_error("imu_lowpass was removed (the server uses the raw IMU): delete the block");
    if (r["sensor_yaml"]) throw std::runtime_error("sensor_yaml (the dataset test mode) was removed");
    rd(r, "output_pipe", c.output_pipe);
    rd(r, "output_pipe_extended", c.output_pipe_extended);
    rd(r, "soft_reset_keeps_frame", c.soft_reset_keeps_frame);
    rd(r, "opencv_threads", c.opencv_threads);
    const YAML::Node rt = r["realtime"];
    rd(rt, "max_pending_frames", c.max_pending_frames);
    rd(rt, "imu_wait_timeout_s", c.imu_wait_timeout_s);
    rd(rt, "async_covariance", c.async_covariance);
    const YAML::Node h = r["health"];
    rd(h, "auto_reset", c.auto_reset);
    rd(h, "max_velocity_mps", c.max_velocity_mps);
    rd(h, "min_features", c.min_features);
    rd(h, "min_features_timeout_s", c.min_features_timeout_s);
    rd(h, "max_imu_gap_s", c.max_imu_gap_s);
    rd(h, "max_frame_gap_s", c.max_frame_gap_s);
    rd(h, "input_timeout_s", c.input_timeout_s);
    rd(r, "log_dir", c.log_dir);
    rd(r, "pid_file", c.pid_file);
    rd(r, "verbose", c.verbose);
    if (!c.pinhole_override.empty() && c.pinhole_override.size() != 4)
        throw std::runtime_error("pinhole_override needs 4 values [fu, fv, cu, cv]");
    return c;
}

std::string ServerConfig::summary() const {
    std::ostringstream s;
    s << "estimator_config " << estimator_config << "\n"
      << "imu_pipe " << imu_pipe << ", camera_pipe " << camera_pipe << " (" << camera_name
      << " in " << imu_frame << ")\n"
      << "intrinsics " << intrinsics_file << ", extrinsics " << extrinsics_file
      << ", balance " << undistort_balance << (pinhole_override.empty() ? "" : ", pinhole PINNED") << "\n"
      << "raw IMU, camera_time_offset_ns " << camera_time_offset_ns << "\n"
      << "output " << output_pipe << " + " << output_pipe_extended << ", auto_reset "
      << (auto_reset ? "on" : "off") << ", log_dir '" << log_dir << "', opencv_threads " << opencv_threads
      << ", covariance " << (async_covariance ? "async" : "inline");
    return s.str();
}

} // namespace gvio
