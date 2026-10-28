#include "mpa_bridge.h"

#include <modal_pipe.h>
#include <cJSON.h>
#include <opencv2/calib3d.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace gvio {

#define SERVER_NAME "voxl-gtsam-vio-server"
static const char* kControlCommands = RESET_VIO_SOFT "," RESET_VIO_HARD;
static constexpr int kImuPipeSize = 1 * 1024 * 1024;     // QVIO's sizes: big buffers, never drop on the pipe
static constexpr int kCamPipeSize = 256 * 1024 * 1024;

static int64_t wall_ns() { return mpa_time_monotonic_ns(); }

// A pipe given by name lives under MODAL_PIPE_DEFAULT_BASE_DIR, a full path (desktop tests) as is.
static void make_info(pipe_info_t& info, const std::string& name_or_path, const char* type, int size) {
    std::string name = name_or_path, loc;
    if (!name_or_path.empty() && name_or_path[0] == '/') {
        loc = name_or_path.back() == '/' ? name_or_path : name_or_path + "/";
        std::string t = loc.substr(0, loc.size() - 1);
        name = t.substr(t.find_last_of('/') + 1);
    } else {
        loc = std::string(MODAL_PIPE_DEFAULT_BASE_DIR) + name_or_path + "/";
    }
    std::memset(&info, 0, sizeof(info));
    std::strncpy(info.name, name.c_str(), MODAL_PIPE_MAX_NAME_LEN - 1);
    std::strncpy(info.location, loc.c_str(), MODAL_PIPE_MAX_DIR_LEN - 1);
    std::strncpy(info.type, type, MODAL_PIPE_MAX_TYPE_LEN - 1);
    std::strncpy(info.server_name, SERVER_NAME, MODAL_PIPE_MAX_NAME_LEN - 1);
    info.size_bytes = size;
}

MpaBridge::MpaBridge(const ServerConfig& sc, VioEngine& engine, const OpenCvIntrinsics& raw_cam)
    : sc_(sc), engine_(engine), raw_cam_(raw_cam), frame_(sc.soft_reset_keeps_frame) {}

MpaBridge::~MpaBridge() { close(); }

bool MpaBridge::open() {
    // outputs
    out_ch_ = pipe_server_get_next_available_channel();
    pipe_info_t info;
    make_info(info, sc_.output_pipe, "vio_data_t", VIO_RECOMMENDED_PIPE_SIZE);
    if (pipe_server_create(out_ch_, info, SERVER_FLAG_EN_CONTROL_PIPE)) return false;
    ext_ch_ = pipe_server_get_next_available_channel();
    make_info(info, sc_.output_pipe_extended, "ext_vio_data_t", VIO_RECOMMENDED_PIPE_SIZE);
    if (pipe_server_create(ext_ch_, info, SERVER_FLAG_EN_CONTROL_PIPE)) return false;
    for (int ch : {out_ch_, ext_ch_}) {
        cJSON* j = pipe_server_get_info_json_ptr(ch);
        cJSON_AddStringToObject(j, "imu", sc_.imu_pipe.c_str());
        cJSON_AddStringToObject(j, "cam", sc_.camera_pipe.c_str());
        cJSON_AddStringToObject(j, "estimator", "GTSAM iSAM2 fixed-lag VIO (cpp_gtsam_vio)");
        pipe_server_update_info(ch);
        pipe_server_set_control_cb(ch, &MpaBridge::control_cb, this);
        pipe_server_set_available_control_commands(ch, kControlCommands);
    }
    engine_.set_output_callback([this](const EngineOutput& o) { on_output(o); });

    // inputs (auto-reconnect: open succeeds while the servers are still starting)
    imu_ch_ = pipe_client_get_next_available_channel();
    pipe_client_set_connect_cb(imu_ch_, &MpaBridge::imu_connect_cb, this);
    pipe_client_set_disconnect_cb(imu_ch_, &MpaBridge::imu_disconnect_cb, this);
    pipe_client_set_simple_helper_cb(imu_ch_, &MpaBridge::imu_cb, this);
    if (pipe_client_open(imu_ch_, sc_.imu_pipe.c_str(), SERVER_NAME, CLIENT_FLAG_EN_SIMPLE_HELPER,
                         IMU_RECOMMENDED_READ_BUF_SIZE)) {
        std::cerr << "ERROR: cannot open the IMU pipe " << sc_.imu_pipe << "\n";
        return false;
    }
    cam_ch_ = pipe_client_get_next_available_channel();
    pipe_client_set_connect_cb(cam_ch_, &MpaBridge::cam_connect_cb, this);
    pipe_client_set_disconnect_cb(cam_ch_, &MpaBridge::cam_disconnect_cb, this);
    pipe_client_set_camera_helper_cb(cam_ch_, &MpaBridge::cam_cb, this);
    if (pipe_client_open(cam_ch_, sc_.camera_pipe.c_str(), SERVER_NAME, CLIENT_FLAG_EN_CAMERA_HELPER, 0)) {
        std::cerr << "ERROR: cannot open the camera pipe " << sc_.camera_pipe << "\n";
        return false;
    }
    last_out_wall_ns_ = wall_ns();
    watchdog_ = std::thread(&MpaBridge::watchdog_loop, this);
    return true;
}

void MpaBridge::close() {
    stop_ = true;
    if (watchdog_.joinable()) watchdog_.join();
    if (imu_ch_ >= 0) pipe_client_close(imu_ch_);
    if (cam_ch_ >= 0) pipe_client_close(cam_ch_);
    imu_ch_ = cam_ch_ = -1;
    std::lock_guard<std::mutex> lk(pub_m_);
    if (out_ch_ >= 0) pipe_server_close(out_ch_);
    if (ext_ch_ >= 0) pipe_server_close(ext_ch_);
    out_ch_ = ext_ch_ = -1;
}

// ------------------------------------------------------------------ inputs
void MpaBridge::imu_cb(int, char* data, int bytes, void* ctx) {
    auto* b = static_cast<MpaBridge*>(ctx);
    int n = 0;
    imu_data_t* d = pipe_validate_imu_data_t(data, bytes, &n);
    if (!d || n <= 0) return;
    b->imu_connected_ = true;
    b->last_imu_wall_ns_ = wall_ns();
    for (int i = 0; i < n; ++i) {
        vio::ImuSample s;
        s.t_ns = (int64_t)d[i].timestamp_ns;
        s.w = Eigen::Vector3d(d[i].gyro_rad[0], d[i].gyro_rad[1], d[i].gyro_rad[2]);
        s.a = Eigen::Vector3d(d[i].accl_ms2[0], d[i].accl_ms2[1], d[i].accl_ms2[2]);
        b->engine_.push_imu(s);
    }
}

void MpaBridge::cam_cb(int, camera_image_metadata_t meta, char* frame, void* ctx) {
    auto* b = static_cast<MpaBridge*>(ctx);
    b->cam_connected_ = true;
    b->last_cam_wall_ns_ = wall_ns();
    // Tracking cameras send RAW8. NV12 and NV21 start with the full-resolution Y plane.
    if (meta.format != IMAGE_FORMAT_RAW8 && meta.format != IMAGE_FORMAT_NV12 && meta.format != IMAGE_FORMAT_NV21) {
        b->engine_.flag_input_error(err::CAM_BAD_FORMAT);
        return;
    }
    const int stride = meta.stride > 0 ? meta.stride : meta.width;
    const cv::Mat view(meta.height, meta.width, CV_8UC1, frame, (size_t)stride);
    const int64_t t = meta.timestamp_ns + b->sc_.camera_time_offset_ns;
    b->last_frame_id_ = meta.frame_id;
    b->last_cam_t_ns_ = t;
    b->engine_.push_image(t, meta.frame_id, view);   // copied inside
}

void MpaBridge::imu_connect_cb(int ch, void*) {
    pipe_client_set_pipe_size(ch, kImuPipeSize);
    std::cout << "connected to the IMU server\n";
}
void MpaBridge::imu_disconnect_cb(int, void* ctx) {
    auto* b = static_cast<MpaBridge*>(ctx);
    b->imu_connected_ = false;
    std::cerr << "WARNING: disconnected from the IMU server, resetting\n";
    b->engine_.request_reset(true);
}
void MpaBridge::cam_connect_cb(int ch, void*) {
    pipe_client_set_pipe_size(ch, kCamPipeSize);
    std::cout << "connected to the camera server\n";
}
void MpaBridge::cam_disconnect_cb(int, void* ctx) {
    auto* b = static_cast<MpaBridge*>(ctx);
    b->cam_connected_ = false;
    std::cerr << "WARNING: disconnected from the camera server, resetting\n";
    b->engine_.request_reset(true);
}

void MpaBridge::control_cb(int, char* str, int bytes, void* ctx) {
    auto* b = static_cast<MpaBridge*>(ctx);
    const std::string cmd(str, strnlen(str, (size_t)bytes));
    if (cmd.rfind(RESET_VIO_HARD, 0) == 0) { std::cout << "control: hard reset\n"; b->engine_.request_reset(true); }
    else if (cmd.rfind(RESET_VIO_SOFT, 0) == 0) { std::cout << "control: soft reset\n"; b->engine_.request_reset(false); }
    else std::cerr << "control: unknown command '" << cmd << "'\n";
}

// ------------------------------------------------------------------ outputs
void MpaBridge::on_output(const EngineOutput& o) {
    const PublishedState s = frame_.convert(o);
    publish(o, s);
}

void MpaBridge::publish(const EngineOutput& o, const PublishedState& s) {
    ext_vio_data_t e;
    std::memset(&e, 0, sizeof(e));
    vio_data_t& d = e.v;
    d.magic_number = VIO_MAGIC_NUMBER;
    d.timestamp_ns = o.t_ns;
    d.state = (s.valid ? VIO_STATE_OK : (o.state == STATE_OK ? VIO_STATE_FAILED : (uint8_t)o.state));
    d.quality = (d.state == VIO_STATE_OK) ? std::min(100, std::max(1, o.n_features)) : -1;
    d.error_code = o.error_code;
    d.n_feature_points = (uint16_t)std::min(o.n_features, 65535);
    d.frame = VIO_FRAME_BODY_ALIGNED;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    d.pose_covariance[0] = nan;
    d.velocity_covariance[0] = nan;
    if (s.valid) {
        for (int i = 0; i < 3; ++i) {
            d.T_imu_wrt_vio[i] = (float)s.p(i);
            d.vel_imu_wrt_vio[i] = (float)s.v(i);
            d.imu_angular_vel[i] = (float)o.w_body(i);
            d.gravity_vector[i] = (float)s.gravity(i);
            for (int j = 0; j < 3; ++j) d.R_imu_to_vio[i][j] = (float)s.R(i, j);
        }
        if (s.cov_valid) std::memcpy(d.pose_covariance, s.pose_cov21, sizeof(d.pose_covariance));
    }
    const Eigen::Matrix4d& T_BC = engine_.config().calib.T_BC;
    for (int i = 0; i < 3; ++i) {
        d.T_cam_wrt_imu[i] = (float)T_BC(i, 3);
        for (int j = 0; j < 3; ++j) d.R_cam_to_imu[i][j] = (float)T_BC(i, j);
    }
    e.last_cam_frame_id = (int32_t)o.frame_id;
    e.last_cam_timestamp_ns = o.t_ns;
    e.imu_cam_time_shift_s = 0.0f;
    for (int i = 0; i < 3; ++i) { e.gyro_bias[i] = (float)o.bg(i); e.accl_bias[i] = (float)o.ba(i); }
    e.gravity_covariance[0][0] = nan;

    // Tracked features. Pixels are reported in the RAW camera image (voxl-portal overlays draw on
    // it), so pinhole pixels are mapped back through the fisheye model when the frames are remapped.
    const int nf = std::min((int)o.features.size(), VIO_MAX_REPORTED_FEATURES);
    e.n_total_features = (uint32_t)o.features.size();
    std::vector<cv::Point2d> norm(nf), raw;
    const auto& K = engine_.config().calib.K;
    for (int i = 0; i < nf; ++i)
        norm[i] = cv::Point2d((o.features[i].second.x() - K(0, 2)) / K(0, 0), (o.features[i].second.y() - K(1, 2)) / K(1, 1));
    if (nf > 0 && engine_.config().undistort.remaps()) cv::fisheye::distortPoints(norm, raw, raw_cam_.K, raw_cam_.D);
    for (int i = 0; i < nf; ++i) {
        vio_feature_t f;                              // ext_vio_data_t is packed: fill a copy
        std::memset(&f, 0, sizeof(f));
        f.id = (uint32_t)o.features[i].first;
        f.cam_id = 0;
        const cv::Point2d px = raw.empty() ? cv::Point2d(o.features[i].second.x(), o.features[i].second.y()) : raw[i];
        f.pix_loc[0] = (float)px.x;
        f.pix_loc[1] = (float)px.y;
        f.depth = nan;
        f.depth_error_stddev = nan;
        f.point_quality = MEDIUM;
        std::memcpy(&e.features[i], &f, sizeof(f));
    }

    std::lock_guard<std::mutex> lk(pub_m_);
    if (out_ch_ < 0) return;
    pipe_server_write(out_ch_, &d, sizeof(vio_data_t));
    pipe_server_write(ext_ch_, &e, sizeof(ext_vio_data_t));
    last_out_wall_ns_ = wall_ns();
    ++n_published_;
    if (d.state == VIO_STATE_OK) ++n_ok_;
}

// No output for input_timeout_s: say why at 5 Hz (vision-hub then reports VIO as failed).
void MpaBridge::watchdog_loop() {
    while (!stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const int64_t now = wall_ns();
        if ((now - last_out_wall_ns_) * 1e-9 < sc_.input_timeout_s) continue;
        uint32_t e = 0;
        if (!imu_connected_ || (now - last_imu_wall_ns_) * 1e-9 > sc_.input_timeout_s) e |= err::IMU_MISSING;
        if (!cam_connected_ || (now - last_cam_wall_ns_) * 1e-9 > sc_.input_timeout_s) e |= err::CAM_MISSING;
        if (!e) e = err::STALLED;
        EngineOutput o;
        o.t_ns = now;
        o.state = STATE_FAILED;
        o.error_code = e;
        publish(o, PublishedState{});
        last_out_wall_ns_ = now - (int64_t)(1e9 * sc_.input_timeout_s);   // keep the 5 Hz cadence
    }
}

void MpaBridge::print_status() const {
    const auto s = engine_.stats();
    std::cout << "[status] imu " << (imu_connected_ ? "up" : "down") << " cam " << (cam_connected_ ? "up" : "down")
              << " | frames in " << s.frames_in << " out " << s.frames_out << " ok " << n_ok_
              << " kf " << s.keyframes << " dropped " << (s.dropped_backlog + s.dropped_no_imu)
              << " resets " << s.resets << " | imu " << s.imu_in << " @ " << s.imu_rate_hz << " Hz, lpf delay "
              << s.imu_tau_ms << " ms\n";
}

} // namespace gvio
