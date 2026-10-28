#pragma once
// MpaBridge: the only part of the server that knows libmodal-pipe. It subscribes to the IMU and
// camera pipes, feeds the VioEngine from the MPA helper threads, and publishes every engine output
// as vio_data_t and ext_vio_data_t, ModalAI's VIO interface, so voxl-vision-hub (vio_pipe), voxl-
// inspect-vio and the logger read it like the qvio and ov pipes. Control commands reset_vio_soft and
// reset_vio_hard on either output pipe reset the estimator.
#include "engine.h"
#include "publish_frame.h"
#include "server_config.h"
#include "camera_model.h"

#include <modal_pipe_interfaces.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace gvio {

class MpaBridge {
public:
    MpaBridge(const ServerConfig& sc, VioEngine& engine, const OpenCvIntrinsics& raw_cam);
    ~MpaBridge();

    // Creates the output pipes and opens the inputs (which reconnect in the background, so the
    // server may start before the IMU and camera servers). Returns false on a hard failure.
    bool open();
    void close();
    void print_status() const;

private:
    static void imu_cb(int ch, char* data, int bytes, void* ctx);
    static void cam_cb(int ch, camera_image_metadata_t meta, char* frame, void* ctx);
    static void imu_connect_cb(int ch, void* ctx);
    static void imu_disconnect_cb(int ch, void* ctx);
    static void cam_connect_cb(int ch, void* ctx);
    static void cam_disconnect_cb(int ch, void* ctx);
    static void control_cb(int ch, char* str, int bytes, void* ctx);

    void on_output(const EngineOutput& o);
    void publish(const EngineOutput& o, const PublishedState& s);
    void watchdog_loop();

    const ServerConfig sc_;
    VioEngine& engine_;
    OpenCvIntrinsics raw_cam_;                 // fisheye K, D: feature pixels back to the raw image
    PublishFrame frame_;
    std::mutex pub_m_;
    int imu_ch_ = -1, cam_ch_ = -1, out_ch_ = -1, ext_ch_ = -1;
    std::atomic<bool> imu_connected_{false}, cam_connected_{false};
    std::atomic<int64_t> last_imu_wall_ns_{0}, last_cam_wall_ns_{0}, last_out_wall_ns_{0};
    std::atomic<int64_t> last_frame_id_{-1}, last_cam_t_ns_{0};
    std::atomic<uint64_t> n_published_{0}, n_ok_{0};
    std::atomic<bool> stop_{false};
    std::thread watchdog_;
};

} // namespace gvio
