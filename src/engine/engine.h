#pragma once
// =============================================================================
// VioEngine: the online estimator, independent of the transport.
//
// The offline driver (cpp_gtsam_vio/main.cpp) loads a whole sequence, tracks every frame on a
// front-end thread, then runs the initialisation search and the backend frame by frame. Its
// estimator is causal, so the same per-frame logic runs here on a live stream. Producers (the MPA
// callbacks in the server, or the replay harness) push raw IMU samples and camera frames from any
// thread. Four threads do the rest:
//
//   producer (IMU)    raw sample -> ImuFilter (the _lpf low-pass, online) -> IMU fifo
//   producer (camera) frame copy -> pending frames (bounded, drop oldest unless lockstep)
//   sync              waits until the filtered IMU passes a frame's stamp, then hands every IMU
//                     sample up to that stamp to the front-end and the estimator BEFORE the frame,
//                     undistorts it and queues it: the offline producer's order, exactly
//   FrontendThread    the vendored KLT tracker, unchanged
//   estimator         main.cpp Phase 1 (init search, VINS-style defer-until-well-conditioned) and
//                     Phase 3 (integrate_imu_batch, detect_stationary, add_frame) per frame, plus
//                     the health checks and resets that an online system needs
//
// Given the same inputs the estimate is the offline one: same helpers (offline_helpers.cpp), same
// call order, same RNG seeding, same covariance queries (README, Validation).
// =============================================================================
#include "camera_model.h"
#include "frame_queue.h"
#include "frontend_thread.h"
#include "imu_filter.h"
#include "init_align.h"
#include "init_buffer.h"
#include "output_log.h"
#include "params.h"
#include "vio_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace gvio {

// States and error bits with the numeric values of libmodal-pipe's vio_data_t.h, so the MPA bridge
// copies them through unchanged.
enum EngineState : int { STATE_FAILED = 0, STATE_INITIALIZING = 1, STATE_OK = 2 };
namespace err {
constexpr uint32_t COVARIANCE     = 1u << 0;    // estimator exception or non-finite state
constexpr uint32_t NO_FEATURES    = 1u << 4;    // fewer than min_features for min_features_timeout_s
constexpr uint32_t VEL_INST_CERT  = 1u << 7;    // velocity above max_velocity_mps
constexpr uint32_t DROPPED_IMU    = 1u << 10;   // IMU gap inside a frame interval
constexpr uint32_t DROPPED_CAM    = 1u << 13;   // frame dropped (backlog or no IMU coverage)
constexpr uint32_t BAD_TIMESTAMP  = 1u << 15;   // non-increasing frame or IMU stamp
constexpr uint32_t IMU_MISSING    = 1u << 16;
constexpr uint32_t CAM_MISSING    = 1u << 17;
constexpr uint32_t CAM_BAD_RES    = 1u << 18;
constexpr uint32_t CAM_BAD_FORMAT = 1u << 19;
constexpr uint32_t STALLED        = 1u << 21;   // camera gap above max_frame_gap_s
}

struct EngineConfig {
    vio::VioParams P;              // the estimator config (config/estimator_mantis2.yaml)
    Undistorter    undistort;      // raw frame -> the pinhole camera of `calib`
    vio::CamCalib  calib;          // what the estimator sees (make_cam_calib)
    ImuFilter::Params imu_filter;

    // real time
    bool   lockstep = false;           // replay harness: nothing is dropped, producers block
    int    max_pending_frames = 3;     // frames waiting for IMU coverage, the oldest is dropped beyond
    double imu_wait_timeout_s = 0.3;   // a frame still not covered by IMU after this is dropped
    int    img_queue = 2;              // sync -> front-end
    int    obs_queue = 4;              // front-end -> estimator

    // health, defaults from the vehicle's voxl-open-vins-server.conf where it has one
    bool   auto_reset = true;              // en_auto_reset
    double max_velocity_mps = 20.0;        // auto_reset_max_velocity
    int    min_features = 1;               // auto_reset_min_features
    double min_features_timeout_s = 3.0;   // auto_reset_min_feature_timeout_s
    double max_imu_gap_s = 0.1;            // largest IMU gap tolerated inside one frame interval
    double max_frame_gap_s = 1.0;          // largest camera gap tolerated

    std::string log_dir;               // empty: no disk log
};

// One output per processed frame, in the ESTIMATOR world (z up, gravity -z, yaw from the
// initialisation, the frame of vio_results_est.tum). The MPA bridge converts it for the pipes.
struct EngineOutput {
    int64_t  t_ns = 0;             // frame stamp
    int64_t  frame_id = -1;
    int      session = 0;          // resets so far
    bool     session_hard = false; // this session began with a HARD reset (the published frame restarts)
    int      state = STATE_INITIALIZING;
    uint32_t error_code = 0;
    bool     is_keyframe = false;
    int      n_features = 0;
    Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();   // IMU (= body) pose at this frame
    Eigen::Vector3d v_w  = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg   = Eigen::Vector3d::Zero(), ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d w_body = Eigen::Vector3d::Zero();      // last filtered gyro minus bias
    bool has_cov = false;          // keyframes only, when runtime.dump_covariance
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();   // GTSAM [rot, trans]
    std::vector<std::pair<int64_t, Eigen::Vector2d>> features;           // id, pinhole pixel
    double latency_ms = 0.0;       // engine entry of the frame -> this output, wall clock
    double kf_update_ms = 0.0;     // add_frame time on a keyframe
};

class VioEngine {
public:
    explicit VioEngine(EngineConfig cfg);
    ~VioEngine();

    void set_output_callback(std::function<void(const EngineOutput&)> cb) { on_output_ = std::move(cb); }
    void start();

    // Producers, callable from any thread (one thread per stream).
    void push_imu(const vio::ImuSample& raw);
    // gray: 8-bit, the calibrated size. Copied.
    void push_image(int64_t t_ns, int64_t frame_id, const cv::Mat& gray);
    // Producer-side faults (wrong format or size), reported in the next output.
    void flag_input_error(uint32_t bits) { input_err_.fetch_or(bits); }

    // No more IMU will come (a replay that has pushed its last sample): frames past the last
    // sample are then processed with the IMU there is, as the offline driver does, instead of
    // waiting for coverage. A lockstep producer must call it before pushing those frames.
    void imu_finished();
    // End of input: everything still processable is processed, then the threads stop.
    void finish();
    // Immediate stop.
    void stop();
    // Control pipe: a hard reset restarts the published frame at the origin, a soft one continues it.
    void request_reset(bool hard) { (hard ? reset_hard_ : reset_soft_).store(true); }

    struct Stats {
        uint64_t imu_in = 0, imu_rejected = 0, frames_in = 0, frames_rejected = 0;
        uint64_t dropped_backlog = 0, dropped_no_imu = 0, frames_out = 0, keyframes = 0;
        int init_attempts = 0, resets = 0, imu_gaps = 0;
        double imu_rate_hz = 0.0, imu_tau_ms = 0.0;
    };
    Stats stats() const;
    const EngineConfig& config() const { return cfg_; }

private:
    struct Pending { int64_t t_ns; int64_t frame_id; cv::Mat img; std::chrono::steady_clock::time_point entry; };
    struct FrameMeta { int64_t frame_id; std::chrono::steady_clock::time_point entry; };

    void sync_loop();
    void estimator_loop();
    void process(const vio::StampedObservation& so);
    void run_init(double t_s, const vio::Observation& obs, EngineOutput& out);
    void start_backend(vio::AlignedInit& aligned);
    void run_online(double t_s, const vio::Observation& obs, EngineOutput& out);
    void reset(bool hard, int64_t t_ns, const std::string& why);
    void trim_imu();
    void push_cov(int idx, double t_s, const Eigen::Matrix4d& T, EngineOutput* out);
    void push_relcov(int ib, double tb);

    EngineConfig cfg_;
    std::function<void(const EngineOutput&)> on_output_;
    OutputLog log_;

    // producers -> sync
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<vio::ImuSample> imu_fifo_;          // filtered, not yet handed out
    int64_t last_imu_t_ = -1;                       // newest filtered stamp
    std::deque<Pending> pending_;
    int64_t last_frame_t_ = -1;
    bool input_done_ = false, imu_done_ = false, stop_ = false;
    mutable std::mutex imu_filter_m_;
    ImuFilter imu_filter_;
    std::atomic<uint32_t> input_err_{0};

    // sync -> front-end -> estimator
    vio::ThreadSafeQueue<vio::StampedImage>       img_q_;
    vio::ThreadSafeQueue<vio::StampedObservation> obs_q_;
    vio::ThreadSafeQueue<vio::ImuSample>          fe_imu_q_;
    std::unique_ptr<vio::FrontendThread> frontend_;
    std::mutex est_imu_m_;
    std::deque<vio::ImuSample> est_imu_;
    std::mutex meta_m_;
    std::map<int64_t, FrameMeta> meta_;
    std::thread sync_th_, est_th_;
    std::shared_ptr<void> tbb_pin_;                 // tbb::global_control when built with TBB

    // estimator thread state (main.cpp's locals)
    vio::EurocData data_;                           // cam_calib, frames = {t0}, imu = the stream
    int64_t t0_ns_ = -1;
    int frame_idx_ = -1;
    vio::InitBuffer buf_;
    std::unique_ptr<vio::VioBackend> backend_;
    double prev_t_s_ = 0.0;
    vio::Observation prev_obs_;
    bool running_ = false;                          // backend initialised
    bool latched_fail_ = false;                     // failed with auto_reset off
    int session_ = 0;
    bool session_hard_ = false;
    double low_feat_since_s_ = -1.0;
    uint32_t sticky_err_ = 0;                       // reported once, then cleared
    std::vector<std::pair<int, double>> kf_hist_;   // (state index, t_s) of online keyframes
    std::atomic<bool> reset_hard_{false}, reset_soft_{false};

    // stats
    mutable std::mutex stats_m_;
    Stats stats_;
};

} // namespace gvio
