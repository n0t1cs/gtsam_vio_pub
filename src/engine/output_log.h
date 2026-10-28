#pragma once
// Disk log of an online run, in the offline binary's formats so python/ tools read it unchanged:
//   session<k>/vio_results_est.tum   keyframe poses, t = seconds since the run's first frame (the
//                                    offline convention), estimator world, TUM order
//   session<k>/vio_cov.csv           per keyframe pose + 6x6 covariance (runtime.dump_covariance)
//   session<k>/vio_relcov.csv        relative poses + covariance (runtime.dump_relcov)
//   frames.csv                       every frame: absolute stamp, state, pose at that frame, latency
//   events.log                       init, resets, drops, faults
//   meta.txt                         t0_ns of the time base and the configuration in use
// A session is one initialisation: a reset starts session k+1. Rows are flushed as they are
// written, so a run that is killed still leaves a usable log.
#include <Eigen/Dense>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

namespace gvio {

struct EngineOutput;

class OutputLog {
public:
    // Empty dir = disabled (every call is a no-op).
    explicit OutputLog(const std::string& dir);
    bool enabled() const { return !dir_.empty(); }

    void meta(const std::string& line);
    void event(int64_t t_ns, const std::string& what);
    void start_session(int k, bool cov, bool relcov);
    void kf_pose(double t_s, const Eigen::Matrix4d& T);
    void kf_cov(double t_s, const Eigen::Matrix4d& T, const Eigen::Matrix<double, 6, 6>& C);
    void relcov(double ta, double tb, int span, const Eigen::Matrix4d& T, const Eigen::Matrix<double, 6, 6>& S);
    void frame(const EngineOutput& o);

private:
    std::string dir_;
    std::ofstream meta_, events_, frames_, tum_, cov_, rel_;
};

} // namespace gvio
