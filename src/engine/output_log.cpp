#include "output_log.h"
#include "engine.h"

#include <Eigen/Geometry>

#include <filesystem>
#include <iomanip>

namespace gvio {

namespace fs = std::filesystem;

OutputLog::OutputLog(const std::string& dir) : dir_(dir) {
    if (dir_.empty()) return;
    fs::create_directories(dir_);
    meta_.open(dir_ + "/meta.txt");
    events_.open(dir_ + "/events.log");
    frames_.open(dir_ + "/frames.csv");
    frames_ << "t_ns,frame_id,session,state,error_code,is_kf,n_features,px,py,pz,qw,qx,qy,qz,vx,vy,vz,"
               "bgx,bgy,bgz,bax,bay,baz,latency_ms,kf_update_ms,"
               "remap_ms,fe_clahe_ms,fe_predict_ms,fe_klt_fwd_ms,fe_klt_bwd_ms,fe_reject_ms,fe_detect_ms,"
               "fe_copy_ms,fe_obs_ms,est_ms,add_frame_ms,cov_ms,nav_ms,cpu_sync_ms,cpu_fe_ms,cpu_est_ms\n"
            << std::flush;
}

void OutputLog::meta(const std::string& line) {
    if (enabled()) meta_ << line << "\n" << std::flush;
}

void OutputLog::event(int64_t t_ns, const std::string& what) {
    if (enabled()) events_ << t_ns << " " << what << "\n" << std::flush;
}

void OutputLog::start_session(int k, bool cov, bool relcov) {
    if (!enabled()) return;
    const std::string d = dir_ + "/session" + std::to_string(k);
    fs::create_directories(d);
    tum_.close(); cov_.close(); rel_.close();
    tum_.open(d + "/vio_results_est.tum");
    if (cov) {
        cov_.open(d + "/vio_cov.csv");
        cov_ << "t_s,px,py,pz,qw,qx,qy,qz";
        for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) cov_ << ",c" << r << c;
        cov_ << "\n" << std::setprecision(12) << std::flush;
    }
    if (relcov) {
        rel_.open(d + "/vio_relcov.csv");
        rel_ << "t_a,t_b,span,px,py,pz,qw,qx,qy,qz";
        for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) rel_ << ",c" << r << c;
        rel_ << "\n" << std::setprecision(12) << std::flush;
    }
}

// Same writer as main.cpp evaluate_and_write_csv (write_tum_pose).
void OutputLog::kf_pose(double t_s, const Eigen::Matrix4d& T) {
    if (!tum_.is_open()) return;
    Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
    q.normalize();
    tum_ << std::fixed << std::setprecision(9)
         << t_s << " " << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " "
         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n" << std::flush;
}

// Same rows as main.cpp's vio_cov.csv writer.
void OutputLog::kf_cov(double t_s, const Eigen::Matrix4d& T, const Eigen::Matrix<double, 6, 6>& C) {
    if (!cov_.is_open()) return;
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    cov_ << t_s << ',' << T(0, 3) << ',' << T(1, 3) << ',' << T(2, 3)
         << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) cov_ << ',' << C(r, c);
    cov_ << "\n" << std::flush;
}

void OutputLog::relcov(double ta, double tb, int span, const Eigen::Matrix4d& T,
                       const Eigen::Matrix<double, 6, 6>& S) {
    if (!rel_.is_open()) return;
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    rel_ << ta << ',' << tb << ',' << span
         << ',' << T(0, 3) << ',' << T(1, 3) << ',' << T(2, 3)
         << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) rel_ << ',' << S(r, c);
    rel_ << "\n" << std::flush;
}

void OutputLog::frame(const EngineOutput& o) {
    if (!enabled()) return;
    Eigen::Quaterniond q(Eigen::Matrix3d(o.T_wb.block<3, 3>(0, 0)));
    q.normalize();
    frames_ << o.t_ns << ',' << o.frame_id << ',' << o.session << ',' << o.state << ',' << o.error_code
            << ',' << (o.is_keyframe ? 1 : 0) << ',' << o.n_features << std::setprecision(9)
            << ',' << o.T_wb(0, 3) << ',' << o.T_wb(1, 3) << ',' << o.T_wb(2, 3)
            << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z()
            << ',' << o.v_w.x() << ',' << o.v_w.y() << ',' << o.v_w.z()
            << ',' << o.bg.x() << ',' << o.bg.y() << ',' << o.bg.z()
            << ',' << o.ba.x() << ',' << o.ba.y() << ',' << o.ba.z()
            << std::setprecision(4) << ',' << o.latency_ms << ',' << o.kf_update_ms
            << ',' << o.remap_ms << ',' << o.fe.clahe << ',' << o.fe.predict << ',' << o.fe.klt_fwd
            << ',' << o.fe.klt_bwd << ',' << o.fe.reject << ',' << o.fe.detect << ',' << o.fe.copy
            << ',' << o.fe.obs << ',' << o.est_ms << ',' << o.add_frame_ms << ',' << o.cov_ms << ',' << o.nav_ms
            << std::fixed << std::setprecision(3) << ',' << o.sync_cpu_ms << ',' << o.fe.thread_cpu_ms
            << ',' << o.est_cpu_ms << std::defaultfloat << "\n" << std::flush;
}

void OutputLog::open_cov_jobs() {
    if (!enabled()) return;
    cov_jobs_.open(dir_ + "/cov_jobs.csv");
    cov_jobs_ << "t_ns,session,n_pose,n_rel,wait_ms,work_ms,cpu_cov_ms\n" << std::flush;
}

void OutputLog::cov_job(int64_t t_ns, int session, int n_pose, int n_rel, double wait_ms, double work_ms,
                        double cpu_ms) {
    if (!cov_jobs_.is_open()) return;
    cov_jobs_ << t_ns << ',' << session << ',' << n_pose << ',' << n_rel << std::setprecision(4)
              << ',' << wait_ms << ',' << work_ms << std::fixed << std::setprecision(3) << ',' << cpu_ms
              << std::defaultfloat << "\n" << std::flush;
}

} // namespace gvio
