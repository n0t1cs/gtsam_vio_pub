#include "publish_frame.h"

#include <cmath>

namespace gvio {

static double heading(const Eigen::Matrix3d& R) { return std::atan2(R(1, 0), R(0, 0)); }

static Eigen::Matrix3d rot_z(double a) {
    return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

void PublishFrame::upper21(const Eigen::Matrix<double, 6, 6>& P, float out[21]) {
    int k = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) out[k++] = (float)P(i, j);
}

Eigen::Matrix<double, 6, 6> PublishFrame::to_xyzrpy(const Eigen::Matrix<double, 6, 6>& C, const Eigen::Matrix3d& R) {
    Eigen::Matrix<double, 6, 6> P;
    const Eigen::Matrix3d Crr = C.block<3, 3>(0, 0), Crt = C.block<3, 3>(0, 3);
    const Eigen::Matrix3d Ctr = C.block<3, 3>(3, 0), Ctt = C.block<3, 3>(3, 3);
    P.block<3, 3>(0, 0) = R * Ctt * R.transpose();
    P.block<3, 3>(0, 3) = R * Ctr * R.transpose();
    P.block<3, 3>(3, 0) = R * Crt * R.transpose();
    P.block<3, 3>(3, 3) = R * Crr * R.transpose();
    return P;
}

PublishedState PublishFrame::convert(const EngineOutput& o) {
    PublishedState s;
    if (o.session != session_) {                     // a new initialisation follows
        session_ = o.session;
        aligned_ = false;
        have_cov_ = false;
    }
    if (o.state != STATE_OK) return s;

    const Eigen::Matrix3d Rw = o.T_wb.block<3, 3>(0, 0);
    const Eigen::Vector3d pw = o.T_wb.block<3, 1>(0, 3);
    if (!aligned_) {
        const Eigen::Matrix3d flip = Eigen::Vector3d(1, -1, -1).asDiagonal();
        const double yaw0 = heading(flip * Rw);
        const bool cont = keep_ && have_last_ && !o.session_hard && o.session > 0;
        M_ = rot_z((cont ? last_yaw_ : 0.0) - yaw0) * flip;
        off_ = (cont ? last_p_ : Eigen::Vector3d::Zero()) - M_ * pw;
        aligned_ = true;
    }
    s.R = M_ * Rw;
    s.p = M_ * pw + off_;
    s.v = M_ * o.v_w;
    s.gravity = Eigen::Vector3d(0, 0, 1);            // M maps the estimator's -z gravity onto +z
    if (o.has_cov && o.cov.allFinite()) {
        upper21(to_xyzrpy(o.cov, s.R), last_cov_);
        have_cov_ = true;
    }
    s.cov_valid = have_cov_;
    if (have_cov_) std::copy(last_cov_, last_cov_ + 21, s.pose_cov21);
    s.valid = s.R.allFinite() && s.p.allFinite() && s.v.allFinite();
    if (s.valid) {
        have_last_ = true;
        last_p_ = s.p;
        last_yaw_ = heading(s.R);
    }
    return s;
}

} // namespace gvio
