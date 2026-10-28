#include "imu_preintegrate.h"
#include <cmath>

namespace vio {

PreintResult::PreintResult()
    : dR(Eigen::Matrix3d::Identity()),
      dV(Eigen::Vector3d::Zero()),
      dP(Eigen::Vector3d::Zero()),
      dt(0.0),
      J_g_R(Eigen::Matrix3d::Zero()),
      J_g_V(Eigen::Matrix3d::Zero()),
      J_g_P(Eigen::Matrix3d::Zero()),
      J_a_V(Eigen::Matrix3d::Zero()),
      J_a_P(Eigen::Matrix3d::Zero()) {}

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d K;
    K <<     0, -v.z(),  v.y(),
         v.z(),     0,  -v.x(),
        -v.y(),  v.x(),     0;
    return K;
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& phi) {
    double th = phi.norm();
    if (th < 1e-9) {
        return Eigen::Matrix3d::Identity() + skew(phi);
    }
    Eigen::Vector3d a = phi / th;
    Eigen::Matrix3d K = skew(a);
    return Eigen::Matrix3d::Identity()
           + std::sin(th) * K
           + (1.0 - std::cos(th)) * (K * K);
}

Eigen::Matrix3d so3_jr(const Eigen::Vector3d& phi) {
    double th = phi.norm();
    if (th < 1e-9) {
        return Eigen::Matrix3d::Identity() - 0.5 * skew(phi);
    }
    Eigen::Vector3d a = phi / th;
    Eigen::Matrix3d K = skew(a);
    return Eigen::Matrix3d::Identity()
           - ((1.0 - std::cos(th)) / th) * K
           + ((th - std::sin(th)) / th) * (K * K);
}

Eigen::Vector3d rot_to_vec(const Eigen::Matrix3d& R) {
    double c = (R.trace() - 1.0) * 0.5;
    if (c >  1.0) c =  1.0;
    if (c < -1.0) c = -1.0;
    double theta = std::acos(c);
    Eigen::Vector3d v(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    if (std::abs(theta) < 1e-7) return 0.5 * v;
    return (theta / (2.0 * std::sin(theta))) * v;
}

PreintResult imu_preintegrate(const std::vector<double>& ts_in,
                              const std::vector<Eigen::Vector3d>& gyro_in,
                              const std::vector<Eigen::Vector3d>& accel_in,
                              const Eigen::Vector3d& b_g,
                              const Eigen::Vector3d& b_a,
                              std::optional<double> t_start) {
    std::vector<double> ts = ts_in;
    std::vector<Eigen::Vector3d> gyro = gyro_in;
    std::vector<Eigen::Vector3d> accel = accel_in;

    if (t_start.has_value() && !ts.empty() && ts.front() > *t_start) {
        ts.insert(ts.begin(), *t_start);
        gyro.insert(gyro.begin(), gyro.front());
        accel.insert(accel.begin(), accel.front());
    }

    PreintResult res;
    const int M = (int)ts.size();
    if (M < 2) return res;

    for (int k = 0; k < M - 1; ++k) {
        double dt = ts[k + 1] - ts[k];
        if (dt <= 0) continue;

        // Mid-point average, bias-corrected
        Eigen::Vector3d w = 0.5 * (gyro[k] + gyro[k + 1]) - b_g;
        Eigen::Vector3d a = 0.5 * (accel[k] + accel[k + 1]) - b_a;

        Eigen::Matrix3d dR_inc = so3_exp(w * dt);
        Eigen::Matrix3d Jr_w = so3_jr(w * dt);
        Eigen::Matrix3d a_x = skew(a);

        // right-trivialized Jacobian updates (Forster et al. 2015)
        Eigen::Matrix3d new_J_g_R = dR_inc.transpose() * res.J_g_R - Jr_w * dt;
        Eigen::Matrix3d new_J_a_V = res.J_a_V - res.dR * dt;
        Eigen::Matrix3d new_J_g_V = res.J_g_V - res.dR * a_x * res.J_g_R * dt;
        Eigen::Matrix3d new_J_a_P = res.J_a_P + res.J_a_V * dt - 0.5 * res.dR * (dt * dt);
        Eigen::Matrix3d new_J_g_P = res.J_g_P + res.J_g_V * dt
                                    - 0.5 * res.dR * a_x * res.J_g_R * (dt * dt);

        // state integration (order matters!)
        res.dP = res.dP + res.dV * dt + 0.5 * (res.dR * a) * (dt * dt);
        res.dV = res.dV + (res.dR * a) * dt;
        res.dR = res.dR * dR_inc;

        res.J_g_R = new_J_g_R;
        res.J_g_V = new_J_g_V;
        res.J_g_P = new_J_g_P;
        res.J_a_V = new_J_a_V;
        res.J_a_P = new_J_a_P;
    }

    res.dt = ts.back() - ts.front();
    return res;
}

} // namespace vio