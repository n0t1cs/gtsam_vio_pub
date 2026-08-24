#pragma once
#include <Eigen/Dense>
#include <vector>
#include <optional>

namespace vio {

struct PreintResult {
    Eigen::Matrix3d dR;
    Eigen::Vector3d dV;
    Eigen::Vector3d dP;
    double dt;
    Eigen::Matrix3d J_g_R;
    Eigen::Matrix3d J_g_V;
    Eigen::Matrix3d J_g_P;
    Eigen::Matrix3d J_a_V;
    Eigen::Matrix3d J_a_P;

    PreintResult();
};

// ts:    (M,) timestamps in seconds
// gyro:  (M, 3) gyro samples (rad/s)
// accel: (M, 3) accel samples (m/s^2, specific force minus gravity supplied externally)
// b_g, b_a: linearization biases (defaults zero)
// t_start: if set and < ts[0], prepend partial interval [t_start, ts[0]]
PreintResult imu_preintegrate(const std::vector<double>& ts,
                              const std::vector<Eigen::Vector3d>& gyro,
                              const std::vector<Eigen::Vector3d>& accel,
                              const Eigen::Vector3d& b_g = Eigen::Vector3d::Zero(),
                              const Eigen::Vector3d& b_a = Eigen::Vector3d::Zero(),
                              std::optional<double> t_start = std::nullopt);

// SO(3) helpers, exposed for the alignment stages.
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& phi);
Eigen::Matrix3d so3_jr(const Eigen::Vector3d& phi);
Eigen::Matrix3d skew(const Eigen::Vector3d& v);
Eigen::Vector3d rot_to_vec(const Eigen::Matrix3d& R);

} // namespace vio