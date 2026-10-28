#include "init_align.h"
#include "imu_preintegrate.h"

#include <algorithm>
#include <iostream>
#include <cmath>

namespace vio {

// Local gravity magnitude at the EuRoC site (ETH Zurich), not the 9.81 textbook
// value. This is the single source of truth: the backend reads the realigned
// gravity vector's norm (aligned.g_w.norm()) so it never re-hardcodes it.
static const double G_MAG = 9.8066;

// ---------------------------------------------------------------------------
// helpers: IMU subset between two seconds-since-t0 timestamps
// ---------------------------------------------------------------------------
struct ImuBatch {
    std::vector<double> ts;                       // seconds since t0
    std::vector<Eigen::Vector3d> gyro;
    std::vector<Eigen::Vector3d> accel;
};

static ImuBatch imu_between(const EurocData& data, double t_a, double t_b) {
    int64_t t0_ns = data.frames.front().t_ns;
    int64_t t_a_ns = t0_ns + (int64_t)std::llround(t_a * 1e9);
    int64_t t_b_ns = t0_ns + (int64_t)std::llround(t_b * 1e9);

    auto lo = std::upper_bound(data.imu.begin(), data.imu.end(), t_a_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    auto hi = std::upper_bound(data.imu.begin(), data.imu.end(), t_b_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });

    ImuBatch b;
    for (auto it = lo; it != hi; ++it) {
        b.ts.push_back((it->t_ns - t0_ns) * 1e-9);
        b.gyro.push_back(it->w);
        b.accel.push_back(it->a);
    }
    return b;
}

// ---------------------------------------------------------------------------
// Stage 1: gyro bias
// ---------------------------------------------------------------------------
static Eigen::Vector3d solve_gyro_bias(const SfmResult& sfm,
                                       const EurocData& data,
                                       const Eigen::Matrix3d& R_bc) {
    int N = (int)sfm.frames.size();
    Eigen::MatrixXd A((N - 1) * 3, 3);
    Eigen::VectorXd B_v((N - 1) * 3);
    A.setZero();
    B_v.setZero();

    for (int i = 0; i < N - 1; ++i) {
        auto bat = imu_between(data, sfm.frames[i].t, sfm.frames[i + 1].t);
        auto pre = imu_preintegrate(bat.ts, bat.gyro, bat.accel,
                                    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                    sfm.frames[i].t);

        Eigen::Matrix3d R_wi = sfm.T_wc[i].block<3, 3>(0, 0) * R_bc.transpose();
        Eigen::Matrix3d R_wj = sfm.T_wc[i + 1].block<3, 3>(0, 0) * R_bc.transpose();
        Eigen::Matrix3d R_ij_sfm = R_wi.transpose() * R_wj;

        A.block<3, 3>(3 * i, 0) = pre.J_g_R;
        Eigen::Matrix3d err_R = pre.dR.transpose() * R_ij_sfm;
        B_v.segment<3>(3 * i) = rot_to_vec(err_R);
    }

    Eigen::Matrix3d AtA = A.transpose() * A;
    Eigen::Vector3d Atb = A.transpose() * B_v;
    Eigen::Vector3d dbg = AtA.ldlt().solve(Atb);
    return dbg;
}

// ---------------------------------------------------------------------------
// Stage 2: linear alignment [v0..vN-1, g_w, s]
// ---------------------------------------------------------------------------
static bool solve_linear_alignment(const SfmResult& sfm,
                                   const EurocData& data,
                                   const Eigen::Vector3d& b_g,
                                   const Eigen::Matrix3d& R_bc,
                                   const Eigen::Vector3d& t_bc,
                                   Eigen::VectorXd& x_out) {
    int N = (int)sfm.frames.size();
    int dim = 3 * N + 3 + 1;
    Eigen::MatrixXd A((N - 1) * 6, dim);
    Eigen::VectorXd B_v((N - 1) * 6);
    A.setZero();
    B_v.setZero();

    for (int i = 0; i < N - 1; ++i) {
        double dt = sfm.frames[i + 1].t - sfm.frames[i].t;
        auto bat = imu_between(data, sfm.frames[i].t, sfm.frames[i + 1].t);
        auto pre = imu_preintegrate(bat.ts, bat.gyro, bat.accel,
                                    b_g, Eigen::Vector3d::Zero(),
                                    sfm.frames[i].t);

        Eigen::Matrix3d R_wi = sfm.T_wc[i].block<3, 3>(0, 0) * R_bc.transpose();
        Eigen::Matrix3d R_wj = sfm.T_wc[i + 1].block<3, 3>(0, 0) * R_bc.transpose();
        Eigen::Vector3d p_ci = sfm.T_wc[i].block<3, 1>(0, 3);
        Eigen::Vector3d p_cj = sfm.T_wc[i + 1].block<3, 1>(0, 3);

        int row = 6 * i;
        // Velocity equation
        A.block<3, 3>(row, 3 * i) = -Eigen::Matrix3d::Identity();
        A.block<3, 3>(row, 3 * (i + 1)) = Eigen::Matrix3d::Identity();
        A.block<3, 3>(row, 3 * N) = -Eigen::Matrix3d::Identity() * dt;
        B_v.segment<3>(row) = R_wi * pre.dV;

        // Position equation
        A.block<3, 3>(row + 3, 3 * i) = Eigen::Matrix3d::Identity() * dt;
        A.block<3, 3>(row + 3, 3 * N) = 0.5 * Eigen::Matrix3d::Identity() * dt * dt;
        A.block<3, 1>(row + 3, dim - 1) = -(p_cj - p_ci);
        B_v.segment<3>(row + 3) = -R_wi * pre.dP + (R_wi - R_wj) * t_bc;
    }

    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::VectorXd Atb = A.transpose() * B_v;
    x_out = AtA.ldlt().solve(Atb);

    double s = x_out(x_out.size() - 1);
    Eigen::Vector3d g_w = x_out.segment<3>(x_out.size() - 4);
    bool ok = (s > 0.1) && (s < 10.0) && (std::abs(g_w.norm() - G_MAG) < 1.5);
    return ok;
}

// ---------------------------------------------------------------------------
// Stage 3: refine gravity (4 GN iterations on tangent of unit sphere)
// ---------------------------------------------------------------------------
static void refine_gravity(const SfmResult& sfm,
                           const EurocData& data,
                           const Eigen::Vector3d& b_g,
                           const Eigen::Matrix3d& R_bc,
                           const Eigen::Vector3d& t_bc,
                           double s,
                           const Eigen::Vector3d& g_w_init,
                           const std::vector<Eigen::Vector3d>& vel_init,
                           bool est_ba,
                           Eigen::Vector3d& g_w_out,
                           std::vector<Eigen::Vector3d>& velocities_out,
                           Eigen::Vector3d& b_a_out) {
    int N = (int)sfm.frames.size();
    Eigen::Vector3d g_hat = g_w_init / g_w_init.norm();
    velocities_out = vel_init;
    b_a_out.setZero();

    for (int iter = 0; iter < 4; ++iter) {
        // Orthonormal tangent basis
        Eigen::Vector3d tmp;
        if (std::abs(g_hat(0)) < 0.9) tmp = Eigen::Vector3d(1, 0, 0);
        else                          tmp = Eigen::Vector3d(0, 1, 0);
        Eigen::Vector3d b1 = g_hat.cross(tmp);
        b1.normalize();
        Eigen::Vector3d b2 = g_hat.cross(b1);

        int dim = 3 * N + 2 + (est_ba ? 3 : 0);   // + accel-bias (3) when co-estimating
        Eigen::MatrixXd A((N - 1) * 6, dim);
        Eigen::VectorXd B_v((N - 1) * 6);
        A.setZero();
        B_v.setZero();

        for (int i = 0; i < N - 1; ++i) {
            double dt = sfm.frames[i + 1].t - sfm.frames[i].t;
            auto bat = imu_between(data, sfm.frames[i].t, sfm.frames[i + 1].t);
            auto pre = imu_preintegrate(bat.ts, bat.gyro, bat.accel,
                                        b_g, Eigen::Vector3d::Zero(),
                                        sfm.frames[i].t);

            Eigen::Matrix3d R_wi = sfm.T_wc[i].block<3, 3>(0, 0) * R_bc.transpose();
            Eigen::Matrix3d R_wj = sfm.T_wc[i + 1].block<3, 3>(0, 0) * R_bc.transpose();
            Eigen::Vector3d p_ci = sfm.T_wc[i].block<3, 1>(0, 3);
            Eigen::Vector3d p_cj = sfm.T_wc[i + 1].block<3, 1>(0, 3);

            int row = 6 * i;
            // Velocity equation
            A.block<3, 3>(row, 3 * i) = -Eigen::Matrix3d::Identity();
            A.block<3, 3>(row, 3 * (i + 1)) = Eigen::Matrix3d::Identity();
            A.block<3, 1>(row, 3 * N) = -G_MAG * dt * b1;
            A.block<3, 1>(row, 3 * N + 1) = -G_MAG * dt * b2;
            B_v.segment<3>(row) = g_hat * G_MAG * dt + R_wi * pre.dV;
            // accel bias enters via dV = dV(0) + J_a_V * b_a  (first-order)
            if (est_ba) A.block<3, 3>(row, 3 * N + 2) = -R_wi * pre.J_a_V;

            // Position equation
            A.block<3, 3>(row + 3, 3 * i) = Eigen::Matrix3d::Identity() * dt;
            A.block<3, 1>(row + 3, 3 * N) = 0.5 * G_MAG * dt * dt * b1;
            A.block<3, 1>(row + 3, 3 * N + 1) = 0.5 * G_MAG * dt * dt * b2;
            B_v.segment<3>(row + 3) = 0.5 * g_hat * G_MAG * dt * dt
                                       - s * (p_cj - p_ci)
                                       - R_wi * pre.dP
                                       + (R_wi - R_wj) * t_bc;
            // accel bias enters via dP = dP(0) + J_a_P * b_a  (first-order)
            if (est_ba) A.block<3, 3>(row + 3, 3 * N + 2) = R_wi * pre.J_a_P;
        }

        Eigen::VectorXd x_ref = (A.transpose() * A).ldlt().solve(A.transpose() * B_v);

        if (est_ba) b_a_out = x_ref.segment<3>(3 * N + 2);
        Eigen::Vector2d dw = x_ref.segment<2>(3 * N);
        g_hat = g_hat + b1 * dw(0) + b2 * dw(1);
        g_hat.normalize();

        for (int i = 0; i < N; ++i) {
            velocities_out[i] = x_ref.segment<3>(3 * i);
        }
    }

    g_w_out = g_hat * G_MAG;
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------
AlignedInit init_align(const SfmResult& sfm, const EurocData& data,
                       bool est_accel_bias) {
    AlignedInit out;
    out.sfm = sfm;

    if (!sfm.success) {
        std::cout << "init_align: SfM not successful, skipping.\n";
        return out;
    }

    Eigen::Matrix3d R_bc = sfm.calib.T_BC.block<3, 3>(0, 0);
    Eigen::Vector3d t_bc = sfm.calib.T_BC.block<3, 1>(0, 3);

    // Stage 1
    Eigen::Vector3d b_g = solve_gyro_bias(sfm, data, R_bc);
    std::cout << "init_align: gyro bias = ["
              << b_g(0) << ", " << b_g(1) << ", " << b_g(2) << "] rad/s\n";

    // Stage 2
    Eigen::VectorXd x;
    bool ok = solve_linear_alignment(sfm, data, b_g, R_bc, t_bc, x);

    int N = (int)sfm.frames.size();
    double s = x(x.size() - 1);
    Eigen::Vector3d g_w_init = x.segment<3>(x.size() - 4);
    std::vector<Eigen::Vector3d> vel_init(N);
    for (int i = 0; i < N; ++i) vel_init[i] = x.segment<3>(3 * i);

    std::cout << "init_align: initial linear solution  scale=" << s
              << "  g_w=[" << g_w_init(0) << "," << g_w_init(1) << "," << g_w_init(2) << "]"
              << "  |g|=" << g_w_init.norm() << "\n";

    if (!ok) {
        std::cout << "init_align: linear alignment rejected (scale or gravity out of range)\n";
        return out;
    }

    // Stage 3
    Eigen::Vector3d g_w, b_a;
    std::vector<Eigen::Vector3d> velocities;
    refine_gravity(sfm, data, b_g, R_bc, t_bc, s, g_w_init, vel_init,
                   est_accel_bias, g_w, velocities, b_a);
    // Sanity: a degenerate window can produce an implausible accel bias; EuRoC
    // accel bias is ~0.01-0.1 m/s^2, so discard anything beyond a few tenths and
    // fall back to zero rather than poison gravity/bias with a solver artefact.
    if (est_accel_bias && b_a.norm() > 0.5) {
        // Ill-conditioned (low excitation): the accel-bias/gravity split is
        // unobservable, so the augmented solve corrupts gravity. Discard b_a and
        // RE-RUN the plain refinement so gravity falls back to the safe estimate.
        std::cout << "init_align: est accel bias |b_a|=" << b_a.norm()
                  << " m/s^2 implausible (unobservable) -> fallback to plain gravity refine\n";
        b_a.setZero();
        refine_gravity(sfm, data, b_g, R_bc, t_bc, s, g_w_init, vel_init,
                       false, g_w, velocities, b_a);
    }
    std::cout << "init_align: refined gravity  g_w=["
              << g_w(0) << "," << g_w(1) << "," << g_w(2) << "]"
              << "  |g|=" << g_w.norm();
    if (est_accel_bias)
        std::cout << "  b_a=[" << b_a(0) << "," << b_a(1) << "," << b_a(2) << "]";
    std::cout << "\n";

    out.success = true;
    out.scale = s;
    out.g_w = g_w;
    out.b_g = b_g;
    out.b_a = b_a;
    out.velocities = std::move(velocities);
    return out;
}

} // namespace vio