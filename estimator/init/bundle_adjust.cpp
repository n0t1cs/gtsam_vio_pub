#include "bundle_adjust.h"

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

namespace vio {

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::L;

SfmResult& bundle_adjust(SfmResult& sfm, int max_iters) {
    if (!sfm.success) {
        std::cout << "bundle_adjust: SfM did not succeed, skipping.\n";
        return sfm;
    }

    const Eigen::Matrix3d& K = sfm.calib.K;
    double fx = K(0, 0), fy = K(1, 1);
    double cx = K(0, 2), cy = K(1, 2);

    gtsam::Cal3_S2::shared_ptr calib_ptr(new gtsam::Cal3_S2(fx, fy, 0.0, cx, cy));
    gtsam::Pose3 T_bc_id;  // identity

    auto proj_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);

    // Pose prior (tight, 6D)
    Eigen::Matrix<double, 6, 1> pose_sigma;
    pose_sigma.setConstant(1e-6);
    auto pose_prior = gtsam::noiseModel::Diagonal::Sigmas(pose_sigma);

    // Scale-locking prior (rotation free, translation locked)
    Eigen::Matrix<double, 6, 1> scale_sigmas;
    scale_sigmas << 1e4, 1e4, 1e4, 1e-6, 1e-6, 1e-6;
    auto scale_prior = gtsam::noiseModel::Diagonal::Sigmas(scale_sigmas);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    int N = (int)sfm.T_wc.size();

    // Camera poses
    for (int i = 0; i < N; ++i) {
        const Eigen::Matrix4d& T = sfm.T_wc[i];
        gtsam::Rot3 R(T.block<3, 3>(0, 0));
        gtsam::Point3 t(T.block<3, 1>(0, 3));
        values.insert(X(i), gtsam::Pose3(R, t));
    }

    graph.addPrior(X(0), values.at<gtsam::Pose3>(X(0)), pose_prior);
    if (N >= 2) {
        graph.addPrior(X(N - 1), values.at<gtsam::Pose3>(X(N - 1)), scale_prior);
    }

    using ProjFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;

    int n_lm = 0, n_obs = 0;

    for (auto& kv : sfm.points) {
        int64_t fid = kv.first;
        const Eigen::Vector3d& X_world = kv.second;

        // Count views
        int views = 0;
        for (const auto& f : sfm.frames) {
            for (int64_t id : f.ids) {
                if (id == fid) { views++; break; }
            }
        }
        if (views < 2) continue;

        values.insert(L(fid), gtsam::Point3(X_world));
        n_lm++;

        for (int i = 0; i < N; ++i) {
            const auto& f = sfm.frames[i];
            int j = -1;
            for (size_t kk = 0; kk < f.ids.size(); ++kk) {
                if (f.ids[kk] == fid) { j = (int)kk; break; }
            }
            if (j < 0) continue;

            double pn_x = f.pts_norm(j, 0);
            double pn_y = f.pts_norm(j, 1);
            double u_ud = fx * pn_x + cx;
            double v_ud = fy * pn_y + cy;

            graph.emplace_shared<ProjFactor>(
                gtsam::Point2(u_ud, v_ud), proj_noise,
                X(i), L(fid), calib_ptr, T_bc_id);
            n_obs++;
        }
    }

    std::cout << "bundle_adjust: " << n_lm << " landmarks  |  "
              << n_obs << " reprojection factors\n";
    sfm.ba_n_obs = n_obs;
    if (n_obs == 0) {
        std::cout << "bundle_adjust: nothing to optimize, skipping.\n";
        return sfm;
    }

    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(max_iters);
    params.setRelativeErrorTol(1e-5);
    params.setAbsoluteErrorTol(1e-5);

    double err_before = graph.error(values);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    gtsam::Values result = optimizer.optimize();
    double err_after = graph.error(result);
    sfm.ba_err_before = err_before;
    sfm.ba_err_after  = err_after;
    double ratio = err_before / std::max(err_after, 1e-9);
    std::cout << "bundle_adjust: reprojection error  "
              << err_before << " -> " << err_after
              << "  (" << ratio << "x reduction)\n";

    // Write refined poses back
    for (int i = 0; i < N; ++i) {
        gtsam::Pose3 p = result.at<gtsam::Pose3>(X(i));
        Eigen::Matrix4d T_new = Eigen::Matrix4d::Identity();
        T_new.block<3, 3>(0, 0) = p.rotation().matrix();
        T_new.block<3, 1>(0, 3) = p.translation();
        sfm.T_wc[i] = T_new;
        sfm.Tcw[i] = inv_se3(T_new);
    }

    // Write refined landmarks back
    for (auto& kv : sfm.points) {
        int64_t fid = kv.first;
        gtsam::Key lk = L(fid);
        if (result.exists(lk)) {
            kv.second = result.at<gtsam::Point3>(lk);
        }
    }

    return sfm;
}

} // namespace vio