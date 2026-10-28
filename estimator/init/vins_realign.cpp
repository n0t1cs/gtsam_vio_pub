#include "vins_realign.h"
#include "init_sfm.h"   // inv_se3

namespace vio {

AlignedInit& vins_realign(AlignedInit& aligned) {
    if (!aligned.success) return aligned;

    // 1) Rotation that aligns g_w -> -Z
    Eigen::Vector3d g_dir = aligned.g_w / aligned.g_w.norm();
    Eigen::Vector3d g_target(0.0, 0.0, -1.0);

    Eigen::Vector3d v = g_dir.cross(g_target);
    double c = g_dir.dot(g_target);

    Eigen::Matrix3d R_align;
    if (c < -0.9999) {
        R_align = Eigen::Matrix3d::Identity();
        R_align(1, 1) = -1.0;
        R_align(2, 2) = -1.0;
    } else {
        Eigen::Matrix3d v_skew;
        v_skew <<     0, -v(2),  v(1),
                  v(2),     0, -v(0),
                 -v(1),  v(0),     0;
        R_align = Eigen::Matrix3d::Identity() + v_skew + (v_skew * v_skew) * (1.0 / (1.0 + c));
    }

    double s = aligned.scale;

    Eigen::Matrix4d T_BC = aligned.sfm.calib.T_BC;
    Eigen::Matrix4d T_CB = T_BC.inverse();

    aligned.T_wb.clear();
    aligned.T_wb.reserve(aligned.sfm.T_wc.size());

    for (size_t i = 0; i < aligned.sfm.T_wc.size(); ++i) {
        Eigen::Matrix4d& T_wc = aligned.sfm.T_wc[i];

        // Scale translation, then rotate
        Eigen::Vector3d t_new = R_align * (T_wc.block<3, 1>(0, 3) * s);
        Eigen::Matrix3d R_new = R_align * T_wc.block<3, 3>(0, 0);
        T_wc.block<3, 1>(0, 3) = t_new;
        T_wc.block<3, 3>(0, 0) = R_new;

        // Update Tcw
        Eigen::Matrix3d R_cw = R_new.transpose();
        Eigen::Vector3d t_cw = -R_cw * t_new;
        aligned.sfm.Tcw[i].block<3, 3>(0, 0) = R_cw;
        aligned.sfm.Tcw[i].block<3, 1>(0, 3) = t_cw;

        // Body pose
        aligned.T_wb.push_back(T_wc * T_CB);
    }

    // 4) Rotate velocities
    for (auto& v : aligned.velocities) v = R_align * v;

    // 5) Rotate gravity (~[0,0,-9.81])
    aligned.g_w = R_align * aligned.g_w;

    // 6) Scale and rotate SfM landmarks
    for (auto& kv : aligned.sfm.points) {
        kv.second = R_align * (kv.second * s);
    }

    return aligned;
}

} // namespace vio