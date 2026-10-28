#pragma once
// The camera the estimator sees, built from the VOXL calibration files exactly as the offline
// datasets were (extract_voxl_qualisys.py, write_dataset):
//   * intrinsics: /data/modalai/opencv_<camera>_intrinsics.yml, OpenCV fisheye (M, D, width,
//     height). Every frame is undistorted to a virtual pinhole of the same size, newK from
//     cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, size, I, balance), remap
//     INTER_LINEAR with BORDER_CONSTANT. Balance 0 gave the datasets' focal 389.0 px (front).
//   * extrinsic: /etc/modalai/extrinsics.conf, the entry parent imu_apps -> child <camera>,
//     R_bc = Rx(roll) Ry(pitch) Rz(yaw) of RPY_parent_to_child in degrees, t = T_child_wrt_parent.
//     That convention gives the front camera's matrix from [0, 90, 90], and a ground-truth probe
//     preferred it on the down camera by 2.5 px against 13 to 30 px for every alternative.
// The body frame is imu_apps, as in the offline datasets (imu0 T_BS identity).
#include "vio_types.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <string>

namespace gvio {

struct OpenCvIntrinsics {
    cv::Mat K, D;            // 3x3, and 4x1 (fisheye) or 1xN (pinhole radtan)
    int width = 0, height = 0;
    std::string model;       // "fisheye" or "pinhole"
};

// Reads a VOXL /data/modalai/opencv_*_intrinsics.yml. Throws on a missing or malformed file.
OpenCvIntrinsics load_opencv_intrinsics(const std::string& path);

// Camera-in-body T_BC from a ModalAI extrinsics.conf (JSON with a comment header). Throws when the
// parent/child pair is absent.
Eigen::Matrix4d load_voxl_extrinsic(const std::string& path, const std::string& parent,
                                    const std::string& child);

// R = Rx(r) Ry(p) Rz(y), angles in degrees (ModalAI RPY_parent_to_child).
Eigen::Matrix3d rpy_deg_to_R(double r, double p, double y);

class Undistorter {
public:
    // Fisheye -> virtual pinhole of the same size (the offline extraction). newK is computed from
    // the balance, unless `pinned` (fu, fv, cu, cv) is given: estimateNewCameraMatrixForUndistort-
    // Rectify differs in the 4th decimal between OpenCV versions (389.03429 px with 4.6, 389.03439
    // px with the 4.9 that made the datasets), so a parity test pins the dataset's value.
    void init_fisheye(const OpenCvIntrinsics& c, double balance, const Eigen::Vector4d* pinned = nullptr);
    // No remap: the input is already the pinhole image (an undistorted dataset replay).
    void init_passthrough(const Eigen::Matrix3d& K, int width, int height);

    cv::Mat apply(const cv::Mat& gray) const;   // gray in, pinhole gray out
    bool remaps() const { return remap_; }
    const Eigen::Matrix3d& K() const { return K_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    bool remap_ = false;
    cv::Mat map1_, map2_;
    Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
    int w_ = 0, h_ = 0;
};

// The estimator's CamCalib: the pinhole K of the undistorter, zero distortion, T_BC, size, rate.
vio::CamCalib make_cam_calib(const Undistorter& u, const Eigen::Matrix4d& T_BC, int rate_hz);

} // namespace gvio
