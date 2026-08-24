#pragma once
#include "vio_types.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

namespace vio {

class FeatureTracker {
public:
    FeatureTracker(const CamCalib& calib,
                   int max_pts = 150,
                   int min_dist = 30,
                   double fb_err = 1.0,
                   double ransac_th = 1.0,
                   double quality = 0.01);

    // Track features into `img`. dR_body = the BODY-frame rotation since the previous
    // frame (gyro-integrated); identity = no prediction (falls back to prev-frame seed).
    // Used to rotation-compensate the KLT initial guess for large-motion robustness.
    Observation step(const cv::Mat& img,
                     const Eigen::Matrix3d& dR_body = Eigen::Matrix3d::Identity());

private:
    // calibration (cached)
    cv::Mat K_cv_;
    cv::Mat dist_cv_;
    double fx_, fy_, cx_, cy_;
    Eigen::Matrix3d R_bc_, R_cb_;   // camera<->body rotation (from T_BC) for IMU-aided prediction

    // params
    int max_pts_;
    int min_dist_;
    double fb_err_;
    double ransac_th_;
    double quality_;

    // KLT params
    cv::Size klt_win_;
    int klt_max_level_;
    cv::TermCriteria klt_crit_;

    // Shi-Tomasi params
    int block_size_;

    // state
    cv::Mat prev_img_;
    std::vector<cv::Point2f> prev_pts_;
    std::vector<int64_t> ids_;
    std::vector<int> track_cnt_;
    int64_t next_id_;

    // helpers
    void detect_new(const cv::Mat& img);
    void filter(const std::vector<unsigned char>& mask);
    void filter(const std::vector<char>& mask);
    std::vector<cv::Point2f> undistort(const std::vector<cv::Point2f>& pts) const;
    Observation make_observation() const;
    static cv::Mat ensure_gray_uint8(const cv::Mat& img);
};

} // namespace vio