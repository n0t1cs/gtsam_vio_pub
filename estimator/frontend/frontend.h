#pragma once
#include "vio_types.h"
#include <unordered_map>
#include <random>
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
                   double quality = 0.01,
                   int grid_x = 1,
                   int grid_y = 1,
                   int histeq = 0,            // 0=none, 1=global equalizeHist (OpenVINS EuRoC), 2=CLAHE
                   double clahe_clip = 4.0);

    // Track features into `img`. dR_body = the BODY-frame rotation since the previous
    // frame (gyro-integrated); identity = no prediction (falls back to prev-frame seed).
    // Used to rotation-compensate the KLT initial guess for large-motion robustness.
    Observation step(const cv::Mat& img,
                     const Eigen::Matrix3d& dR_body = Eigen::Matrix3d::Identity());

    // Frame-to-frame outlier rejection (point_frontend.ransac_method): 0 = OpenCV 8-point F
    // RANSAC (default), 1 = OpenCV USAC MAGSAC++ on F, 2 = translation-only 2-point RANSAC
    // on E with the gyro rotation (falls back to 0 when no rotation is given).
    // sample_seed > 0 (runtime.rng_seed): OpenCV's RANSAC draws from its own fixed generator,
    // not cv::theRNG(), so a seed set permutes the correspondences before the call instead,
    // which maps the fixed draw onto different points. 0 = no permutation (historical runs).
    void set_outlier_rejection(int method, double conf, int iters, int sample_seed) {
        rej_method_ = method; rej_conf_ = conf; rej_iters_ = iters;
        perm_seed_ = sample_seed > 0 ? sample_seed : 0;
        perm_rng_.seed(static_cast<std::uint32_t>(perm_seed_));
    }

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
    int    rej_method_ = 0;        // outlier rejection, see set_outlier_rejection()
    double rej_conf_   = 0.99;
    int    rej_iters_  = 2000;
    int    perm_seed_  = 0;        // > 0: permute correspondences before RANSAC (seed sets)
    std::mt19937 perm_rng_;
    double quality_;
    int grid_x_;                   // detection grid columns (1 = global; >1 = per-cell uniform)
    int grid_y_;                   // detection grid rows
    int histeq_ = 0;               // 0=none, 1=global equalizeHist, 2=CLAHE (OpenVINS histogram modes)
    cv::Ptr<cv::CLAHE> clahe_;     // built only when histeq_==2

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
    std::unordered_map<int64_t, double> fb_last_;   // id -> this-frame forward-backward residual [px]
    int64_t next_id_;

    // helpers
    void detect_new(const cv::Mat& img);
    void filter(const std::vector<unsigned char>& mask);
    void filter(const std::vector<char>& mask);
    std::vector<cv::Point2f> undistort(const std::vector<cv::Point2f>& pts) const;
    std::vector<unsigned char> two_point_inliers(const std::vector<cv::Point2f>& p0,
                                                 const std::vector<cv::Point2f>& p1,
                                                 const Eigen::Matrix3d& dR_body) const;
    Observation make_observation() const;
    static cv::Mat ensure_gray_uint8(const cv::Mat& img);
};

} // namespace vio