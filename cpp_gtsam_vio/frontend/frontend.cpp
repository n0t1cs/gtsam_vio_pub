#include "frontend.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace vio {

FeatureTracker::FeatureTracker(const CamCalib& calib, int max_pts, int min_dist,
                               double fb_err, double ransac_th, double quality)
    : max_pts_(max_pts), min_dist_(min_dist), fb_err_(fb_err),
      ransac_th_(ransac_th), quality_(quality),
      klt_win_(21, 21), klt_max_level_(3),
      klt_crit_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01),
      block_size_(3), next_id_(0)
{
    K_cv_ = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K_cv_.at<double>(i, j) = calib.K(i, j);

    dist_cv_ = (cv::Mat_<double>(4, 1) << calib.dist(0), calib.dist(1), calib.dist(2), calib.dist(3));
    
    
    fx_ = calib.K(0, 0);
    fy_ = calib.K(1, 1);
    cx_ = calib.K(0, 2);
    cy_ = calib.K(1, 2);

    R_bc_ = calib.T_BC.block<3, 3>(0, 0);   // camera->body rotation
    R_cb_ = R_bc_.transpose();              // body->camera
}

Observation FeatureTracker::step(const cv::Mat& img_in, const Eigen::Matrix3d& dR_body) {
    cv::Mat img = ensure_gray_uint8(img_in);

    if (prev_img_.empty()) {
        detect_new(img);
        prev_img_ = img.clone();
        return make_observation();
    }

    // ---- 1) KLT forward-backward ------------------------------------------
    std::vector<cv::Point2f> next_pts, back_pts;
    std::vector<unsigned char> status, status_b;
    std::vector<float> err, err_b;

    if (prev_pts_.empty()) {
        // Nothing to track; just (re-)detect.
        prev_img_ = img.clone();
        detect_new(img);
        return make_observation();
    }

    // ---- IMU-aided prediction: seed the KLT search by rotating the prior points by the
    //      inter-frame CAMERA rotation (VINS-Mono / OpenVINS). A static world point's
    //      apparent motion under a camera rotation dR_cam is dR_cam^T on its bearing:
    //        bearing_prev = undistort(px_prev);  bearing_pred = dR_cam^T bearing_prev;
    //        px_pred      = distort(K bearing_pred);   dR_cam = R_cb * dR_body * R_bc.
    //      Distortion is handled on BOTH ends -- prev pixels are undistorted to true
    //      bearings, and the rotated bearing is RE-DISTORTED back to image space (where
    //      KLT operates). A pinhole-only warp mis-seeds under strong radtan (EuRoC
    //      k1=-0.283): the distortion error swamps the small 20Hz rotation and the seed
    //      lands worse than the (distortion-free) prev-pixel -- measured net-negative on
    //      the full 11-seq sweep. This is only the SEED; KLT refines from it.
    const bool use_pred = !dR_body.isIdentity(1e-9);
    next_pts = prev_pts_;                                   // default seed = prev positions
    if (use_pred && !prev_pts_.empty()) {
        // 1) distorted prev pixels -> ideal normalized bearings (z = 1)
        std::vector<cv::Point2f> norm;
        cv::undistortPoints(prev_pts_, norm, K_cv_, dist_cv_);
        // 2) rotate each bearing by the inter-frame camera rotation
        const Eigen::Matrix3d Rp = (R_cb_ * dR_body * R_bc_).transpose();  // dR_cam^T
        std::vector<cv::Point3f> rays(norm.size());
        for (size_t i = 0; i < norm.size(); ++i) {
            const Eigen::Vector3d b(norm[i].x, norm[i].y, 1.0);
            const Eigen::Vector3d r = Rp * b;
            rays[i] = cv::Point3f((float)r.x(), (float)r.y(), (float)r.z());
        }
        // 3) re-project WITH distortion -> predicted distorted pixels
        std::vector<cv::Point2f> pred;
        cv::projectPoints(rays, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                          K_cv_, dist_cv_, pred);
        for (size_t i = 0; i < pred.size(); ++i)
            if (rays[i].z > 1e-6f) next_pts[i] = pred[i];   // keep prev seed if behind cam
    }

    cv::calcOpticalFlowPyrLK(prev_img_, img, prev_pts_, next_pts, status, err,
                             klt_win_, klt_max_level_, klt_crit_,
                             use_pred ? cv::OPTFLOW_USE_INITIAL_FLOW : 0);
    cv::calcOpticalFlowPyrLK(img, prev_img_, next_pts, back_pts, status_b, err_b,
                             klt_win_, klt_max_level_, klt_crit_);

    std::vector<unsigned char> good(prev_pts_.size(), 0);
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
        if (status[i] == 1 && status_b[i] == 1) {
            cv::Point2f d = prev_pts_[i] - back_pts[i];
            double fb_dist = std::sqrt(d.x * d.x + d.y * d.y);
            if (fb_dist < fb_err_) good[i] = 1;
        }
    }

    // Take the subset of next_pts where good == 1.
    std::vector<cv::Point2f> cur_pts;
    cur_pts.reserve(prev_pts_.size());
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
        if (good[i]) cur_pts.push_back(next_pts[i]);
    }
    filter(good);

    // ---- 2) Drop out-of-bounds --------------------------------------------
    const int H = img.rows, W = img.cols;
    std::vector<unsigned char> in_bounds(cur_pts.size(), 0);
    for (size_t i = 0; i < cur_pts.size(); ++i) {
        if (cur_pts[i].x >= 0 && cur_pts[i].x < W &&
            cur_pts[i].y >= 0 && cur_pts[i].y < H) {
            in_bounds[i] = 1;
        }
    }
    std::vector<cv::Point2f> cur2;
    cur2.reserve(cur_pts.size());
    for (size_t i = 0; i < cur_pts.size(); ++i)
        if (in_bounds[i]) cur2.push_back(cur_pts[i]);
    cur_pts.swap(cur2);
    filter(in_bounds);

    // ---- 3) F-matrix RANSAC on undistorted points -------------------------
    if (cur_pts.size() >= 8) {
        std::vector<cv::Point2f> p0 = undistort(prev_pts_);
        std::vector<cv::Point2f> p1 = undistort(cur_pts);
        try {
            std::vector<unsigned char> inl;
            cv::Mat F = cv::findFundamentalMat(p0, p1, cv::FM_RANSAC,
                                               ransac_th_, 0.99, 2000, inl);
            if (!F.empty() && !inl.empty()) {
                std::vector<cv::Point2f> kept;
                kept.reserve(cur_pts.size());
                for (size_t i = 0; i < cur_pts.size(); ++i)
                    if (inl[i]) kept.push_back(cur_pts[i]);
                cur_pts.swap(kept);
                filter(inl);
            }
        } catch (const cv::Exception&) {
            // degenerate config -> keep what we have
        }
    }

    // ---- 4) Commit and bump track ages ------------------------------------
    prev_pts_ = cur_pts;
    for (auto& c : track_cnt_) c++;

    // ---- 5) Refill --------------------------------------------------------
    if ((int)prev_pts_.size() < max_pts_) {
        detect_new(img);
    }

    prev_img_ = img.clone();
    return make_observation();
}

void FeatureTracker::detect_new(const cv::Mat& img) {
    int n_need = max_pts_ - (int)prev_pts_.size();
    if (n_need <= 0) return;

    cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
    for (const auto& p : prev_pts_) {
        cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), min_dist_, 0, -1);
    }

    std::vector<cv::Point2f> new_pts;
    cv::goodFeaturesToTrack(img, new_pts, n_need, quality_, (double)min_dist_,
                            mask, block_size_);
    if (new_pts.empty()) return;

    for (const auto& p : new_pts) {
        prev_pts_.push_back(p);
        ids_.push_back(next_id_++);
        track_cnt_.push_back(1);
    }
}

void FeatureTracker::filter(const std::vector<unsigned char>& mask) {
    std::vector<cv::Point2f> p;
    std::vector<int64_t> ids;
    std::vector<int> cnt;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            p.push_back(prev_pts_[i]);
            ids.push_back(ids_[i]);
            cnt.push_back(track_cnt_[i]);
        }
    }
    prev_pts_.swap(p);
    ids_.swap(ids);
    track_cnt_.swap(cnt);
}

void FeatureTracker::filter(const std::vector<char>& mask) {
    std::vector<unsigned char> m(mask.size());
    for (size_t i = 0; i < mask.size(); ++i) m[i] = mask[i] ? 1 : 0;
    filter(m);
}

std::vector<cv::Point2f> FeatureTracker::undistort(const std::vector<cv::Point2f>& pts) const {
    std::vector<cv::Point2f> out;
    if (pts.empty()) return out;
    cv::undistortPoints(pts, out, K_cv_, dist_cv_, cv::noArray(), K_cv_);
    return out;
}

Observation FeatureTracker::make_observation() const {
    Observation o;
    const int N = (int)prev_pts_.size();
    o.ids = ids_;
    o.track_cnt = track_cnt_;
    o.pts_raw.resize(N, 2);
    o.pts_norm.resize(N, 2);
    if (N == 0) return o;

    std::vector<cv::Point2f> und = undistort(prev_pts_);
    for (int i = 0; i < N; ++i) {
        o.pts_raw(i, 0) = prev_pts_[i].x;
        o.pts_raw(i, 1) = prev_pts_[i].y;
        // normalized bearings from undistorted pixel coords
        o.pts_norm(i, 0) = (und[i].x - cx_) / fx_;
        o.pts_norm(i, 1) = (und[i].y - cy_) / fy_;
    }
    return o;
}

cv::Mat FeatureTracker::ensure_gray_uint8(const cv::Mat& img) {
    cv::Mat g;
    if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    else g = img;
    if (g.type() != CV_8U) {
        cv::Mat g2;
        g.convertTo(g2, CV_8U);
        return g2;
    }
    return g;
}

} // namespace vio