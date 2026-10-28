#include "frontend.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

namespace vio {

FeatureTracker::FeatureTracker(const CamCalib& calib, int max_pts, int min_dist,
                               double fb_err, double ransac_th, double quality,
                               int grid_x, int grid_y, int histeq, double clahe_clip)
    : max_pts_(max_pts), min_dist_(min_dist), fb_err_(fb_err),
      ransac_th_(ransac_th), quality_(quality),
      grid_x_(grid_x < 1 ? 1 : grid_x), grid_y_(grid_y < 1 ? 1 : grid_y),
      histeq_(histeq),
      klt_win_(21, 21), klt_max_level_(3),
      klt_crit_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01),
      block_size_(3), next_id_(0)
{
    // Optional histogram equalization (OpenVINS modes), applied to the grayscale
    // image before tracking + detection (prev_img_ stored equalized -> KLT is
    // equalized-to-equalized). 1 = global equalizeHist (OpenVINS EuRoC default,
    // gentle); 2 = CLAHE (adaptive, clip_limit tunable, 8x8 tiles -- aggressive,
    // amplifies noise in dark scenes). 0 -> byte-identical to the raw pipeline.
    if (histeq == 2) clahe_ = cv::createCLAHE(clahe_clip, cv::Size(8, 8));
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
    if (histeq_ == 1)      cv::equalizeHist(img, img);   // global (OpenVINS EuRoC)
    else if (clahe_)       clahe_->apply(img, img);      // CLAHE (adaptive)

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
    fb_last_.clear();   // id -> this-frame FB residual, for heteroscedastic reprojection cov
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
        if (status[i] == 1 && status_b[i] == 1) {
            cv::Point2f d = prev_pts_[i] - back_pts[i];
            double fb_dist = std::sqrt(d.x * d.x + d.y * d.y);
            if (fb_dist < fb_err_) { good[i] = 1; fb_last_[ids_[i]] = fb_dist; }
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

    // ---- 3) Outlier rejection on undistorted points ------------------------
    // Default: 8-point F RANSAC. magsac: the same model under USAC MAGSAC++. twopt: the gyro
    // gives the rotation, so only the translation direction is sampled (2 points per
    // hypothesis), see two_point_inliers(). Without a rotation twopt falls back to F RANSAC.
    if (cur_pts.size() >= 8) {
        std::vector<cv::Point2f> p0 = undistort(prev_pts_);
        std::vector<cv::Point2f> p1 = undistort(cur_pts);
        std::vector<unsigned char> inl;
        if (rej_method_ == 2 && use_pred) {
            inl = two_point_inliers(p0, p1, dR_body);
        } else {
            // Seed sets: the same RANSAC on a seeded permutation of the correspondences.
            std::vector<int> perm;
            if (perm_seed_ > 0) {
                perm.resize(p0.size());
                for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int)i;
                std::shuffle(perm.begin(), perm.end(), perm_rng_);
                std::vector<cv::Point2f> q0(p0.size()), q1(p1.size());
                for (size_t i = 0; i < perm.size(); ++i) { q0[i] = p0[perm[i]]; q1[i] = p1[perm[i]]; }
                p0.swap(q0); p1.swap(q1);
            }
            try {
                const int method = (rej_method_ == 1) ? static_cast<int>(cv::USAC_MAGSAC)
                                                      : static_cast<int>(cv::FM_RANSAC);
                cv::Mat F = cv::findFundamentalMat(p0, p1, method,
                                                   ransac_th_, rej_conf_, rej_iters_, inl);
                if (F.empty()) inl.clear();
            } catch (const cv::Exception&) {
                inl.clear();   // degenerate config -> keep what we have
            }
            if (!perm.empty() && !inl.empty()) {
                std::vector<unsigned char> back(inl.size());
                for (size_t i = 0; i < perm.size(); ++i) back[perm[i]] = inl[i];
                inl.swap(back);
            }
        }
        if (!inl.empty()) {
            std::vector<cv::Point2f> kept;
            kept.reserve(cur_pts.size());
            for (size_t i = 0; i < cur_pts.size(); ++i)
                if (inl[i]) kept.push_back(cur_pts[i]);
            cur_pts.swap(kept);
            filter(inl);
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
    for (const auto& p : prev_pts_)
        cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), min_dist_, 0, -1);

    auto add_pt = [&](const cv::Point2f& p) {
        prev_pts_.push_back(p);
        ids_.push_back(next_id_++);
        track_cnt_.push_back(1);
    };

    if (grid_x_ <= 1 && grid_y_ <= 1) {
        // GLOBAL extraction (default): the strongest corners image-wide, subject only to
        // the min-distance mask. Simple, but corners clump in the highest-texture region.
        std::vector<cv::Point2f> new_pts;
        cv::goodFeaturesToTrack(img, new_pts, n_need, quality_, (double)min_dist_,
                                mask, block_size_);
        for (const auto& p : new_pts) add_pt(p);
        return;
    }

    // GRID-UNIFORM extraction (OpenVINS-style): give each of the grid_x x grid_y image
    // cells its own detection quota, so features cover the frame instead of clumping where
    // texture is strongest. That spatial spread is what conditions the pose estimate under
    // fast rotation (the source of the V1_01 over-confidence, per the observability study).
    // Points are drawn into the mask as they are added, so the min-distance spacing holds
    // across cell boundaries too. Deterministic (goodFeaturesToTrack + fixed cell order).
    const int W = img.cols, H = img.rows;
    const int cw = (W + grid_x_ - 1) / grid_x_;
    const int ch = (H + grid_y_ - 1) / grid_y_;
    const int cells = grid_x_ * grid_y_;
    const int per_cell = std::max(1, (max_pts_ + cells - 1) / cells);

    std::vector<int> have(cells, 0);            // tracks already present per cell
    for (const auto& p : prev_pts_) {
        const int gx = std::min(grid_x_ - 1, (int)p.x / cw);
        const int gy = std::min(grid_y_ - 1, (int)p.y / ch);
        have[gy * grid_x_ + gx]++;
    }

    int added = 0;
    for (int gy = 0; gy < grid_y_ && added < n_need; ++gy)
        for (int gx = 0; gx < grid_x_ && added < n_need; ++gx) {
            const int c = gy * grid_x_ + gx;
            const int need = std::min(per_cell - have[c], n_need - added);
            if (need <= 0) continue;
            cv::Rect roi(gx * cw, gy * ch, std::min(cw, W - gx * cw), std::min(ch, H - gy * ch));
            if (roi.width < block_size_ || roi.height < block_size_) continue;
            std::vector<cv::Point2f> cell_pts;
            cv::goodFeaturesToTrack(img(roi), cell_pts, need, quality_, (double)min_dist_,
                                    mask(roi), block_size_);
            for (cv::Point2f p : cell_pts) {
                p.x += roi.x; p.y += roi.y;
                add_pt(p);
                cv::circle(mask, cv::Point(cvRound(p.x), cvRound(p.y)), min_dist_, 0, -1);
                if (++added >= n_need) break;
            }
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

// Translation-only RANSAC with a known rotation (Kimera's 2-point mono check). The gyro gives
// the inter-frame camera rotation, so a static point's bearings satisfy x1 ~ R x0 + t with
// R = dR_cam^T and the epipolar constraint x1^T [t]x R x0 = t . ((R x0) x x1) = 0: every
// correspondence is one linear constraint a_i on the translation direction, and two of them
// fix it (t = a_i x a_j). A fixed number of hypotheses (no adaptive early stop) keeps the
// consensus set stable, the winner is refined by least squares on its inliers, and a point is
// an inlier when its Sampson distance, in pixels, is below ransac_thresh. Pure rotation makes
// every a_i small, so every point consistent with the rotation is kept.
std::vector<unsigned char> FeatureTracker::two_point_inliers(
        const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1,
        const Eigen::Matrix3d& dR_body) const {
    static constexpr int kHypotheses = 200;
    const size_t n = p0.size();
    const Eigen::Matrix3d R = (R_cb_ * dR_body * R_bc_).transpose();   // prev cam -> cur cam
    const double f = 0.5 * (fx_ + fy_);
    std::vector<Eigen::Vector3d> x1(n), y(n), a(n);
    for (size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d x0((p0[i].x - cx_) / fx_, (p0[i].y - cy_) / fy_, 1.0);
        x1[i] = Eigen::Vector3d((p1[i].x - cx_) / fx_, (p1[i].y - cy_) / fy_, 1.0);
        y[i] = R * x0;
        a[i] = y[i].cross(x1[i]);
    }
    auto sampson_px = [&](const Eigen::Vector3d& t, size_t i) {
        const double r = t.dot(a[i]);                         // x1^T E x0, E = [t]x R
        const Eigen::Vector3d ex = t.cross(y[i]);             // E x0
        const Eigen::Vector3d ey = R.transpose() * x1[i].cross(t);   // E^T x1
        const double den = ex.x() * ex.x() + ex.y() * ex.y() + ey.x() * ey.x() + ey.y() * ey.y();
        return den > 1e-18 ? f * std::abs(r) / std::sqrt(den) : 0.0;
    };
    auto consensus = [&](const Eigen::Vector3d& t, std::vector<unsigned char>& m) {
        int c = 0;
        for (size_t i = 0; i < n; ++i) { m[i] = sampson_px(t, i) < ransac_th_; c += m[i]; }
        return c;
    };

    std::vector<unsigned char> best(n, 1), cur(n);
    int best_n = -1;
    Eigen::Vector3d best_t = Eigen::Vector3d::UnitZ();
    cv::RNG& rng = cv::theRNG();
    for (int h = 0; h < kHypotheses; ++h) {
        const int i = rng.uniform(0, (int)n);
        int j = rng.uniform(0, (int)n - 1);
        if (j >= i) ++j;
        Eigen::Vector3d t = a[i].cross(a[j]);
        if (t.norm() < 1e-12) continue;
        t.normalize();
        const int c = consensus(t, cur);
        if (c > best_n) { best_n = c; best_t = t; best.swap(cur); cur.resize(n); }
    }
    if (best_n < 0) return std::vector<unsigned char>(n, 1);

    // Refine the direction on the consensus set: the null vector of the stacked a_i.
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < n; ++i) if (best[i]) M += a[i] * a[i].transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
    const Eigen::Vector3d t_ref = es.eigenvectors().col(0);
    std::vector<unsigned char> ref(n);
    return consensus(t_ref, ref) >= best_n ? ref : best;
}

Observation FeatureTracker::make_observation() const {
    Observation o;
    const int N = (int)prev_pts_.size();
    o.ids = ids_;
    o.track_cnt = track_cnt_;
    o.fb_res.resize(N);
    for (int j = 0; j < N; ++j) {
        auto it = fb_last_.find(ids_[j]);
        o.fb_res[j] = (it != fb_last_.end()) ? it->second : 0.0;  // fresh track -> 0 (full trust)
    }
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