#include "init_sfm.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <optional>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace vio {

// ---- SE(3) helpers ---------------------------------------------------------
static Eigen::Matrix4d se3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

Eigen::Matrix4d inv_se3(const Eigen::Matrix4d& T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    Ti.block<3, 3>(0, 0) = R.transpose();
    Ti.block<3, 1>(0, 3) = -R.transpose() * t;
    return Ti;
}

// ---- intersect1d-like helper -----------------------------------------------
struct Intersect {
    std::vector<int64_t> common;
    std::vector<int> ia;
    std::vector<int> ib;
};

static Intersect intersect1d(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    Intersect out;
    std::unordered_map<int64_t, int> idx_b;
    idx_b.reserve(b.size());
    for (int j = 0; j < (int)b.size(); ++j) idx_b[b[j]] = j;
    for (int i = 0; i < (int)a.size(); ++i) {
        auto it = idx_b.find(a[i]);
        if (it != idx_b.end()) {
            out.common.push_back(a[i]);
            out.ia.push_back(i);
            out.ib.push_back(it->second);
        }
    }
    return out;
}

// ---- look up a point by feature id within a KF -----------------------------
static std::optional<Eigen::Vector2d> pt_norm(const KeyframeEntry& f, int64_t fid) {
    for (size_t i = 0; i < f.ids.size(); ++i) {
        if (f.ids[i] == fid) {
            return Eigen::Vector2d(f.pts_norm(i, 0), f.pts_norm(i, 1));
        }
    }
    return std::nullopt;
}

// ---- DLT triangulation -----------------------------------------------------
static Eigen::Vector3d triangulate_dlt(const Eigen::Vector2d& p1n,
                                       const Eigen::Vector2d& p2n,
                                       const Eigen::Matrix<double, 3, 4>& P1,
                                       const Eigen::Matrix<double, 3, 4>& P2) {
    Eigen::Matrix4d A;
    A.row(0) = p1n(0) * P1.row(2) - P1.row(0);
    A.row(1) = p1n(1) * P1.row(2) - P1.row(1);
    A.row(2) = p2n(0) * P2.row(2) - P2.row(0);
    A.row(3) = p2n(1) * P2.row(2) - P2.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    return (X.head<3>() / X(3));
}

static bool good_point(const Eigen::Vector3d& X,
                       const Eigen::Matrix4d& Tcw1,
                       const Eigen::Matrix4d& Tcw2) {
    Eigen::Vector3d X1 = Tcw1.block<3, 3>(0, 0) * X + Tcw1.block<3, 1>(0, 3);
    Eigen::Vector3d X2 = Tcw2.block<3, 3>(0, 0) * X + Tcw2.block<3, 1>(0, 3);
    return X1.z() > 0 && X2.z() > 0 && X.norm() < 200.0;
}

// ---- find reference pair ---------------------------------------------------
static bool find_ref_pair(const std::vector<KeyframeEntry>& frames,
                          double min_par, int min_count, int& l_out) {
    int N = (int)frames.size();
    for (int li = 0; li < N - 1; ++li) {
        Intersect it = intersect1d(frames[li].ids, frames[N - 1].ids);
        if ((int)it.common.size() < min_count) continue;
        double sum_d = 0.0;
        for (size_t k = 0; k < it.common.size(); ++k) {
            double dx = frames[li].pts_norm(it.ia[k], 0) - frames[N - 1].pts_norm(it.ib[k], 0);
            double dy = frames[li].pts_norm(it.ia[k], 1) - frames[N - 1].pts_norm(it.ib[k], 1);
            sum_d += std::sqrt(dx * dx + dy * dy);
        }
        double mean_par = sum_d / (double)it.common.size();
        if (mean_par >= min_par) {
            l_out = li;
            return true;
        }
    }
    l_out = 0;
    return false;
}

// ---- essential matrix ------------------------------------------------------
struct EResult {
    bool ok;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    std::vector<int64_t> common_ids;
    std::vector<unsigned char> inlier_mask;
};

static EResult solve_essential(const KeyframeEntry& f1, const KeyframeEntry& f2,
                               const CamCalib& calib) {
    EResult er;
    er.ok = false;
    Intersect it = intersect1d(f1.ids, f2.ids);
    er.common_ids = it.common;
    if (it.common.size() < 8) return er;

    std::vector<cv::Point2f> p1n, p2n;
    p1n.reserve(it.common.size());
    p2n.reserve(it.common.size());
    for (size_t k = 0; k < it.common.size(); ++k) {
        p1n.emplace_back((float)f1.pts_norm(it.ia[k], 0), (float)f1.pts_norm(it.ia[k], 1));
        p2n.emplace_back((float)f2.pts_norm(it.ib[k], 0), (float)f2.pts_norm(it.ib[k], 1));
    }

    double fx = calib.K(0, 0);
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat mask;
    cv::Mat E;
    try {
        E = cv::findEssentialMat(p1n, p2n, I, cv::RANSAC, 0.99, 1.0 / fx, mask);
    } catch (const cv::Exception&) {
        return er;
    }
    if (E.empty() || mask.empty()) return er;

    er.inlier_mask.assign(mask.rows, 0);
    int n_in = 0;
    for (int i = 0; i < mask.rows; ++i) {
        er.inlier_mask[i] = mask.at<unsigned char>(i, 0);
        if (er.inlier_mask[i]) n_in++;
    }
    if (n_in < 8) return er;

    std::vector<cv::Point2f> p1_in, p2_in;
    for (int i = 0; i < (int)er.inlier_mask.size(); ++i)
        if (er.inlier_mask[i]) {
            p1_in.push_back(p1n[i]);
            p2_in.push_back(p2n[i]);
        }

    cv::Mat R_cv, t_cv;
    int n = cv::recoverPose(E, p1_in, p2_in, I, R_cv, t_cv);
    if (n < 8) return er;

    cv::cv2eigen(R_cv, er.R);
    Eigen::Vector3d tt;
    cv::cv2eigen(t_cv, tt);
    er.t = tt;
    er.ok = true;
    return er;
}

// ---- PnP for one KF --------------------------------------------------------
static std::optional<Eigen::Matrix4d>
pnp_frame(const KeyframeEntry& f,
          const std::map<int64_t, Eigen::Vector3d>& points,
          const CamCalib& calib) {
    if (points.empty()) return std::nullopt;

    std::vector<int64_t> known_ids;
    known_ids.reserve(points.size());
    for (const auto& kv : points) known_ids.push_back(kv.first);

    Intersect it = intersect1d(f.ids, known_ids);
    if (it.common.size() < 6) return std::nullopt;

    std::vector<cv::Point2f> img_pix;
    std::vector<cv::Point3f> world_pts;
    img_pix.reserve(it.common.size());
    world_pts.reserve(it.common.size());
    for (size_t k = 0; k < it.common.size(); ++k) {
        int i = it.ia[k];
        img_pix.emplace_back((float)f.pts_raw(i, 0), (float)f.pts_raw(i, 1));
        const auto& W = points.at(it.common[k]);
        world_pts.emplace_back((float)W.x(), (float)W.y(), (float)W.z());
    }

    cv::Mat K_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) K_cv.at<double>(i, j) = calib.K(i, j);

    cv::Mat dist_cv = cv::Mat::zeros(4, 1, CV_64F);
    for (int i = 0; i < 4; ++i) dist_cv.at<double>(i, 0) = calib.dist(i);

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool ok = false;
    try {
        ok = cv::solvePnPRansac(world_pts, img_pix, K_cv, dist_cv,
                                rvec, tvec, false, 4000, 6.0f, 0.90,
                                inliers, cv::SOLVEPNP_EPNP);
    } catch (const cv::Exception& e) {
        std::cerr << "init_sfm: cv::solvePnPRansac failed: " << e.what() << "\n";
        return std::nullopt;
    }
    if (!ok || inliers.size() < 6) return std::nullopt;

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cv::cv2eigen(R_cv, R);
    cv::cv2eigen(tvec, t);

    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    Tcw.block<3, 3>(0, 0) = R;
    Tcw.block<3, 1>(0, 3) = t;
    return Tcw;
}

// ---- triangulate more points -----------------------------------------------
static void triangulate_more(std::map<int64_t, Eigen::Vector3d>& points,
                             const std::vector<KeyframeEntry>& frames,
                             const std::vector<std::optional<Eigen::Matrix4d>>& Tcw,
                             int k_new) {
    if (!Tcw[k_new].has_value()) return;
    int N = (int)frames.size();

    std::vector<int64_t> known_ids;
    known_ids.reserve(points.size());
    for (const auto& kv : points) known_ids.push_back(kv.first);
    std::sort(known_ids.begin(), known_ids.end());

    for (int k_other = 0; k_other < N; ++k_other) {
        if (k_other == k_new || !Tcw[k_other].has_value()) continue;

        // common ids of k_new & k_other
        std::vector<int64_t> sa = frames[k_new].ids, sb = frames[k_other].ids;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        std::vector<int64_t> common;
        std::set_intersection(sa.begin(), sa.end(), sb.begin(), sb.end(),
                              std::back_inserter(common));

        // remove those already triangulated
        std::vector<int64_t> new_ids;
        std::set_difference(common.begin(), common.end(),
                            known_ids.begin(), known_ids.end(),
                            std::back_inserter(new_ids));

        for (int64_t fid : new_ids) {
            auto p1 = pt_norm(frames[k_new], fid);
            auto p2 = pt_norm(frames[k_other], fid);
            if (!p1.has_value() || !p2.has_value()) continue;
            Eigen::Matrix<double, 3, 4> P1 = Tcw[k_new]->block<3, 4>(0, 0);
            Eigen::Matrix<double, 3, 4> P2 = Tcw[k_other]->block<3, 4>(0, 0);
            Eigen::Vector3d X = triangulate_dlt(*p1, *p2, P1, P2);
            if (good_point(X, *Tcw[k_new], *Tcw[k_other])) {
                points[fid] = X;
            }
        }
        // refresh known_ids
        known_ids.clear();
        for (const auto& kv : points) known_ids.push_back(kv.first);
        std::sort(known_ids.begin(), known_ids.end());
    }
}

// ---- main entry ------------------------------------------------------------
SfmResult init_sfm(const std::vector<KeyframeEntry>& frames, const CamCalib& calib) {
    SfmResult res;
    res.frames = frames;
    res.calib = calib;

    int N = (int)frames.size();
    if (N < 3) {
        std::cout << "init_sfm: need >=3 keyframes, got " << N << "\n";
        return res;
    }

    double fx = calib.K(0, 0);

    int l = 0;
    if (!find_ref_pair(frames, 30.0 / fx, 20, l)) {
        std::cout << "init_sfm: no reference pair with enough parallax\n";
        return res;
    }
    std::cout << "init_sfm: reference pair = (" << l << ", " << (N - 1) << ") of " << N << "\n";

    EResult er = solve_essential(frames[l], frames[N - 1], calib);
    if (!er.ok) {
        std::cout << "init_sfm: essential matrix failed\n";
        return res;
    }

    std::vector<std::optional<Eigen::Matrix4d>> Tcw(N, std::nullopt);
    Tcw[l] = Eigen::Matrix4d::Identity();
    Tcw[N - 1] = se3(er.R, er.t);

    std::map<int64_t, Eigen::Vector3d> points;
    for (size_t k = 0; k < er.common_ids.size(); ++k) {
        if (!er.inlier_mask[k]) continue;
        int64_t fid = er.common_ids[k];
        auto p1 = pt_norm(frames[l], fid);
        auto p2 = pt_norm(frames[N - 1], fid);
        if (!p1.has_value() || !p2.has_value()) continue;
        Eigen::Matrix<double, 3, 4> P1 = Tcw[l]->block<3, 4>(0, 0);
        Eigen::Matrix<double, 3, 4> P2 = Tcw[N - 1]->block<3, 4>(0, 0);
        Eigen::Vector3d X = triangulate_dlt(*p1, *p2, P1, P2);
        if (good_point(X, *Tcw[l], *Tcw[N - 1])) {
            points[fid] = X;
        }
    }
    std::cout << "init_sfm: seeded " << points.size() << " points from ("
              << l << "," << (N - 1) << ")\n";
    if (points.size() < 10) {
        std::cout << "init_sfm: too few seed points\n";
        return res;
    }

    // PnP outward
    std::vector<int> order;
    for (int k = l - 1; k >= 0; --k) order.push_back(k);
    for (int k = l + 1; k < N - 1; ++k) order.push_back(k);
    for (int k : order) {
        if (!Tcw[k].has_value()) {
            auto T = pnp_frame(frames[k], points, calib);
            if (!T.has_value()) {
                std::cout << "init_sfm: PnP failed at frame index " << k << "\n";
                return res;
            }
            Tcw[k] = T;
        }
        triangulate_more(points, frames, Tcw, k);
    }

    for (int k = 0; k < N; ++k) {
        if (!Tcw[k].has_value()) {
            std::cout << "init_sfm: frame index " << k << " has no pose\n";
            return res;
        }
    }

    res.T_wc.resize(N);
    res.Tcw.resize(N);
    for (int k = 0; k < N; ++k) {
        res.Tcw[k] = *Tcw[k];
        res.T_wc[k] = inv_se3(*Tcw[k]);
    }
    res.points = points;
    res.ref = l;
    res.success = true;
    std::cout << "init_sfm: SUCCESS  poses=" << N
              << "  points=" << points.size() << "\n";
    return res;
}

} // namespace vio