#include "loop_database.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

namespace vio {

// Detection tuning now lives in LoopParams (lp_), threaded from VioParams; the
// struct defaults reproduce the previous VINS loop_fusion values. The DBoW query
// and BRIEF matching normalize the px thresholds by the EuRoC focal (~460).
static const double kFocalPx = 460.0;

static inline int hamming(const DVision::BRIEF::bitset& a,
                          const DVision::BRIEF::bitset& b) {
    return (int)(a ^ b).count();
}
static cv::Mat eig2cv33(const Eigen::Matrix3d& M) {
    cv::Mat m(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) m.at<double>(r, c) = M(r, c);
    return m;
}
static Eigen::Matrix3d cv2eig33(const cv::Mat& m) {
    Eigen::Matrix3d M;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M(r, c) = m.at<double>(r, c);
    return M;
}

LoopDatabase::LoopDatabase(const std::string& vocab_path, const LoopParams& lp) : lp_(lp) {
    // TemplatedVocabulary auto-detects the binary format and calls loadBin().
    voc_.reset(new BriefVocabulary(vocab_path));
    db_.setVocabulary(*voc_, false, 0);
    std::cout << "[loop] loaded BRIEF vocabulary (" << voc_->size()
              << " words) from " << vocab_path << "\n";
}

void LoopDatabase::add(const LoopKeyframe& kf) {
    db_.add(kf.brief_desc);
    kfs_.push_back(kf);
}

bool LoopDatabase::verify_loop(const LoopKeyframe& cur, const LoopKeyframe& old,
                               const Eigen::Matrix4d& T_BC, LoopEdge& edge) const {
    // 1) BRIEF-match cur's 3D-point descriptors against old's FAST descriptors.
    std::vector<cv::Point3f> matched_3d;
    std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;   // normalized image plane
    for (size_t i = 0; i < cur.point_desc.size(); ++i) {
        int best = lp_.hamming_th, bestj = -1;
        for (size_t j = 0; j < old.brief_desc.size(); ++j) {
            int d = hamming(cur.point_desc[i], old.brief_desc[j]);
            if (d < best) { best = d; bestj = (int)j; }
        }
        if (bestj >= 0) {
            matched_3d.push_back(cur.point_3d[i]);
            matched_2d_cur.push_back(cur.point_2d_norm[i]);
            matched_2d_old.push_back(old.keypoints_norm[bestj].pt);
        }
    }
    if ((int)matched_3d.size() < lp_.min_inliers) return false;

    // 1b) Fundamental-matrix RANSAC (2D-2D cur<->old) to drop gross mismatches
    //     before PnP (mirrors VINS findConnection). Needs >= 8 correspondences.
    if ((int)matched_3d.size() >= 8) {
        std::vector<uchar> fmask;
        cv::findFundamentalMat(matched_2d_cur, matched_2d_old, cv::FM_RANSAC,
                               lp_.fund_thresh_px / kFocalPx, 0.99, fmask);
        if ((int)fmask.size() == (int)matched_3d.size()) {
            std::vector<cv::Point3f> k3d; std::vector<cv::Point2f> kold;
            for (size_t i = 0; i < fmask.size(); ++i)
                if (fmask[i]) { k3d.push_back(matched_3d[i]); kold.push_back(matched_2d_old[i]); }
            matched_3d.swap(k3d);
            matched_2d_old.swap(kold);
        }
    }
    if ((int)matched_3d.size() < lp_.min_inliers) return false;

    // 2) PnP-RANSAC: recover the OLD camera pose (in cur's world) from cur's 3D
    //    points seen at old's 2D, initialized from old's VIO pose. K = identity
    //    because the 2D points are already normalized.
    const Eigen::Matrix3d R_BC = T_BC.block<3, 3>(0, 0);
    const Eigen::Vector3d t_BC = T_BC.block<3, 1>(0, 3);
    const Eigen::Matrix3d R_old_w = old.T_wb.block<3, 3>(0, 0);
    const Eigen::Vector3d t_old_w = old.T_wb.block<3, 1>(0, 3);
    const Eigen::Matrix3d Rwc = R_old_w * R_BC;
    const Eigen::Vector3d Twc = t_old_w + R_old_w * t_BC;
    const Eigen::Matrix3d Rinit = Rwc.transpose();
    const Eigen::Vector3d Pinit = -(Rinit * Twc);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F), D;
    cv::Mat rvec, tvec(3, 1, CV_64F);
    cv::Rodrigues(eig2cv33(Rinit), rvec);
    for (int r = 0; r < 3; ++r) tvec.at<double>(r) = Pinit(r);

    cv::Mat inliers;
    try {
        bool ok = cv::solvePnPRansac(matched_3d, matched_2d_old, K, D, rvec, tvec,
                                     true, 100, 10.0 / 460.0, 0.99, inliers);
        if (!ok) return false;
    } catch (const cv::Exception&) { return false; }
    if (inliers.rows < lp_.min_inliers) return false;

    // 3) Recover old BODY pose in cur's world frame, then the relative pose.
    cv::Mat Rcv; cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R_pnp = cv2eig33(Rcv);
    Eigen::Vector3d t_pnp(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    Eigen::Matrix3d R_w_cold = R_pnp.transpose();
    Eigen::Vector3d T_w_cold = R_w_cold * (-t_pnp);
    Eigen::Matrix3d PnP_R_old = R_w_cold * R_BC.transpose();      // old body in cur world
    Eigen::Vector3d PnP_T_old = T_w_cold - PnP_R_old * t_BC;

    const Eigen::Matrix3d R_cur = cur.T_wb.block<3, 3>(0, 0);
    const Eigen::Vector3d t_cur = cur.T_wb.block<3, 1>(0, 3);
    Eigen::Matrix3d rel_R = PnP_R_old.transpose() * R_cur;        // cur in old frame
    Eigen::Vector3d rel_t = PnP_R_old.transpose() * (t_cur - PnP_T_old);
    double rel_yaw = std::atan2(rel_R(1, 0), rel_R(0, 0)) * 180.0 / M_PI;
    if (rel_t.norm() > lp_.max_trans_m || std::abs(rel_yaw) > lp_.max_yaw_deg) return false;

    edge.index_cur = cur.index;
    edge.index_old = old.index;
    edge.inliers   = inliers.rows;
    edge.T_old_cur = Eigen::Matrix4d::Identity();
    edge.T_old_cur.block<3, 3>(0, 0) = rel_R;
    edge.T_old_cur.block<3, 1>(0, 3) = rel_t;
    return true;
}

std::vector<LoopEdge> LoopDatabase::query_loops(const LoopKeyframe& cur,
                                                const Eigen::Matrix4d& T_BC) {
    std::vector<LoopEdge> edges;
    const int idx = cur.index;                  // node id this KF will occupy
    if (idx <= lp_.skip_recent) return edges;   // not enough history yet

    DBoW2::QueryResults ret;
    // candidates restricted to entries with id <= idx - skip_recent (earlier KFs)
    db_.query(cur.brief_desc, ret, 4, idx - lp_.skip_recent);
    if (ret.empty() || ret[0].Score < lp_.score_best) return edges;

    // VINS min_index: among the candidates that clear the score + temporal gates,
    // verify the OLDEST first. The oldest match is the largest-baseline loop, so it
    // corrects the most accumulated drift; PnP + |t|/yaw still gate precision.
    std::vector<int> cand_ids;
    for (const auto& r : ret) {
        if (r.Score < lp_.score_cand) continue;
        if ((int)r.Id >= (int)kfs_.size()) continue;
        // DBoW's queryL1 always leaks the most-recent entry past max_id, so enforce
        // the temporal gap explicitly (kills consecutive-KF "loops").
        if (idx - (int)r.Id < lp_.skip_recent) continue;
        cand_ids.push_back((int)r.Id);
    }
    std::sort(cand_ids.begin(), cand_ids.end());   // oldest (largest-baseline) first
    for (int old_id : cand_ids) {
        LoopEdge e;
        if (verify_loop(cur, kfs_[(size_t)old_id], T_BC, e)) {
            edges.push_back(e);
            std::cout << "[loop] KF " << e.index_cur << " <-> " << e.index_old
                      << "  inliers=" << e.inliers
                      << "  |t|=" << e.T_old_cur.block<3, 1>(0, 3).norm() << "m\n";
            break;   // one loop per keyframe
        }
    }
    return edges;
}

} // namespace vio
