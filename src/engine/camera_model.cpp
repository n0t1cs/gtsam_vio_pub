#include "camera_model.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gvio {

OpenCvIntrinsics load_opencv_intrinsics(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("cannot open camera calibration " + path);
    OpenCvIntrinsics c;
    fs["M"] >> c.K;
    fs["D"] >> c.D;
    c.width  = (int)fs["width"].real();
    c.height = (int)fs["height"].real();
    c.model  = (std::string)fs["distortion_model"];
    fs.release();
    if (c.K.rows != 3 || c.K.cols != 3 || c.D.empty() || c.width <= 0 || c.height <= 0)
        throw std::runtime_error("malformed camera calibration " + path);
    c.K.convertTo(c.K, CV_64F);
    c.D.convertTo(c.D, CV_64F);
    if (c.model == "fisheye") c.D = c.D.reshape(1, 4);
    return c;
}

Eigen::Matrix3d rpy_deg_to_R(double r, double p, double y) {
    const double d = M_PI / 180.0;
    return (Eigen::AngleAxisd(r * d, Eigen::Vector3d::UnitX())
          * Eigen::AngleAxisd(p * d, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(y * d, Eigen::Vector3d::UnitZ())).toRotationMatrix();
}

// ModalAI .conf files are JSON behind a /** ... */ header and are tab-indented. JSON is YAML flow
// style, so after removing comments and tabs yaml-cpp reads it.
static std::string strip_json_comments(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            out += c;
            if (c == '\\' && i + 1 < s.size()) out += s[++i];
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; out += c; continue; }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
            const size_t e = s.find("*/", i + 2);
            if (e == std::string::npos) break;
            i = e + 1;
            continue;
        }
        if (c == '/' && i + 1 < s.size() && s[i + 1] == '/') {
            while (i < s.size() && s[i] != '\n') ++i;
            out += '\n';
            continue;
        }
        out += (c == '\t') ? ' ' : c;
    }
    return out;
}

Eigen::Matrix4d load_voxl_extrinsic(const std::string& path, const std::string& parent,
                                    const std::string& child) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open extrinsics " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const YAML::Node root = YAML::Load(strip_json_comments(ss.str()));
    const YAML::Node list = root["extrinsics"];
    if (!list || !list.IsSequence()) throw std::runtime_error("no extrinsics list in " + path);
    for (const auto& e : list) {
        if (e["parent"].as<std::string>() != parent || e["child"].as<std::string>() != child) continue;
        const auto t = e["T_child_wrt_parent"].as<std::vector<double>>();
        const auto rpy = e["RPY_parent_to_child"].as<std::vector<double>>();
        if (t.size() != 3 || rpy.size() != 3) throw std::runtime_error("malformed extrinsic in " + path);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rpy_deg_to_R(rpy[0], rpy[1], rpy[2]);
        T.block<3, 1>(0, 3) = Eigen::Vector3d(t[0], t[1], t[2]);
        return T;
    }
    throw std::runtime_error("extrinsic " + parent + " -> " + child + " not in " + path);
}

void Undistorter::init_fisheye(const OpenCvIntrinsics& c, double balance, const Eigen::Vector4d* pinned) {
    if (c.model != "fisheye") throw std::runtime_error("expected a fisheye calibration, got " + c.model);
    const cv::Size size(c.width, c.height);
    cv::Mat newK;
    if (pinned) {
        newK = (cv::Mat_<double>(3, 3) << (*pinned)(0), 0, (*pinned)(2), 0, (*pinned)(1), (*pinned)(3), 0, 0, 1);
    } else {
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(c.K, c.D, size, cv::Matx33d::eye(), newK,
                                                                balance, cv::Size(), 1.0);
    }
    cv::fisheye::initUndistortRectifyMap(c.K, c.D, cv::Matx33d::eye(), newK, size, CV_16SC2,
                                         map1_, map2_);
    for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) K_(r, k) = newK.at<double>(r, k);
    w_ = c.width; h_ = c.height;
    remap_ = true;
}

void Undistorter::init_passthrough(const Eigen::Matrix3d& K, int width, int height) {
    K_ = K; w_ = width; h_ = height;
    remap_ = false;
}

cv::Mat Undistorter::apply(const cv::Mat& gray) const {
    if (!remap_) return gray;
    cv::Mat und;
    cv::remap(gray, und, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return und;
}

vio::CamCalib make_cam_calib(const Undistorter& u, const Eigen::Matrix4d& T_BC, int rate_hz) {
    vio::CamCalib c;
    c.K = u.K();
    c.dist.setZero();
    c.T_BC = T_BC;
    c.width = u.width();
    c.height = u.height();
    c.rate_hz = rate_hz;
    return c;
}

} // namespace gvio
