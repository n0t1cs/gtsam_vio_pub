#include "load.h"

#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace vio {

namespace fs = std::filesystem;

// ---- small CSV helper ------------------------------------------------------
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static void check_strictly_increasing(const std::vector<int64_t>& t, const std::string& src) {
    for (size_t i = 1; i < t.size(); ++i) {
        if (t[i] <= t[i - 1]) {
            throw std::runtime_error("Timestamps not strictly increasing in " + src);
        }
    }
}

// ---- calibration loaders ---------------------------------------------------
CamCalib load_cam_calib(const std::string& yaml_path) {
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    auto intr = cfg["intrinsics"].as<std::vector<double>>();
    auto dist = cfg["distortion_coefficients"].as<std::vector<double>>();
    auto T_data = cfg["T_BS"]["data"].as<std::vector<double>>();
    auto res = cfg["resolution"].as<std::vector<int>>();

    if (dist.size() != 4) throw std::runtime_error("Expected 4 distortion coefficients");
    if (T_data.size() != 16) throw std::runtime_error("Expected 16 entries in T_BS.data");

    CamCalib c;
    c.K.setIdentity();
    c.K(0, 0) = intr[0]; c.K(1, 1) = intr[1];
    c.K(0, 2) = intr[2]; c.K(1, 2) = intr[3];
    c.dist = Eigen::Vector4d(dist[0], dist[1], dist[2], dist[3]);
    c.T_BC = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_data.data());
    c.width = res[0];
    c.height = res[1];
    c.rate_hz = cfg["rate_hz"].as<int>();
    return c;
}

ImuCalib load_imu_calib(const std::string& yaml_path) {
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    auto T_data = cfg["T_BS"]["data"].as<std::vector<double>>();
    if (T_data.size() != 16) throw std::runtime_error("Expected 16 entries in T_BS.data");

    ImuCalib im;
    im.gyro_noise_density = cfg["gyroscope_noise_density"].as<double>();
    im.gyro_random_walk = cfg["gyroscope_random_walk"].as<double>();
    im.accel_noise_density = cfg["accelerometer_noise_density"].as<double>();
    im.accel_random_walk = cfg["accelerometer_random_walk"].as<double>();
    im.T_BI = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_data.data());
    im.rate_hz = cfg["rate_hz"].as<int>();
    return im;
}

// ---- CSV loaders -----------------------------------------------------------
std::vector<Frame> load_cam_index(const std::string& csv_path, const std::string& img_dir) {
    std::ifstream f(csv_path);
    if (!f.is_open()) throw std::runtime_error("Could not open " + csv_path);

    std::vector<Frame> frames;
    std::vector<int64_t> times;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tok = split_csv(line);
        if (tok.size() < 2) continue;
        Frame fr;
        fr.t_ns = std::stoll(tok[0]);
        fr.path = (fs::path(img_dir) / tok[1]).string();
        frames.push_back(std::move(fr));
        times.push_back(frames.back().t_ns);
    }
    check_strictly_increasing(times, csv_path);
    return frames;
}

std::vector<ImuSample> load_imu(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open()) throw std::runtime_error("Could not open " + csv_path);

    std::vector<ImuSample> imu;
    std::vector<int64_t> times;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tok = split_csv(line);
        if (tok.size() < 7) continue;
        ImuSample s;
        s.t_ns = std::stoll(tok[0]);
        s.w << std::stod(tok[1]), std::stod(tok[2]), std::stod(tok[3]);
        s.a << std::stod(tok[4]), std::stod(tok[5]), std::stod(tok[6]);
        imu.push_back(s);
        times.push_back(s.t_ns);
    }
    check_strictly_increasing(times, csv_path);
    return imu;
}

std::vector<GtSample> load_groundtruth(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open()) throw std::runtime_error("Could not open " + csv_path);

    std::vector<GtSample> gt;
    std::vector<int64_t> times;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto tok = split_csv(line);
        if (tok.size() < 17) continue;
        GtSample g;
        g.t_ns = std::stoll(tok[0]);
        g.p << std::stod(tok[1]), std::stod(tok[2]), std::stod(tok[3]);
        g.q_wxyz << std::stod(tok[4]), std::stod(tok[5]), std::stod(tok[6]), std::stod(tok[7]);
        g.v << std::stod(tok[8]), std::stod(tok[9]), std::stod(tok[10]);
        g.bg << std::stod(tok[11]), std::stod(tok[12]), std::stod(tok[13]);
        g.ba << std::stod(tok[14]), std::stod(tok[15]), std::stod(tok[16]);
        gt.push_back(g);
        times.push_back(g.t_ns);
    }
    check_strictly_increasing(times, csv_path);
    return gt;
}

cv::Mat load_image(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Could not read image: " + path);
    return img;
}

EurocData load_dataset(const std::string& root) {
    fs::path mav0 = fs::path(root) / "mav0";
    if (!fs::is_directory(mav0)) throw std::runtime_error("Expected EuRoC layout at " + mav0.string());

    EurocData d;
    d.cam_calib = load_cam_calib((mav0 / "cam0" / "sensor.yaml").string());
    d.imu_calib = load_imu_calib((mav0 / "imu0" / "sensor.yaml").string());
    d.frames = load_cam_index((mav0 / "cam0" / "data.csv").string(),
                              (mav0 / "cam0" / "data").string());
    d.imu = load_imu((mav0 / "imu0" / "data.csv").string());
    d.gt = load_groundtruth((mav0 / "state_groundtruth_estimate0" / "data.csv").string());
    return d;
}

} // namespace vio