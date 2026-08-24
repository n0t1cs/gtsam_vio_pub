#pragma once
#include "vio_types.h"
#include <opencv2/core.hpp>
#include <string>

namespace vio {

CamCalib load_cam_calib(const std::string& yaml_path);
ImuCalib load_imu_calib(const std::string& yaml_path);
std::vector<Frame> load_cam_index(const std::string& csv_path, const std::string& img_dir);
std::vector<ImuSample> load_imu(const std::string& csv_path);
std::vector<GtSample> load_groundtruth(const std::string& csv_path);
cv::Mat load_image(const std::string& path);
EurocData load_dataset(const std::string& root);

} // namespace vio