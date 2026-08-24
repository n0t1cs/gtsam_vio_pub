#pragma once
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "BRIEF.h"   // DVision::BRIEF (loop/ThirdParty/DVision on the include path)

namespace vio {

// Thin wrapper over DVision::BRIEF that loads the SAME sampling pattern used to
// build brief_k10L6.bin, so the descriptors are compatible with that vocabulary.
// (Mirrors VINS loop_fusion's BriefExtractor, decoupled from its KeyFrame.)
class BriefExtractor {
public:
    explicit BriefExtractor(const std::string& pattern_file);

    // descriptors[i] corresponds 1:1 to keys[i] (compute() resizes to keys.size()
    // and leaves out-of-border sample bits unset; it does NOT drop keypoints), so
    // parallel arrays (e.g. 3D points) stay aligned. `im` MUST be CV_8UC1.
    void operator()(const cv::Mat& im,
                    std::vector<cv::KeyPoint>& keys,
                    std::vector<DVision::BRIEF::bitset>& descriptors) const;

private:
    DVision::BRIEF brief_;
};

} // namespace vio
