#include "brief_extractor.h"
#include <opencv2/core/core.hpp>
#include <stdexcept>

namespace vio {

BriefExtractor::BriefExtractor(const std::string& pattern_file) {
    // The pattern must match the one used to train brief_k10L6.bin, else the
    // descriptors are incompatible with the vocabulary.
    cv::FileStorage fs(pattern_file, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("BriefExtractor: cannot open pattern file " + pattern_file);

    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;  fs["x2"] >> x2;
    fs["y1"] >> y1;  fs["y2"] >> y2;
    brief_.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator()(const cv::Mat& im,
                                std::vector<cv::KeyPoint>& keys,
                                std::vector<DVision::BRIEF::bitset>& descriptors) const {
    brief_.compute(im, keys, descriptors);
}

} // namespace vio
