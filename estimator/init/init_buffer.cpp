#include "init_buffer.h"
#include <algorithm>
#include <unordered_map>
#include <cmath>

namespace vio {

InitBuffer::InitBuffer(int window_size, double kf_parallax_px, int kf_min_common)
    : window_size_(window_size),
      kf_parallax_px_(kf_parallax_px),
      kf_min_common_(kf_min_common) {}

bool InitBuffer::try_add(int idx, double t, const Observation& obs) {
    if ((int)obs.ids.size() < kf_min_common_) return false;

    if (frames.empty()) {
        append(idx, t, obs);
        return true;
    }

    const auto& last = frames.back();

    // intersect ids and gather paired pixel coords.
    std::unordered_map<int64_t, int> last_idx_of;
    last_idx_of.reserve(last.ids.size());
    for (size_t i = 0; i < last.ids.size(); ++i) last_idx_of[last.ids[i]] = (int)i;

    int n_common = 0;
    double sum_d = 0.0;
    for (size_t i = 0; i < obs.ids.size(); ++i) {
        auto it = last_idx_of.find(obs.ids[i]);
        if (it == last_idx_of.end()) continue;
        int j = it->second;
        double dx = last.pts_raw(j, 0) - obs.pts_raw(i, 0);
        double dy = last.pts_raw(j, 1) - obs.pts_raw(i, 1);
        sum_d += std::sqrt(dx * dx + dy * dy);
        n_common++;
    }

    if (n_common < kf_min_common_) {
        append(idx, t, obs);
        return true;
    }
    double mean_par = sum_d / (double)n_common;
    if (mean_par >= kf_parallax_px_) {
        append(idx, t, obs);
        return true;
    }
    return false;
}

void InitBuffer::shift() {
    if (!frames.empty()) frames.erase(frames.begin());
}

void InitBuffer::reset() { frames.clear(); }

void InitBuffer::append(int idx, double t, const Observation& obs) {
    KeyframeEntry kf;
    kf.idx = idx;
    kf.t = t;
    kf.ids = obs.ids;
    kf.pts_raw = obs.pts_raw;
    kf.pts_norm = obs.pts_norm;
    frames.push_back(std::move(kf));
}

} // namespace vio