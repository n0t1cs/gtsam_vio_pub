#pragma once
#include "vio_types.h"

namespace vio {

class InitBuffer {
public:
    InitBuffer(int window_size = 10,
               double kf_parallax_px = 10.0,
               int kf_min_common = 20);

    // Returns true iff this frame was accepted as a keyframe.
    bool try_add(int idx, double t, const Observation& obs);

    bool is_full() const { return (int)frames.size() >= window_size_; }
    void shift();    // drop oldest
    void reset();

    std::vector<KeyframeEntry> frames;

private:
    int window_size_;
    double kf_parallax_px_;
    int kf_min_common_;

    void append(int idx, double t, const Observation& obs);
};

} // namespace vio