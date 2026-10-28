// VERBATIM copies of cpp_gtsam_vio/main.cpp (integrate_imu_batch, check_imu_excitation,
// detect_stationary). Only the namespace qualifiers differ. See offline_helpers.h.
#include "offline_helpers.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace gvio {

using namespace vio;

// ---------- IMU batch -> backend ----------
void integrate_imu_batch(VioBackend& backend, const EurocData& data,
                         double t0_s, double t1_s, int64_t t0_ns_base) {
    int64_t t0_ns = t0_ns_base + (int64_t)(t0_s * 1e9);
    int64_t t1_ns = t0_ns_base + (int64_t)(t1_s * 1e9);

    auto lo = std::lower_bound(data.imu.begin(), data.imu.end(), t0_ns,
                               [](const ImuSample& s, int64_t v){ return s.t_ns < v; });
    auto hi = std::upper_bound(data.imu.begin(), data.imu.end(), t1_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });

    auto it_prev = lo;
    if (it_prev == hi) return;
    for (auto it = it_prev + 1; it != hi; ++it) {
        double dt = (it->t_ns - it_prev->t_ns) * 1e-9;
        if (dt > 0) backend.add_imu(dt, it->a, it->w);
        it_prev = it;
    }
}

bool check_imu_excitation(const std::vector<KeyframeEntry>& frames,
                          const EurocData& data,
                          double min_acc_var) {
    if (frames.size() < 2) return false;

    const int64_t t0_ns = data.frames.front().t_ns;
    const auto& imu = data.imu;

    // Use binary search since imu is sorted by t_ns.
    auto find_lo = [&](int64_t t) {
        return std::upper_bound(imu.begin(), imu.end(), t,
                                [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    };
    auto find_hi = [&](int64_t t) {
        // upper_bound on t_b -> all with t_ns <= t_b are before this iterator.
        return std::upper_bound(imu.begin(), imu.end(), t,
                                [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    };

    // Collect all accel samples across all consecutive intervals.
    std::vector<Eigen::Vector3d> all_acc;
    all_acc.reserve(2048);

    for (size_t i = 0; i + 1 < frames.size(); ++i) {
        int64_t t_a = t0_ns + (int64_t)std::llround(frames[i].t * 1e9);
        int64_t t_b = t0_ns + (int64_t)std::llround(frames[i + 1].t * 1e9);

        // Mask: t_ns > t_a AND t_ns <= t_b.
        auto lo = find_lo(t_a);   // first with t_ns > t_a
        auto hi = find_hi(t_b);   // first with t_ns > t_b
        if (std::distance(lo, hi) < 2) return false;

        for (auto it = lo; it != hi; ++it) all_acc.push_back(it->a);
    }

    if (all_acc.size() < 2) return false;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& a : all_acc) mean += a;
    mean /= (double)all_acc.size();

    Eigen::Vector3d var = Eigen::Vector3d::Zero();
    for (const auto& a : all_acc) {
        Eigen::Vector3d d = a - mean;
        var += d.cwiseProduct(d);
    }
    var /= (double)all_acc.size();   // population variance (matches np.var default)

    double total_var = var.sum();
    return total_var > min_acc_var;
}

// Stationary detector: true only when IMU is quiet AND tracked features barely
// move. The vision gate is what rejects constant-velocity motion (low accel
// variance but real translation -> features still move).
bool detect_stationary(const EurocData& data,
                       double t0_s, double t1_s, int64_t t0_ns_base,
                       const Observation& prev_obs,
                       const Observation& cur_obs,
                       double acc_var_th, double gyr_var_th, double disp_th) {
    int64_t t0_ns = t0_ns_base + (int64_t)(t0_s * 1e9);
    int64_t t1_ns = t0_ns_base + (int64_t)(t1_s * 1e9);
    auto lo = std::lower_bound(data.imu.begin(), data.imu.end(), t0_ns,
                               [](const ImuSample& s, int64_t v){ return s.t_ns < v; });
    auto hi = std::upper_bound(data.imu.begin(), data.imu.end(), t1_ns,
                               [](int64_t v, const ImuSample& s){ return v < s.t_ns; });
    int n = (int)std::distance(lo, hi);
    if (n < 5) return false;

    Eigen::Vector3d a_mean = Eigen::Vector3d::Zero(), w_mean = Eigen::Vector3d::Zero();
    for (auto it = lo; it != hi; ++it) { a_mean += it->a; w_mean += it->w; }
    a_mean /= n;  w_mean /= n;
    double a_var = 0.0, w_var = 0.0;
    for (auto it = lo; it != hi; ++it) {
        a_var += (it->a - a_mean).squaredNorm();
        w_var += (it->w - w_mean).squaredNorm();
    }
    a_var /= n;  w_var /= n;

    std::unordered_map<int64_t, Eigen::Vector2d> prev_map;
    prev_map.reserve(prev_obs.ids.size());
    for (size_t i = 0; i < prev_obs.ids.size(); ++i)
        prev_map[prev_obs.ids[i]] = Eigen::Vector2d(prev_obs.pts_raw(i, 0),
                                                    prev_obs.pts_raw(i, 1));
    std::vector<double> disp;
    disp.reserve(cur_obs.ids.size());
    for (size_t i = 0; i < cur_obs.ids.size(); ++i) {
        auto it = prev_map.find(cur_obs.ids[i]);
        if (it == prev_map.end()) continue;
        double dx = cur_obs.pts_raw(i, 0) - it->second(0);
        double dy = cur_obs.pts_raw(i, 1) - it->second(1);
        disp.push_back(std::sqrt(dx * dx + dy * dy));
    }
    double median_disp = 1e9;
    if (!disp.empty()) {
        std::nth_element(disp.begin(), disp.begin() + disp.size() / 2, disp.end());
        median_disp = disp[disp.size() / 2];
    }

    return a_var < acc_var_th && w_var < gyr_var_th && median_disp < disp_th;
}

} // namespace gvio
