#include "imu_filter.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

namespace gvio {

using cd = std::complex<double>;

std::vector<Biquad> butter_lowpass_sos(int order, double cutoff_hz, double fs_hz) {
    if (order < 1 || cutoff_hz <= 0.0 || cutoff_hz >= 0.5 * fs_hz)
        throw std::invalid_argument("butter_lowpass_sos: need order >= 1 and 0 < cutoff < fs/2");
    // buttap: p = -exp(j pi m / (2N)), m = -N+1, -N+3, ..., N-1, no zeros, gain 1.
    std::vector<cd> p;
    for (int m = -order + 1; m <= order - 1; m += 2)
        p.push_back(-std::exp(cd(0.0, M_PI * m / (2.0 * order))));
    // Prewarped analog cutoff, lp2lp_zpk: p -> wo p, k -> wo^N.
    const double wo = 2.0 * fs_hz * std::tan(M_PI * cutoff_hz / fs_hz);
    double k = std::pow(wo, order);
    for (auto& pi : p) pi *= wo;
    // bilinear_zpk: p -> (2fs + p) / (2fs - p), N zeros at -1, k -> k * real(1 / prod(2fs - p)).
    const double fs2 = 2.0 * fs_hz;
    cd prod(1.0, 0.0);
    for (auto& pi : p) { prod *= (fs2 - pi); pi = (fs2 + pi) / (fs2 - pi); }
    k *= std::real(cd(1.0, 0.0) / prod);

    // zpk2sos, pairing 'nearest': the pole closest to the unit circle goes to the LAST section, its
    // conjugate with it. Every section gets the zero pair (-1, -1), b = [1, 2, 1].
    std::vector<cd> upper;                                // one pole per conjugate pair, plus reals
    std::vector<double> reals;
    for (const auto& pi : p) {
        if (std::abs(pi.imag()) < 1e-12) reals.push_back(pi.real());
        else if (pi.imag() > 0) upper.push_back(pi);
    }
    std::sort(upper.begin(), upper.end(), [](const cd& a, const cd& b) { return std::abs(a) < std::abs(b); });
    std::vector<Biquad> sos;
    if (!reals.empty()) {                                 // odd order: one first-order section first
        sos.push_back({1.0, 1.0, 0.0, -reals.front(), 0.0});
    }
    for (const auto& pi : upper)
        sos.push_back({1.0, 2.0, 1.0, -2.0 * pi.real(), std::norm(pi)});
    sos.front().b0 *= k; sos.front().b1 *= k; sos.front().b2 *= k;
    return sos;
}

static cd sos_response(const std::vector<Biquad>& sos, double w) {   // w in rad per sample
    const cd z1 = std::exp(cd(0.0, -w)), z2 = z1 * z1;
    cd h(1.0, 0.0);
    for (const auto& s : sos) h *= (s.b0 + s.b1 * z1 + s.b2 * z2) / (1.0 + s.a1 * z1 + s.a2 * z2);
    return h;
}

double sos_group_delay_s(const std::vector<Biquad>& sos, double fs_hz, double f_hz) {
    // -d(phase)/d(w) by a central difference. Each section's phase is unwrapped separately, so a
    // jump of 2 pi in the product cannot leak in.
    const double w = 2.0 * M_PI * f_hz / fs_hz, dw = 1e-6;
    double gd = 0.0;
    for (const auto& s : sos) {
        const std::vector<Biquad> one{s};
        double d = std::arg(sos_response(one, w + dw)) - std::arg(sos_response(one, w - dw));
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        gd += -d / (2.0 * dw);
    }
    return gd / fs_hz;
}

double sos_gain(const std::vector<Biquad>& sos, double fs_hz, double f_hz) {
    return std::abs(sos_response(sos, 2.0 * M_PI * f_hz / fs_hz));
}

ImuFilter::ImuFilter(const Params& p) : p_(p) {
    if (p_.enable && p_.rate_hz > 0.0) design(p_.rate_hz);
}

void ImuFilter::design(double fs) {
    fs_ = fs;
    period_s_ = 1.0 / fs;
    sos_ = butter_lowpass_sos(p_.order, p_.cutoff_hz, fs);
    tau_s_ = sos_group_delay_s(sos_, fs, p_.tau_eval_hz);
    tau_ns_ = static_cast<int64_t>(std::llround(tau_s_ * 1e9));
    z_.assign(6 * sos_.size() * 2, 0.0);
    designed_ = true;
    std::cout << "[imu-lpf] Butterworth order " << p_.order << ", cutoff " << p_.cutoff_hz
              << " Hz at " << fs << " Hz: group delay " << 1e3 * tau_s_ << " ms at "
              << p_.tau_eval_hz << " Hz, gain at 85 Hz " << sos_gain(sos_, fs, 85.0) << "\n";
}

// Steady state of every section for a constant input x0 (scipy sosfilt_zi(sos) * x0): with input u
// and section DC gain G, y = G u, z2 = b2 u - a2 y, z1 = b1 u - a1 y + z2, and the next section's
// input is y.
void ImuFilter::reset_state(const double x0[6]) {
    const size_t ns = sos_.size();
    for (int c = 0; c < 6; ++c) {
        double u = x0[c];
        for (size_t s = 0; s < ns; ++s) {
            const Biquad& q = sos_[s];
            const double G = (q.b0 + q.b1 + q.b2) / (1.0 + q.a1 + q.a2);
            const double y = G * u;
            const double z2 = q.b2 * u - q.a2 * y;
            const double z1 = q.b1 * u - q.a1 * y + z2;
            z_[(c * ns + s) * 2 + 0] = z1;
            z_[(c * ns + s) * 2 + 1] = z2;
            u = y;
        }
    }
    state_valid_ = true;
}

void ImuFilter::filter_one(const vio::ImuSample& raw, std::vector<vio::ImuSample>& out) {
    double x[6] = {raw.w.x(), raw.w.y(), raw.w.z(), raw.a.x(), raw.a.y(), raw.a.z()};
    if (!state_valid_) reset_state(x);
    const size_t ns = sos_.size();
    for (int c = 0; c < 6; ++c) {
        double u = x[c];
        for (size_t s = 0; s < ns; ++s) {
            const Biquad& q = sos_[s];
            double& z1 = z_[(c * ns + s) * 2 + 0];
            double& z2 = z_[(c * ns + s) * 2 + 1];
            const double y = q.b0 * u + z1;              // direct form II transposed
            z1 = q.b1 * u - q.a1 * y + z2;
            z2 = q.b2 * u - q.a2 * y;
            u = y;
        }
        x[c] = u;
    }
    vio::ImuSample f;
    f.t_ns = raw.t_ns - tau_ns_;
    f.w = Eigen::Vector3d(x[0], x[1], x[2]);
    f.a = Eigen::Vector3d(x[3], x[4], x[5]);
    out.push_back(f);
}

bool ImuFilter::push(const vio::ImuSample& raw, std::vector<vio::ImuSample>& out) {
    if (last_t_ns_ >= 0 && raw.t_ns <= last_t_ns_) { ++n_rejected_; return false; }
    const int64_t prev_t = last_t_ns_;
    last_t_ns_ = raw.t_ns;
    if (!p_.enable) { out.push_back(raw); return true; }

    if (!designed_) {
        warm_.push_back(raw);
        if ((int)warm_.size() < std::max(16, p_.warmup_samples)) return true;
        // Period = least-squares slope of stamp against sample index over the warm-up (the offline
        // filter took it from the same fit over the whole flight).
        const double n = (double)warm_.size();
        double si = 0, st = 0, sii = 0, sit = 0;
        const int64_t t0 = warm_.front().t_ns;
        for (size_t i = 0; i < warm_.size(); ++i) {
            const double t = (warm_[i].t_ns - t0) * 1e-9;
            si += i; st += t; sii += (double)i * i; sit += i * t;
        }
        const double slope = (n * sit - si * st) / (n * sii - si * si);
        design(1.0 / slope);
        for (const auto& s : warm_) filter_one(s, out);   // backlog, from the first sample's steady state
        warm_.clear();
        warm_.shrink_to_fit();
        return true;
    }
    if (prev_t >= 0 && (raw.t_ns - prev_t) * 1e-9 > p_.gap_reset_periods * period_s_) {
        ++n_gaps_;
        state_valid_ = false;                            // restart from this sample's steady state
        std::cerr << "[imu-lpf] gap of " << (raw.t_ns - prev_t) * 1e-6
                  << " ms, filter restarted (" << n_gaps_ << " so far)\n";
    }
    filter_one(raw, out);
    return true;
}

} // namespace gvio
