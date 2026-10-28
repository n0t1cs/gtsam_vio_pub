#pragma once
// Online counterpart of the IMU low-pass that produced every mantis_qualisys_<log>_lpf dataset
// (python/extract_mantis_bag.py lowpass_imu, called by lpf_dataset.py): a causal Butterworth on all
// six channels, run in sample index, started from the steady state of the first sample, output
// stamps moved back by the group delay measured at 1 Hz (16.6 ms at order 4, 25 Hz, 1024 Hz).
// The estimator's MANTIS IMU noise was calibrated on that filtered stream, so the online system
// must see the same filter.
//
// One documented difference. Offline the stamps were re-fitted to ONE linear clock over the whole
// flight. Online that fit does not exist, so the raw stamp minus the group delay is used. On log0000
// and log0003 the raw imu_apps stamps sit within 1.8 ms of that linear clock (std 0.3 to 0.4 ms,
// a slow wander, consecutive intervals 0.96 to 1.01 ms), so the two time bases differ by at most
// 1.8 ms and by much less from one frame to the next.
#include "vio_types.h"

#include <cstdint>
#include <vector>

namespace gvio {

// One second-order section, a0 = 1, run as direct form II transposed (scipy.signal.sosfilt).
struct Biquad { double b0, b1, b2, a1, a2; };

// scipy.signal.butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos'), reproduced step by
// step: analog prototype poles, prewarped lowpass scaling, bilinear transform, conjugate poles paired
// into sections (zpk2sos, 'nearest'): the gain sits in the first section and the poles closest to
// the unit circle in the last one.
std::vector<Biquad> butter_lowpass_sos(int order, double cutoff_hz, double fs_hz);

// Group delay [s] and magnitude of the cascade at f_hz.
double sos_group_delay_s(const std::vector<Biquad>& sos, double fs_hz, double f_hz);
double sos_gain(const std::vector<Biquad>& sos, double fs_hz, double f_hz);

class ImuFilter {
public:
    struct Params {
        bool   enable            = true;   // false: samples pass through unchanged (a dataset that
                                           // is already low-passed, the _lpf parity replays)
        int    order             = 4;
        double cutoff_hz         = 25.0;
        double rate_hz           = 0.0;    // 0: measured from the first warmup_samples stamps
        int    warmup_samples    = 1024;
        double gap_reset_periods = 5.0;    // a gap above this many periods restarts the filter
                                           // (offline: the stream was split into segments there)
        double tau_eval_hz       = 1.0;    // where the group delay is measured (offline: 1 Hz)
    };

    explicit ImuFilter(const Params& p);

    // Feed one raw sample. Filtered samples (none during the warm-up, then one per input, or the
    // whole warm-up backlog at once) are appended to `out`. Returns false when the sample was
    // rejected because its stamp does not increase.
    bool push(const vio::ImuSample& raw, std::vector<vio::ImuSample>& out);

    bool    designed() const     { return designed_; }
    double  rate_hz() const      { return fs_; }
    double  tau_s() const        { return tau_s_; }
    int64_t tau_ns() const       { return tau_ns_; }
    int     n_gaps() const       { return n_gaps_; }
    int     n_rejected() const   { return n_rejected_; }
    const std::vector<Biquad>& sos() const { return sos_; }

private:
    void design(double fs);
    void reset_state(const double x0[6]);
    void filter_one(const vio::ImuSample& raw, std::vector<vio::ImuSample>& out);

    Params p_;
    bool designed_ = false;
    double fs_ = 0.0, period_s_ = 0.0, tau_s_ = 0.0;
    int64_t tau_ns_ = 0;
    std::vector<Biquad> sos_;
    std::vector<double> z_;          // per channel, per section: z1, z2
    std::vector<vio::ImuSample> warm_;
    int64_t last_t_ns_ = -1;
    bool state_valid_ = false;
    int n_gaps_ = 0, n_rejected_ = 0;
};

} // namespace gvio
