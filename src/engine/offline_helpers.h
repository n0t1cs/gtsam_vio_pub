#pragma once
// Three per-frame helpers of the offline driver (cpp_gtsam_vio/main.cpp), copied VERBATIM so the
// online engine feeds the estimator exactly as the measured system does. main.cpp keeps them as
// file-static functions, so they cannot be linked from there. If main.cpp changes them, change
// these copies too (scripts/check_helpers.sh compares the bodies).
#include "vio_backend.h"
#include "vio_types.h"

#include <cstdint>
#include <vector>

namespace gvio {

// IMU samples with t0 <= t <= t1 into the backend preintegration, each consecutive pair as one
// step of the later sample (main.cpp integrate_imu_batch).
void integrate_imu_batch(vio::VioBackend& backend, const vio::EurocData& data,
                         double t0_s, double t1_s, int64_t t0_ns_base);

// Accelerometer variance over the init window (main.cpp check_imu_excitation).
bool check_imu_excitation(const std::vector<vio::KeyframeEntry>& frames,
                          const vio::EurocData& data, double min_acc_var);

// IMU quiet AND tracked features barely moving (main.cpp detect_stationary).
bool detect_stationary(const vio::EurocData& data,
                       double t0_s, double t1_s, int64_t t0_ns_base,
                       const vio::Observation& prev_obs,
                       const vio::Observation& cur_obs,
                       double acc_var_th, double gyr_var_th, double disp_th);

} // namespace gvio
