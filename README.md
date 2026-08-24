# Monocular Visual-Inertial Odometry on a GTSAM Fixed-Lag Factor Graph

Reference implementation for the manuscript

> *A GTSAM-Based Monocular Visual-Inertial Odometry for Indoor UAVs: Robust
> Initialization and Single-Configuration Validation on EuRoC*
> — submitted to MDPI *Drones* (manuscript `drones-4489122`).

A tightly coupled monocular point-feature VIO built on a GTSAM fixed-lag smoother,
with inverse-depth landmarks, on-manifold IMU preintegration, layered outlier
rejection, and a bootstrap hardened against the degenerate near-zero-baseline
window that opens an indoor flight.

This repository contains the estimator and the configurations used to produce the
reported results. The research repository it is extracted from also holds
in-development components that are not part of this paper; those are not included
here.

## Layout

```
cpp_gtsam_vio/
  main.cpp                  offline driver: streams a EuRoC sequence in timestamp order
  vio_backend.{cpp,h}       keyframe gating, IMU preintegration, landmark bookkeeping,
                            observation rejection, diagnostics; owns the smoother
  factors/                  inverse-depth reprojection factor over body poses
  frontend/                 KLT / Shi-Tomasi tracking with IMU-aided flow prediction
  init/                     the bootstrap: buffer -> SfM -> BA -> visual-inertial align
  loop/                     DBoW2 place recognition + 4-DoF pose graph (runs parallel
                            to the VIO; out of scope for the reported results)
  utils/                    configuration structure and dataset loading
  config/
    mh_params.yaml          Machine Hall
    vicon_params.yaml       Vicon Room
    arms/                   the seven configurations of the component-wise study (A0-A6)
```

The two environment files differ in **exactly one scalar**, the inertial integration
covariance (`integration_cov`). Every gate threshold, noise multiplier, window length,
keyframe criterion and initialization parameter is identical between them, and no
parameter is tuned per sequence.

## Building

Requires GTSAM 4.2a9, OpenCV 4, yaml-cpp, TBB and Boost. From `cpp_gtsam_vio/`:

```bash
make info      # print detected paths -- run this first if anything fails to build
make           # -> build/vio_pipeline
```

There is no header dependency tracking: run `make clean && make` after editing a `.h`.

## Running

Always pass a configuration explicitly.

```bash
./build/vio_pipeline /path/to/EuRoC/MH_01_easy config/mh_params.yaml
```

The binary loads `support_files/` by a **relative** path, so it must be run with
`cpp_gtsam_vio/` as the working directory.

Each run writes `vio_results.csv`, `vio_results_{est,gt}.tum` (and `vio_cov.csv` when
`runtime.dump_covariance` is set) plus a `run.log` containing the ATE summary.

### Reproducing the component-wise study

`config/arms/` holds the seven arms of Section 3.2, one file per environment. Each
differs from the full system (`A0`) in exactly one setting, and every gate is written
so that disabling it is bit-identical to removing it from the code path:

| Arm | Mechanism disabled |
|-----|--------------------|
| A0  | none — the full system |
| A1  | pre-BA conditioning gate |
| A2  | velocity-consistency gate |
| A3  | hard 20 px reprojection reject |
| A4  | soft Huber weighting |
| A5  | continuous-window initialization |
| A6  | inverse-depth parametrization (→ world-frame points) |

### Notes on reproducibility

Runs are deterministic: the task scheduler is pinned to a single worker and every
RANSAC estimator is seeded, so a given configuration and sequence produce identical
output on every execution. Run one process at a time — concurrent runs share the
results tree and make timing measurements meaningless.

The results in the paper use a pre-extracted 20 Hz feature stream
(`frontend_source: folder`) so that the comparison against VINS-Mono and PL-VINS
isolates the back-end. Setting `frontend_source: images` runs the bundled KLT
front-end on the raw images instead.

## Data

The EuRoC MAV benchmark is not redistributed here. Obtain it from
<https://projects.asl.ethz.ch/datasets/>. Ground truth is
`state_groundtruth_estimate0` (the IMU/body frame) for every sequence.

Loop closure additionally needs the DBoW2 BRIEF vocabulary `brief_k10L6.bin` in
`support_files/`, which is distributed with VINS-Mono. It is not required for the
odometry results reported in the paper.

## Third-party code

`loop/ThirdParty/` vendors DBoW2, DUtils and DVision, and `support_files/brief_pattern.yml`
is the BRIEF pattern, all as distributed with VINS-Mono. They retain their original
licences.

## Citation

Please cite the *Drones* manuscript above. Contact: Gabriel André, INESC TEC.
