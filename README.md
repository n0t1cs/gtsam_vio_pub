# Monocular Visual-Inertial Odometry on a GTSAM Fixed-Lag Factor Graph

Reference implementation for the manuscript

> *A GTSAM-Based Monocular Visual-Inertial Odometry for Indoor UAVs: Robust
> Initialization and Single-Configuration Validation on EuRoC*
> MDPI *Drones*, manuscript `drones-4489122`.

A tightly coupled monocular point-feature VIO built on a GTSAM fixed-lag smoother,
with inverse-depth landmarks, on-manifold IMU preintegration, layered outlier
rejection, and a bootstrap hardened against the degenerate near-zero-baseline
window that opens an indoor flight.

This repository contains the estimator and the configurations used to produce the
reported results. The research repository it is extracted from also holds
in-development components that are not part of this paper, and those are not
included here.

## Layout

```
cpp_gtsam_vio/
  main.cpp                  offline driver: streams a EuRoC sequence in timestamp order
  vio_backend.{cpp,h}       keyframe gating, IMU preintegration, landmark bookkeeping,
                            observation rejection and diagnostics, owns the smoother
  factors/                  inverse-depth reprojection factor over body poses
  frontend/                 KLT / Shi-Tomasi tracking with IMU-aided flow prediction
  init/                     the bootstrap: buffer -> SfM -> BA -> visual-inertial align
  loop/                     DBoW2 place recognition + 4-DoF pose graph, running parallel
                            to the VIO and out of scope for the reported results
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

## The two back-ends

The same factor graph is solved by either of two fixed-lag smoothers, selected by the
`backend_mode` field. Both carry a lag of 3.0 s, the same factors and the same gates,
so the solver is isolated as the only variable.

| `backend_mode` | Solver | Role |
|---|---|---|
| `batch` | `BatchFixedLagSmoother` | Levenberg-Marquardt, relinearizes the whole window each step. The paper's reference arm and the shipped default. |
| `incremental` | `IncrementalFixedLagSmoother` | iSAM2, Bayes tree with fluid relinearization. |

**Every result in the paper is reported on both.** The shipped configurations select
`batch`. To reproduce an incremental column, change that one field and nothing else.

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

### Reproducing the accuracy table

`A0` is the full system and the configuration behind the reported per-sequence
accuracy. Run it on all eleven sequences, Machine Hall file for `MH_*` and Vicon Room
file for `V1_*` and `V2_*`:

```bash
for s in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
         V1_01_easy V1_02_medium V1_03_difficult \
         V2_01_easy V2_02_medium V2_03_difficult; do
  case $s in MH_*) env=mh ;; *) env=vicon ;; esac
  ./build/vio_pipeline /path/to/EuRoC/$s config/arms/${env}_A0.yaml
done
```

### Reproducing the component-wise study

`config/arms/` holds the seven arms, one file per environment. Each differs from the
full system (`A0`) in exactly one setting, and every gate is written so that disabling
it is bit-identical to removing it from the code path:

| Arm | Mechanism disabled |
|-----|--------------------|
| A0  | none, the full system |
| A1  | pre-BA conditioning gate |
| A2  | velocity-consistency gate |
| A3  | hard 20 px reprojection reject |
| A4  | soft Huber weighting |
| A5  | continuous-window initialization |
| A6  | inverse-depth parametrization (to world-frame points) |

The published study is these seven arms over eleven sequences on **both** back-ends.
The incremental half is the same files with one field changed:

```bash
for f in config/arms/*.yaml; do
  sed 's/backend_mode: batch/backend_mode: incremental/' "$f" > "${f%.yaml}_inc.yaml"
done
```

Comparisons are **paired per sequence**, and each back-end is paired against its own
`A0` baseline rather than against the other. A median of absolute ATEs across sequences
is not a paired statistic and can reverse the sign of a verdict.

### Reproducing the consistency (NEES) table

The arm configurations already set `runtime.dump_covariance`, so each run writes
`vio_cov.csv` holding the per-keyframe pose and its 6x6 marginal covariance. NEES is
the body-frame residual `e = Log(T_est^-1 T_gt)` weighted by that covariance, which is
invariant to the global alignment gauge.

The statistic reported in the paper is a **median** over keyframes, so it is compared
against the median of the corresponding chi-square distribution and not against the
dimension: **5.35** for the 6-DoF pose and **2.37** for each 3-DoF block. Comparing
against 6 and 3 would make the estimator appear better calibrated than it is.

Enabling the covariance dump costs about 24% of the per-keyframe budget, so timing
figures should be taken from runs with it switched off.

### Notes on reproducibility

Runs are deterministic: the task scheduler is pinned to a single worker and every
RANSAC estimator is seeded, so a given configuration and sequence produce identical
output on every execution. **Run one process at a time.** Concurrent runs share the
results tree and make the timing and memory figures meaningless.

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

Please cite the *Drones* manuscript above. Contact: Gabriel André Araújo, INESC TEC.
