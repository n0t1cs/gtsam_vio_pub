# voxl-gtsam-vio

A monocular visual-inertial odometry on a GTSAM fixed-lag factor graph, running online on a ModalAI
VOXL2. It reads the IMU and a tracking camera straight from the MPA pipes and publishes ModalAI's VIO
interface (`vio_data_t`, `ext_vio_data_t`), so `voxl-vision-hub`, `voxl-inspect-vio`, `voxl-logger`
and the portal read it like the on-board `qvio` and `ov` pipes.

The estimator is the offline one, unchanged: KLT and Shi-Tomasi tracking with IMU-aided flow
prediction, a VINS-Mono-style initialisation, inverse-depth landmarks and iSAM2
(`IncrementalFixedLagSmoother`) over a window of 10 keyframes. It descends from the reference
implementation on branch `paper_repro` of this repository (MDPI *Drones*, manuscript
`drones-4489122`) and has changed since: the window is counted in keyframes instead of 3 s, iSAM2 is
the only backend, landmarks are re-anchored when their anchor keyframe leaves the window, and the
configuration is tuned for a VOXL2 quadrotor. Given the same inputs, the online engine reproduces the
offline estimate byte for byte (Validation).

## Status

| part | state |
|---|---|
| online engine (`src/engine`) | built and validated on an x86 desktop (Validation) |
| published frame | checked against motion capture: no axis flip, 9.7 cm rms after aligning one pose |
| MPA server and pipe replay (`src/server`, `tools/gvio_replay.cpp`) | passes a syntax check against the libmodal-pipe headers, not yet linked or run on real pipes |
| VOXL2 build and package (`scripts/build_voxl2.sh`, `make_package.sh`) | written, untested |
| flight | not done. Bench test first (below) |

## How it works

```
 MPA imu_apps ──► ImuFilter (Butterworth order 4 at 25 Hz, online) ──► IMU fifo ───────────────┐
 MPA tracking_front_misp_norm ──► frame copy ──► pending frames (bounded) ──────────────────────┤
                                                                                                 ▼
  sync thread: waits until the filtered IMU passes a frame's stamp, hands every IMU sample up to it
  to the tracker and the estimator BEFORE the frame, undistorts the fisheye frame to a virtual
  pinhole, queues it
                                                                                                 ▼
  FrontendThread: KLT + Shi-Tomasi + F-RANSAC, IMU-aided prediction
                                                                                                 ▼
  estimator thread: initialisation search with its gates, then per frame IMU preintegration,
  keyframe decision and the iSAM2 update, health checks and resets, per-frame output
                                                                                                 ▼
  MpaBridge: estimator world -> VIO_FRAME_BODY_ALIGNED (FRD, origin at the IMU at initialisation),
  vio_data_t on `gvio`, ext_vio_data_t on `gvio_extended`, control commands reset_vio_soft/hard
```

Design notes:

- **IMU.** The 1 kHz `imu_apps` stream carries the rotor tone (about 85 Hz) and is low-passed online
  (Butterworth order 4 at 25 Hz, gain 6.9e-3 at 85 Hz). Each filtered sample is stamped with its raw
  stamp minus the filter's group delay at 1 Hz (16.6 ms at 1024 Hz).
- **Camera.** Every fisheye frame is remapped to a pinhole of the same size, from the vehicle's own
  calibration files (`/data/modalai/opencv_<camera>_intrinsics.yml`, `/etc/modalai/extrinsics.conf`):
  new camera matrix from `cv::fisheye::estimateNewCameraMatrixForUndistortRectify` at balance 0,
  `INTER_LINEAR`, `BORDER_CONSTANT`.
- **Online additions** the offline driver never needed: waiting for IMU coverage, dropping the oldest
  frame under backlog, health checks (velocity above 20 m/s, no features for 3 s, an IMU gap inside a
  frame interval, a camera gap above 1 s, an estimator exception), automatic soft reset, manual
  resets on the control pipe, a watchdog that publishes FAILED when the inputs stop.
- **Covariance** is the keyframe's marginal, rotated into the published frame and repeated on the
  frames between keyframes. Velocity covariance is not provided (NaN).
- **Not online:** loop closure (the output is odometry), a second camera.

## Requirements

GTSAM 4.2a9 (other versions build but give a different estimate), OpenCV 4 with the `video` module,
yaml-cpp, Boost, TBB on the desktop. The MPA parts need ModalAI's libmodal-pipe and libmodal-json
(LGPL-3.0, linked dynamically).

## Desktop

```bash
./build.sh native        # engine + gvio-engine-replay (+ the MPA parts if libmodal-pipe is found)

# the engine on a EuRoC-format dataset, in lockstep (no MPA)
build/gvio-engine-replay --dataset DATASET --config CONFIG --out /tmp/run1
# a raw voxl-logger log through the deployment path (undistortion and IMU filter online), real time
build/gvio-engine-replay --voxl-log VOXL_LOG --config config/estimator_mantis2.yaml --out /tmp/run2 --rate 1
# ATE against a EuRoC-format ground truth (state_groundtruth_estimate0)
python3 scripts/eval_online.py /tmp/run2 --gt DATASET/mav0
```

A `VOXL_LOG` is a voxl-logger directory that also holds the vehicle's `data/modalai` and
`etc/modalai` calibration files. For the MPA parts on a desktop, `scripts/build_mpa_x86.sh` builds
libmodal-pipe and libmodal-json from ModalAI's public sources (pinned tags) into `~/voxl_x86`. Then two
terminals, as in the header of `config/test_mpa_voxl_log.yaml`. On a desktop `/run/mpa` is
root-owned, so the test configs put every pipe under `/tmp/gvio_mpa/` (libmodal-pipe uses any
location that starts with `/` as is).

## Building for the VOXL2 (untested)

Inside ModalAI's `voxl-cross` docker:

```bash
./build.sh qrb5165        # system image 1.x (Ubuntu 18.04), qrb5165-2 for 2.x (Ubuntu 20.04)
./scripts/make_package.sh qrb5165
```

`build_voxl2.sh` installs libmodal-pipe and libmodal-json from ModalAI's package server, then builds
Boost 1.71, yaml-cpp 0.8.0, OpenCV 4.6.0 (with the `video` module, which the vehicle's OpenCV build
cannot be assumed to carry) and GTSAM 4.2a9, all static, then this project at `-O1 -g`. Expect to
adjust paths on the first run. Known traps: the 18.04 toolchain's gcc 7 has no `<filesystem>` (the
script switches to gcc 8), and GTSAM is built without TBB there (single-threaded, numerically
equivalent to the desktop build, not byte-identical). Building GTSAM and OpenCV on the VOXL2 itself
also works in principle but takes hours and can exhaust memory: limit the parallel jobs.

## On the vehicle

```bash
dpkg -i voxl-gtsam-vio_0.1.0_qrb5165_arm64.deb     # service installed, NOT enabled
voxl-gtsam-vio-server -v                            # foreground first
voxl-inspect-vio gvio                               # the output, ModalAI's tool
systemctl enable --now voxl-gtsam-vio-server        # only after the bench test
```

Logs go to `/data/gvio/<start time>/`: `frames.csv` (every frame, with its latency), `events.log`,
`session<k>/vio_results_est.tum` and `vio_cov.csv` (keyframes, one session per reset), readable by
`scripts/eval_online.py`.

### Bench test with voxl-logger and voxl-replay

Record the two input pipes once, then replay them as often as needed. `voxl-replay` shifts every
channel's stamps by one common offset to the replay clock, so IMU and camera stay consistent.

```bash
voxl-logger -i imu_apps -c tracking_front_misp_norm -t 120          # writes /data/voxl-logger/logNNNN/
voxl-gtsam-vio-server -v                                            # terminal 1
voxl-replay -p /data/voxl-logger/logNNNN/ -i imu_apps -i tracking_front_misp_norm -y   # terminal 2
```

`-y` stops the servers that publish the same pipes (`voxl-imu-server`, `voxl-camera-server`), otherwise
two writers share one pipe. Before any flight, in this order:

1. **Replay on the vehicle.** The output must match a desktop run of the same log within the
   run-to-run noise, and the latency in `frames.csv` must stay below one frame interval. This measures
   the VOXL2's CPU, which the desktop cannot.
2. **Live, hand-held,** next to the on-board VIO (`voxl-cpu-monitor` for the CPU), comparing `gvio`
   with `ov` or `qvio`.
3. **Flight** with `vio_pipe` in `/etc/modalai/voxl-vision-hub.conf` still on the on-board VIO, `gvio`
   only logged.
4. Only then `vio_pipe: gvio`, with a safety pilot.

## Configuration

`/etc/modalai/voxl-gtsam-vio-server.yaml` (commented), and the estimator's own config it points to
(`/usr/share/voxl-gtsam-vio/estimator_mantis2.yaml`, whose header lists the settings that matter).

| key | default | meaning |
|---|---|---|
| `camera_pipe`, `camera_name` | tracking_front_misp_norm, tracking_front | the input stream and its calibration entry |
| `undistort_balance`, `pinhole_override` | 0.0, none | the virtual pinhole. A pin is for parity tests against a fixed dataset only |
| `imu_lowpass` | on, 4, 25 Hz | the estimator's IMU noise was calibrated on the filtered stream |
| `output_pipe(_extended)` | gvio, gvio_extended | point `voxl-vision-hub`'s `vio_pipe` here to fly on it |
| `health.*` | the on-board OpenVINS server's values | automatic reset thresholds |
| `log_dir` | /data/gvio | disk log per start, empty disables it |

## Validation

All on an x86 desktop (`-O1`), on 11 EuRoC sequences and two indoor flights of a VOXL2 quadrotor
with motion capture.

- **Exactness.** In lockstep the online engine reproduces the offline estimator byte for byte on all
  13 sequences (trajectory and covariance), up to a run-to-run floating-point noise of 1e-9 to
  1.5e-7 m that the offline build shows as well.
- **Deployment path.** The raw flight logs through online undistortion and the online IMU filter:
  median ATE over six runs 0.058 m and 0.033 m, against 0.059 m and 0.032 m for the offline estimator
  on the pre-processed datasets. Run by run the two differ by up to 20 % either way, so a single run
  says little.
- **Real time.** Latency from a frame's arrival to its output 38 to 39 ms median, 62 to 66 ms at the
  95th percentile, keyframe update 17 to 19 ms median. Frames were dropped only during the IMU
  filter's 1 s warm-up and after the recorded IMU stream ended. This says nothing about the VOXL2.
- **Published frame.** `build/gvio-publish-check LOG` runs the bridge's conversion on a log. Against
  motion capture: 4.7 deg between the frames (2.7 deg of it is the motion-capture world's tilt),
  positions within 9.7 cm rms over 168 s after aligning one pose, body z down.
- **Not yet:** the MPA transport (server and replay on real pipes), the VOXL2 build, flight.

## Layout

```
estimator/     the estimator: back-end, factors, front-end, initialisation, configuration
src/engine/    transport-free online engine: imu_filter, camera_model, engine, output_log,
               publish_frame, offline_helpers (the offline driver's per-frame helpers)
src/server/    voxl-gtsam-vio-server: server_config, mpa_bridge, main
tools/         gvio_engine_replay.cpp (desktop bench, no MPA), gvio_replay.cpp (a log or a dataset
               published on pipes), gvio_publish_check.cpp (the published frame on a log, no MPA)
config/        voxl-gtsam-vio-server.yaml (vehicle), estimator_mantis2.yaml, test_mpa_*.yaml
               (desktop MPA tests, pipes under /tmp/gvio_mpa)
scripts/       eval_online.py, build_mpa_x86.sh, build_voxl2.sh, make_package.sh
service/       systemd unit
```
