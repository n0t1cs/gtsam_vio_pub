#!/usr/bin/env bash
# =============================================================================
# build_voxl2_native.sh: build voxl-gtsam-vio ON a VOXL2 (system image 1.x, Ubuntu 18.04), as root.
#
#   apt-get install -y cmake build-essential gcc-8 g++-8 libboost-serialization-dev \
#     libboost-system-dev libboost-filesystem-dev libboost-thread-dev libboost-program-options-dev \
#     libboost-date-time-dev libboost-timer-dev libboost-chrono-dev libboost-regex-dev
#   bash scripts/build_voxl2_native.sh
#
# Output goes to the terminal. Re-running it skips the finished stages and resumes an interrupted one.
# The dependency stages were run on a VOXL2 on 2026-10-28 (8 cores, 7.5 GB): about 2 h at 3 jobs.
# What that run taught, built in here:
#   * gcc 8: gcc 7 has no <filesystem>. Ubuntu 18.04's libstdc++6 is gcc 8's, so the binaries run.
#   * 3 jobs (J=...): the board reset twice at 4 jobs without a fan (cause not identified), and
#     finished at 3 jobs with a fan and bench power.
#   * TMPDIR on /data: gcc writes its temporary files to /tmp, on the root filesystem, which can be
#     nearly full on a vehicle.
#   * the 64-bit libmodal-pipe and libmodal-json are in /usr/lib64, /usr/lib holds 32-bit ones that
#     CMake would find first.
# Where things go:
#   /data/gvio_deps/stage  yaml-cpp 0.8.0 and OpenCV 4.6.0 (with the video module, for KLT), static
#   /usr/local             GTSAM 4.2a9, shared, the desktop options without TBB. Uninstall with
#                          xargs -d '\n' rm -f -- < /data/gvio_deps/gtsam/b/install_manifest.txt
#   build/                 this project at -O1 -g
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D=/data/gvio_deps; S="$D/stage"; J="${J:-3}"
export CC=gcc-8 CXX=g++-8 TMPDIR=/data/tmp
MPA_PIPE=/usr/lib64/libmodal_pipe.so; MPA_JSON=/usr/lib64/libmodal_json.so

[[ $(id -u) -eq 0 ]] || { echo "ERROR: run as root (GTSAM installs into /usr/local)" >&2; exit 1; }
for t in gcc-8 g++-8 cmake make curl git; do
  command -v "$t" >/dev/null || { echo "ERROR: $t missing: install the packages in this script's header" >&2; exit 1; }
done
[[ -f /usr/include/boost/version.hpp ]] || { echo "ERROR: Boost headers missing: install the packages in this script's header" >&2; exit 1; }
for f in "$MPA_PIPE" "$MPA_JSON" /usr/include/modal_pipe.h; do
  [[ -f "$f" ]] || { echo "ERROR: $f missing: is this a VOXL2 with voxl-suite installed?" >&2; exit 1; }
done
mkdir -p "$D" "$S" "$TMPDIR"
step() { echo; echo "==== $* ($(date +%T))"; }

fetch() {   # fetch <dir> <url>: download and unpack once, the directory appears only when complete
  [[ -d "$D/$1" ]] && return 0
  rm -rf "$D/$1.part" && mkdir -p "$D/$1.part"
  curl -fL "$2" | tar xz -C "$D/$1.part" --strip-components=1
  mv "$D/$1.part" "$D/$1"
}

# ---- yaml-cpp 0.8.0, static
if [[ ! -f "$S/lib/libyaml-cpp.a" ]]; then
  step "yaml-cpp 0.8.0"
  fetch yaml-cpp https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  mkdir -p "$D/yaml-cpp/b" && cd "$D/yaml-cpp/b"
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$S" -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF ..
  make -j"$J" install
fi
echo "YAML DONE"

# ---- OpenCV 4.6.0, the modules the tracker needs, static
if [[ ! -f "$S/lib/libopencv_video.a" ]]; then
  step "OpenCV 4.6.0"
  fetch opencv https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz
  mkdir -p "$D/opencv/b" && cd "$D/opencv/b"
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$S" -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,video \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
        -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
        -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF \
        -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_ZLIB=ON ..
  make -j"$J" install
fi
echo "OPENCV DONE"

# ---- GTSAM 4.2a9, the desktop options without TBB, shared, /usr/local
if [[ ! -f /usr/local/include/gtsam/config.h || ! -f /usr/local/lib/libgtsam.so ]]; then
  step "GTSAM 4.2a9"
  if [[ ! -d "$D/gtsam" ]]; then
    rm -rf "$D/gtsam.part"
    git clone --depth 1 --branch 4.2a9 https://github.com/borglab/gtsam.git "$D/gtsam.part"
    mv "$D/gtsam.part" "$D/gtsam"
  fi
  mkdir -p "$D/gtsam/b" && cd "$D/gtsam/b"
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
        -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON \
        -DGTSAM_TANGENT_PREINTEGRATION=ON -DGTSAM_USE_QUATERNIONS=OFF -DGTSAM_BUILD_UNSTABLE=ON \
        -DGTSAM_ALLOW_DEPRECATED_SINCE_V42=ON -DGTSAM_SUPPORT_NESTED_DISSECTION=ON \
        -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_PYTHON=OFF ..
  make -j"$J"
  make install
  ldconfig
fi
echo "GTSAM DONE"

# ---- this project
step "voxl-gtsam-vio"
mkdir -p "$HERE/build" && cd "$HERE/build"
cmake -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DGVIO_WITH_TBB=OFF \
      -DCMAKE_PREFIX_PATH="$S" -DOpenCV_DIR="$S/lib/cmake/opencv4" \
      -DMPA_PIPE_LIB="$MPA_PIPE" -DMPA_JSON_LIB="$MPA_JSON" ..
make -j"$J"
ls -l gvio-engine-replay gvio-publish-check voxl-gtsam-vio-server gvio-replay
if ldd voxl-gtsam-vio-server | grep 'not found'; then
  echo "ERROR: voxl-gtsam-vio-server has unresolved libraries (above)" >&2; exit 1
fi
ldd voxl-gtsam-vio-server | grep -E 'gtsam|modal'
echo "BUILD DONE: $HERE/build"
