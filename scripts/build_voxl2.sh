#!/usr/bin/env bash
# =============================================================================
# build_voxl2.sh: cross-build voxl-gtsam-vio for a VOXL2 inside ModalAI's voxl-cross docker (>= 4.0).
#
# STATUS: UNTESTED. Written on 2026-10-28 without access to voxl-cross or a vehicle. Every step is the
# documented ModalAI or upstream procedure, but expect to fix paths on the first run. Read README.md,
# "Building for the VOXL2".
#
#   voxl-docker -i voxl-cross
#   ./scripts/build_voxl2.sh qrb5165 [section]      VOXL2 system image 1.x (Ubuntu 18.04)
#   ./scripts/build_voxl2.sh qrb5165-2 [section]    VOXL2 system image 2.x (Ubuntu 20.04)
#   ./scripts/make_package.sh                        -> voxl-gtsam-vio_<version>_arm64.deb
#
# What is built, and why:
#   * libmodal-pipe, libmodal-json: ModalAI's packages from voxl-packages.modalai.com (section default
#     sdk-1.0 style "dev"), exactly as ModalAI's own install_build_deps.sh does. Linked dynamically,
#     the vehicle has them.
#   * OpenCV 4.6.0 (core imgproc imgcodecs calib3d features2d flann video), static. The vehicle's
#     OpenCV cannot be assumed to carry the video module (KLT): voxl-open-vins-server disables its KLT
#     tracker "to avoid opencv video module dependency". 4.6.0 is the desktop version the offline
#     system and this engine were validated with, so the tracker code is the same.
#   * Boost 1.71 (system filesystem serialization timer thread chrono date_time regex program_options),
#     yaml-cpp 0.8.0 and GTSAM 4.2a9 with the desktop build options (cheirality off, bundled Eigen,
#     no march native, Pose3/Rot3 expmap, tangent preintegration), all static. GTSAM is built WITHOUT
#     TBB here: single-threaded elimination, which is what the pinned desktop build runs anyway, but
#     not byte-identical to it.
#   * this project at -O1 -g (the validated level), static except libmodal-* and the C/C++ runtime.
# A compiler older than gcc 8 has no <filesystem>: on the 18.04 toolchain the script installs and uses
# g++-8-aarch64-linux-gnu (Ubuntu 18.04's libstdc++6 is gcc 8's, so the binary runs on the vehicle).
# =============================================================================
set -euo pipefail
PLATFORM="${1:-}"; SECTION="${2:-dev}"
case "$PLATFORM" in
  qrb5165)   TOOLCHAIN=/opt/cross_toolchain/qrb5165_ubun1_18.04_aarch64.toolchain.cmake ;;
  qrb5165-2) TOOLCHAIN=/opt/cross_toolchain/qrb5165_ubun2_20.04_aarch64.toolchain.cmake ;;
  *) echo "usage: $0 qrb5165|qrb5165-2 [section]" >&2; exit 2 ;;
esac
[[ -f "$TOOLCHAIN" ]] || { echo "ERROR: $TOOLCHAIN missing: run inside voxl-cross" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GVIO_WORK:-$HERE/build_voxl2_deps}"          # sources and builds of the dependencies
STAGE="$WORK/stage"                                   # their install prefix (static libs, headers)
JOBS="${JOBS:-$(nproc)}"
mkdir -p "$WORK/src" "$STAGE"

# ---- compiler: the toolchain's, unless it is older than gcc 8
CXX_T="$(grep -oE 'CMAKE_CXX_COMPILER[[:space:]]+[^ )]+' "$TOOLCHAIN" | awk '{print $2}' | tr -d '"' | head -1)"
CC_T="$(grep -oE 'CMAKE_C_COMPILER[[:space:]]+[^ )]+' "$TOOLCHAIN" | awk '{print $2}' | tr -d '"' | head -1)"
VER="$($CXX_T -dumpversion | cut -d. -f1)"
EXTRA_CMAKE=()
if [[ "$VER" -lt 8 ]]; then
  sudo apt-get install -y g++-8-aarch64-linux-gnu gcc-8-aarch64-linux-gnu
  CXX_T=aarch64-linux-gnu-g++-8; CC_T=aarch64-linux-gnu-gcc-8
  EXTRA_CMAKE=(-DCMAKE_CXX_COMPILER=$CXX_T -DCMAKE_C_COMPILER=$CC_T)
fi
echo "compiler: $CXX_T ($($CXX_T -dumpversion))"
CM=(cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" "${EXTRA_CMAKE[@]}" -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$STAGE" -DCMAKE_PREFIX_PATH="$STAGE" -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

# ---- ModalAI packages (libmodal-pipe, libmodal-json), ModalAI's own procedure
echo "deb [trusted=yes] http://voxl-packages.modalai.com/ ./dists/$PLATFORM/$SECTION/binary-arm64/" \
  | sudo tee /etc/apt/sources.list.d/modalai.list >/dev/null
sudo apt-get update -o Dir::Etc::sourcelist="sources.list.d/modalai.list" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install -y libmodal-pipe libmodal-json

fetch() {   # fetch <dir> <url> : a tarball, once
  [[ -d "$WORK/src/$1" ]] && return 0
  (cd "$WORK/src" && curl -fL "$2" -o "$1.tar.gz" && mkdir -p "$1" && tar -xzf "$1.tar.gz" -C "$1" --strip-components=1)
}

# ---- Boost 1.71, static
if [[ ! -f "$STAGE/lib/libboost_serialization.a" ]]; then
  fetch boost https://archives.boost.io/release/1.71.0/source/boost_1_71_0.tar.gz
  (cd "$WORK/src/boost" \
     && ./bootstrap.sh --with-libraries=system,filesystem,serialization,timer,thread,chrono,date_time,regex,program_options \
     && echo "using gcc : aarch64 : $CXX_T ;" > user-config.jam \
     && ./b2 -j"$JOBS" --user-config=user-config.jam toolset=gcc-aarch64 target-os=linux architecture=arm \
             address-model=64 link=static runtime-link=shared threading=multi variant=release cxxflags=-fPIC \
             --prefix="$STAGE" install)
fi

# ---- yaml-cpp 0.8.0, static
if [[ ! -f "$STAGE/lib/libyaml-cpp.a" ]]; then
  fetch yaml-cpp https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  mkdir -p "$WORK/src/yaml-cpp/b" && (cd "$WORK/src/yaml-cpp/b" && "${CM[@]}" -DYAML_CPP_BUILD_TESTS=OFF \
     -DYAML_CPP_BUILD_TOOLS=OFF .. && make -j"$JOBS" install)
fi

# ---- OpenCV 4.6.0, the modules the tracker needs, static
if [[ ! -f "$STAGE/lib/libopencv_video.a" ]]; then
  fetch opencv https://github.com/opencv/opencv/archive/refs/tags/4.6.0.tar.gz
  mkdir -p "$WORK/src/opencv/b" && (cd "$WORK/src/opencv/b" && "${CM[@]}" \
     -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,video -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
     -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_TBB=OFF \
     -DWITH_OPENMP=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
     -DWITH_TIFF=OFF -DWITH_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_ZLIB=ON \
     -DOPENCV_GENERATE_PKGCONFIG=OFF .. && make -j"$JOBS" install)
fi

# ---- GTSAM 4.2a9, the desktop options, static, no TBB
if [[ ! -f "$STAGE/lib/libgtsam.a" ]]; then
  [[ -d "$WORK/src/gtsam" ]] || git clone -q --depth 1 --branch 4.2a9 https://github.com/borglab/gtsam.git "$WORK/src/gtsam"
  mkdir -p "$WORK/src/gtsam/b" && (cd "$WORK/src/gtsam/b" && "${CM[@]}" -DBOOST_ROOT="$STAGE" -DBoost_USE_STATIC_LIBS=ON \
     -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
     -DGTSAM_THROW_CHEIRALITY_EXCEPTION=OFF -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON \
     -DGTSAM_TANGENT_PREINTEGRATION=ON -DGTSAM_USE_QUATERNIONS=OFF -DGTSAM_BUILD_UNSTABLE=ON \
     -DGTSAM_ALLOW_DEPRECATED_SINCE_V42=ON -DGTSAM_SUPPORT_NESTED_DISSECTION=ON -DGTSAM_BUILD_TESTS=OFF \
     -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_DOCS=OFF .. && make -j"$JOBS" install)
fi

# ---- this project
mkdir -p "$HERE/build_$PLATFORM" && cd "$HERE/build_$PLATFORM"
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" "${EXTRA_CMAKE[@]}" -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_PREFIX_PATH="$STAGE" -DOpenCV_DIR="$STAGE/lib/cmake/opencv4" -DGTSAM_PREFIX="$STAGE" \
      -DGVIO_WITH_TBB=OFF -DGVIO_WITH_MPA=ON -DGVIO_STATIC_DEPS=ON -DBOOST_LIB_DIR="$STAGE/lib" ..
make -j"$JOBS"
echo "built: $HERE/build_$PLATFORM ; package with scripts/make_package.sh $PLATFORM"
