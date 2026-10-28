#!/usr/bin/env bash
# =============================================================================
# build_mpa_x86.sh: build ModalAI's libmodal-json and libmodal-pipe for a desktop (x86_64), so
# that the server and gvio-replay can be built and tested off the vehicle. Both libraries are
# LGPL-3.0, from the public ModalAI GitLab, pinned to tags. Nothing goes to /usr: sources in
# $PREFIX/src, installed with DESTDIR into $PREFIX/root (root/usr/include, root/usr/lib).
#
#   scripts/build_mpa_x86.sh [PREFIX]      PREFIX defaults to ~/voxl_x86
#
# Then build this project with  ./build.sh native  (it reads MPA_ROOT=$PREFIX/root).
# On a desktop /run/mpa is root-owned: run the pipes under a full path instead, for example
# /tmp/gvio_mpa/, as config/test_*.yaml do (libmodal-pipe skips its /run/mpa/ prefix for any
# location that starts with '/').
# =============================================================================
set -euo pipefail
PREFIX="${1:-$HOME/voxl_x86}"
JSON_TAG="v0.4.8"
PIPE_TAG="v2.9.2"
GL="https://gitlab.com/voxl-public/voxl-sdk/core-libs"
mkdir -p "$PREFIX/src" "$PREFIX/root"

fetch() {   # fetch <repo> <tag>
  if [[ ! -d "$PREFIX/src/$1" ]]; then
    git clone -q --depth 1 --branch "$2" "$GL/$1.git" "$PREFIX/src/$1"
  fi
  local have; have="$(git -C "$PREFIX/src/$1" describe --tags 2>/dev/null || echo none)"
  [[ "$have" == "$2" ]] || { echo "ERROR: $PREFIX/src/$1 is at $have, expected $2" >&2; exit 1; }
}

build() {   # build <repo> <extra cmake args...>
  local repo="$1"; shift
  mkdir -p "$PREFIX/src/$repo/build_x86"
  (cd "$PREFIX/src/$repo/build_x86" \
     && cmake .. -DCMAKE_BUILD_TYPE=Release -DLIB_INSTALL_DIR=/usr/lib \
              -DCMAKE_C_FLAGS="-Wno-error" -DCMAKE_CXX_FLAGS="-Wno-error" "$@" >/dev/null \
     && make -j"$(nproc)" >/dev/null \
     && make install DESTDIR="$PREFIX/root" >/dev/null)
  echo "built $repo"
}

fetch libmodal-json "$JSON_TAG"
fetch libmodal-pipe "$PIPE_TAG"
build libmodal-json -DBUILD_EXAMPLES=OFF
build libmodal-pipe -DEN_ION_BUF=OFF -DBUILD_PYMPA=OFF -DENABLE_MAVLINK_SUPPORT=OFF -DBUILD_EXAMPLES=OFF \
      -DCMAKE_PREFIX_PATH="$PREFIX/root/usr" \
      -DCMAKE_C_FLAGS="-Wno-error -I$PREFIX/root/usr/include" -DCMAKE_CXX_FLAGS="-Wno-error -I$PREFIX/root/usr/include" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L$PREFIX/root/usr/lib"
ls "$PREFIX/root/usr/lib" | grep -E "modal_(json|pipe)"
echo "MPA_ROOT=$PREFIX/root"
