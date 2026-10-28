#!/usr/bin/env bash
# build.sh: build voxl-gtsam-vio.
#   ./build.sh native              desktop: engine + gvio-engine-replay, plus the MPA server and gvio-replay
#                                  when libmodal-pipe is found in MPA_ROOT (default ~/voxl_x86/root, made by
#                                  scripts/build_mpa_x86.sh)
#   ./build.sh qrb5165|qrb5165-2   VOXL2, inside voxl-cross: scripts/build_voxl2.sh (UNTESTED)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  native)
    MPA_ROOT="${MPA_ROOT:-$HOME/voxl_x86/root}"
    mkdir -p "$HERE/build" && cd "$HERE/build"
    cmake -DMPA_ROOT="$MPA_ROOT" ..
    make -j"$(nproc)"
    ;;
  qrb5165|qrb5165-2)
    "$HERE/scripts/build_voxl2.sh" "$@"
    ;;
  *)
    echo "usage: ./build.sh native | qrb5165 | qrb5165-2" >&2; exit 2 ;;
esac
