#!/usr/bin/env bash
# build.sh: build voxl-gtsam-vio.
#   ./build.sh native              engine + gvio-engine-replay + gvio-publish-check, plus the MPA server
#                                  where libmodal-pipe is installed (the VOXL2) or found in MPA_ROOT
#   ./build.sh qrb5165|qrb5165-2   VOXL2, inside voxl-cross: scripts/build_voxl2.sh (UNTESTED)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  native)
    mkdir -p "$HERE/build" && cd "$HERE/build"
    cmake ${MPA_ROOT:+-DMPA_ROOT="$MPA_ROOT"} ..
    make -j"$(nproc)"
    ;;
  qrb5165|qrb5165-2)
    "$HERE/scripts/build_voxl2.sh" "$@"
    ;;
  *)
    echo "usage: ./build.sh native | qrb5165 | qrb5165-2" >&2; exit 2 ;;
esac
