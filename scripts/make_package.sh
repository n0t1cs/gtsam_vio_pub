#!/usr/bin/env bash
# make_package.sh: a .deb of the VOXL2 build (scripts/build_voxl2.sh), ModalAI-style.
# STATUS: UNTESTED (see build_voxl2.sh).
#   ./scripts/make_package.sh qrb5165|qrb5165-2
# Installs: /usr/bin/{voxl-gtsam-vio-server,gvio-replay,gvio-engine-replay},
# /etc/modalai/voxl-gtsam-vio-server.yaml (kept on upgrade), /usr/share/voxl-gtsam-vio/estimator_mantis2.yaml,
# /etc/systemd/system/voxl-gtsam-vio-server.service (NOT enabled: enable it by hand after a bench test).
set -euo pipefail
PLATFORM="${1:-qrb5165}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$HERE/build_$PLATFORM"
VERSION="0.1.0"
PKG="$HERE/pkg_$PLATFORM/voxl-gtsam-vio"
[[ -x "$BUILD/voxl-gtsam-vio-server" ]] || { echo "ERROR: build first (scripts/build_voxl2.sh $PLATFORM)" >&2; exit 1; }
mkdir -p "$PKG/DEBIAN"
(cd "$BUILD" && make install DESTDIR="$PKG" >/dev/null)
cat > "$PKG/DEBIAN/control" <<EOF
Package: voxl-gtsam-vio
Version: $VERSION
Section: base
Priority: optional
Architecture: arm64
Depends: libmodal-pipe(>=2.7.2),libmodal-json(>=0.4.0)
Maintainer: n0t1cs
Description: GTSAM visual-inertial odometry (cpp_gtsam_vio) on MPA pipes, publishes vio_data_t
EOF
echo "/etc/modalai/voxl-gtsam-vio-server.yaml" > "$PKG/DEBIAN/conffiles"
cat > "$PKG/DEBIAN/postinst" <<'EOF'
#!/bin/bash
systemctl daemon-reload 2>/dev/null || true
echo "voxl-gtsam-vio installed. Service NOT enabled. Bench test first, then:"
echo "  systemctl enable --now voxl-gtsam-vio-server"
EOF
chmod 755 "$PKG/DEBIAN/postinst"
DEB="$HERE/voxl-gtsam-vio_${VERSION}_${PLATFORM}_arm64.deb"
dpkg-deb --root-owner-group --build "$PKG" "$DEB"
echo "package: $DEB"
