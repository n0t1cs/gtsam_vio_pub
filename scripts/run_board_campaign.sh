#!/usr/bin/env bash
# =============================================================================
# run_board_campaign.sh: seed sets of voxl-gtsam-vio-server ON the VOXL2. ModalAI's voxl-replay
# publishes a recorded log on the board's MPA pipes in real time and voxl-gtsam-vio-server consumes
# them, as on the vehicle. A seed set per log and backend (iSAM2 and the SLKS), one run at a time, in
# one session: per seed, each log with both backends back to back, the backend that goes first
# alternating with the seed, so neither warm-up order nor thermal drift favours one backend.
#
#   bash scripts/run_board_campaign.sh [--out DIR] [--logs "MH_01_easy ..."] [--seeds "0 1 2 3 4"]
#                                      [--backends "incremental keyframe"] [--cooldown S]
#                                      [--opencv-threads N] [--covariance async|inline] [--dry-run]
#
# --opencv-threads N sets the server's opencv_threads (0 = OpenCV runs in the calling thread), so a
# stage's wall time in frames.csv is also its CPU time. --covariance sets realtime.async_covariance
# (the keyframe covariances on their own thread, or in the estimator thread). Without them the
# config's values are kept.
#
# Two kinds of log, told apart by name, each run with its own configuration:
#   EuRoC   MH_* V1_* V2_*  /data/logs/euroc/<seq>/ (a EuRoC sequence written in the voxl-logger layout,
#                           cam0 as tracking_front_misp_norm, imu0 as imu_apps), estimator
#                           /data/logs/euroc/config/estimator_euroc_{mh,vicon}.yaml, server
#                           /data/logs/euroc/config/voxl-gtsam-vio-server_euroc.yaml.
#                           Default: the 11 sequences.
#   VOXL    log<NNNN>       /data/logs/<log>/ (a voxl-logger log), the estimator and server configs of
#                           this repository (config/estimator_mantis2.yaml, config/voxl-gtsam-vio-server.yaml),
#                           e.g. --logs "log0003 log0000".
# Every log holds info.json, run/mpa/imu_apps, run/mpa/tracking_front_misp_norm, data/modalai and
# etc/modalai (its own calibration, which the server is pointed at).
#
# Needs, as root: this repository on the board (e.g. /data/voxl-gtsam-vio), built there by
# scripts/build_voxl2_native.sh. The server is build/voxl-gtsam-vio-server of the same checkout.
#
# Per run, OUT/<log>_<isam2|slks>_s<seed>/: estimator.yaml and server.yaml (generated, every rewritten
# line checked), server.log, replay.log, times.txt (replay start and end), proc.csv (1 Hz: server CPU
# ticks, RSS, board CPU ticks, hottest thermal zone) and srv/<start time>/, the server's own log
# (frames.csv, events.log, meta.txt, session*/). DONE marks a finished run: a re-run skips it and
# repeats an unfinished one from scratch. OUT/campaign.txt records the binary, the commit, the configs
# and the board. --dry-run only writes and checks the configs.
#
# voxl-camera-server, voxl-imu-server and the on-board VIO servers are stopped for the campaign
# (voxl-replay's -y never fires: it checks the pipe without its leading /), and the ones that were
# running are started again at the end, also after Ctrl-C.
#
# Before every run the page cache is dropped and memory compacted, then the server, voxl-replay and
# their libraries are read back (a cold server lost frames at the start). Without the drop, after a
# few tens of runs the kernel refused voxl-replay the camera pipe ("failed to set pipe size: Cannot
# allocate memory", "PIPE NOT CREATED"), the server received IMU only and the run still looked finished
# (12 of 56 runs in one campaign). An infrastructure failure (the server did not start, the replay
# could not create a pipe or hung past REPLAY_TIMEOUT, the server received no frame, which includes a
# server that died at startup) writes no DONE: the run is moved to OUT_failed/ and repeated, up to 3
# attempts, then the campaign stops. A server that exits early after its first frame or an estimator
# that resets is a result and is never repeated. A stopped service that became active again stops the
# campaign.
#
# Launch detached, so that a dropped ssh session cannot stop it (no tmux on the board):
#   setsid nohup bash /data/voxl-gtsam-vio/scripts/run_board_campaign.sh --out /data/gvio_runs \
#       > /data/gvio_runs_console.log 2>&1 < /dev/null &
#   tail -f /data/gvio_runs_console.log                    (Ctrl-C here stops only the viewer)
#   kill -INT -- -$(pgrep -of 'bash /data/voxl-gtsam-vio/scripts/run_board_campaign.sh')
#                                                          (stops the campaign, services restored)
# Copy OUT to a desktop to score it: every run's srv/<start time>/ is the server's log, which
# scripts/eval_online.py scores against ground truth.
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"   # this repository
BIN=$REPO/build/voxl-gtsam-vio-server
LOGS=/data/logs
EUROC=$LOGS/euroc
OUT=/data/gvio_runs
LOG_LIST="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
SEEDS="0 1 2 3 4"
BACKENDS="incremental keyframe"
COOLDOWN=20
OCV=""              # --opencv-threads, empty = the server config's value
COV=""              # --covariance, empty = the server config's value
REPLAY_TIMEOUT=900   # s, five times the longest log
DRY=0
SVCS="voxl-camera-server voxl-imu-server voxl-qvio-server voxl-open-vins-server"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --logs) LOG_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --opencv-threads) OCV="$2"; shift 2 ;;
    --covariance) COV="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^# =====/p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
done
OUT=$(realpath -m "$OUT")   # the server configs carry absolute paths
FAILED="${OUT}_failed"      # failed attempts, kept as evidence

die() { echo "ERROR: $*" >&2; exit 1; }

# per log: its folder, its estimator config and its server config
logdir() { case "$1" in log[0-9]*) echo "$LOGS/$1" ;; MH_*|V1_*|V2_*) echo "$EUROC/$1" ;; *) return 1 ;; esac; }
ecfg() {
  case "$1" in
    log[0-9]*) echo "$REPO/config/estimator_mantis2.yaml" ;;
    MH_*) echo "$EUROC/config/estimator_euroc_mh.yaml" ;;
    V1_*|V2_*) echo "$EUROC/config/estimator_euroc_vicon.yaml" ;;
  esac
}
scfg() { case "$1" in log[0-9]*) echo "$REPO/config/voxl-gtsam-vio-server.yaml" ;; *) echo "$EUROC/config/voxl-gtsam-vio-server_euroc.yaml" ;; esac; }

[[ $(id -u) -eq 0 ]] || die "run as root"
[[ -f "$BIN" ]] || die "$BIN missing"
command -v voxl-replay >/dev/null || die "voxl-replay missing"
pidof voxl-gtsam-vio-server voxl-replay >/dev/null && die "a voxl-gtsam-vio-server or voxl-replay is already running"
# the server, voxl-replay and every shared library they load, read back after each cache drop
WARM=$(for f in "$BIN" "$(command -v voxl-replay)"; do
         echo "$f"; ldd "$f" 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^\//) print $i}'
       done | sort -u | xargs)
for l in $LOG_LIST; do
  d=$(logdir "$l") || die "unknown log $l (expected log<NNNN>, MH_*, V1_* or V2_*)"
  for f in info.json run/mpa/imu_apps/data.csv run/mpa/tracking_front_misp_norm/data.csv \
           data/modalai/opencv_tracking_front_intrinsics.yml etc/modalai/extrinsics.conf; do
    [[ -f "$d/$f" ]] || die "$d/$f missing"
  done
  e=$(ecfg "$l"); s=$(scfg "$l")
  [[ -f "$e" && -f "$s" ]] || die "config missing for $l: $e or $s"
  [[ $(grep -c '^runtime:' "$e") -eq 1 ]] || die "$e: expected one top-level runtime: block"
  [[ $(grep -cE '^  backend_mode: incremental( .*)?$' "$e") -eq 1 ]] || die "$e: expected one '  backend_mode: incremental' line"
done
for b in $BACKENDS; do [[ $b == incremental || $b == keyframe ]] || die "unknown backend $b"; done
[[ -z $OCV || $OCV =~ ^-?[0-9]+$ ]] || die "--opencv-threads needs an integer"
[[ -z $COV || $COV == async || $COV == inline ]] || die "--covariance needs async or inline"
tag() { [[ $1 == keyframe ]] && echo slks || echo isam2; }
mkdir -p "$OUT"

# make_cfgs DIR LOG SEED BACKEND: the log's estimator config with runtime.rng_seed and runtime.backend_mode,
# the log's server config pointing at it and at the log's own calibration
make_cfgs() {
  local d=$1 l=$2 k=$3 b=$4 E S L
  E=$(ecfg "$l"); S=$(scfg "$l"); L=$(logdir "$l")
  if grep -q '^  rng_seed:' "$E"; then
    sed -E -e "s/^  rng_seed:.*/  rng_seed: $k/" -e "s/^  backend_mode: incremental( .*)?\$/  backend_mode: $b/" "$E" > "$d/estimator.yaml"
  else
    sed -E -e "s/^runtime:.*/&\n  rng_seed: $k/" -e "s/^  backend_mode: incremental( .*)?\$/  backend_mode: $b/" "$E" > "$d/estimator.yaml"
  fi
  sed -e "s#^estimator_config:.*#estimator_config: $d/estimator.yaml#" \
      -e "s#^intrinsics_file:.*#intrinsics_file: $L/data/modalai/opencv_tracking_front_intrinsics.yml#" \
      -e "s#^extrinsics_file:.*#extrinsics_file: $L/etc/modalai/extrinsics.conf#" \
      -e "s#^log_dir:.*#log_dir: $d/srv#" "$S" > "$d/server.yaml"
  if [[ -n $OCV ]]; then
    if grep -q '^opencv_threads:' "$d/server.yaml"; then sed -i "s/^opencv_threads:.*/opencv_threads: $OCV/" "$d/server.yaml"
    else echo "opencv_threads: $OCV" >> "$d/server.yaml"; fi
    [[ $(grep -c "^opencv_threads: $OCV\$" "$d/server.yaml") -eq 1 ]] || return 1
  fi
  if [[ -n $COV ]]; then
    local cv=false; [[ $COV == async ]] && cv=true
    if grep -q '^  async_covariance:' "$d/server.yaml"; then sed -i "s/^  async_covariance:.*/  async_covariance: $cv/" "$d/server.yaml"
    elif grep -q '^realtime:' "$d/server.yaml"; then sed -i "s/^realtime:.*/&\n  async_covariance: $cv/" "$d/server.yaml"
    else printf 'realtime:\n  async_covariance: %s\n' "$cv" >> "$d/server.yaml"; fi
    [[ $(grep -c "^  async_covariance: $cv\$" "$d/server.yaml") -eq 1 ]] || return 1
  fi
  # each rewrite landed exactly once, nothing else differs (estimator: the seed line added or replaced,
  # plus the backend line when it changed or lost its comment, server: only the four keys, which the
  # template may already hold, as the EuRoC template does for MH_01_easy's calibration)
  [[ $(grep -c "^  rng_seed: $k\$" "$d/estimator.yaml") -eq 1 ]] || return 1
  [[ $(grep -c "^  backend_mode: $b\$" "$d/estimator.yaml") -eq 1 ]] || return 1
  local n v; n=$(diff "$E" "$d/estimator.yaml" | grep -c '^[<>]')
  [[ $n -ge 1 && $n -le 4 ]] || return 1
  [[ $(diff "$S" "$d/server.yaml" | grep '^[<>]' \
        | grep -cvE '^[<>] ((estimator_config|intrinsics_file|extrinsics_file|log_dir|opencv_threads|  async_covariance): |realtime:$)') -eq 0 ]] || return 1
  for v in "estimator_config: $d/estimator.yaml" "intrinsics_file: $L/data/modalai/opencv_tracking_front_intrinsics.yml" \
           "extrinsics_file: $L/etc/modalai/extrinsics.conf" "log_dir: $d/srv"; do
    [[ $(grep -cxF "$v" "$d/server.yaml") -eq 1 ]] || return 1
  done
}

# alive PID: running and not a zombie (an exited child stays visible to kill -0 until it is reaped)
alive() { local s; s=$(ps -o stat= -p "$1" 2>/dev/null); [[ -n "$s" && "$s" != Z* ]]; }
wait_gone() { local i; for ((i = 0; i < $2 * 2; i++)); do alive "$1" || return 0; sleep 0.5; done; return 1; }

# sample PID: 1 Hz until PID ends. Server CPU ticks (utime+stime), RSS and peak RSS (kB), board CPU busy
# and total ticks (first line of /proc/stat), hottest thermal zone (milli deg C)
sample() {
  local f c rss hwm tz
  echo "t_s,srv_ticks,rss_kb,hwm_kb,cpu_busy,cpu_total,temp_mC"
  while alive "$1"; do
    read -r -a f 2>/dev/null < "/proc/$1/stat" || break
    rss=$(awk '/^VmRSS/ {print $2}' "/proc/$1/status" 2>/dev/null)
    hwm=$(awk '/^VmHWM/ {print $2}' "/proc/$1/status" 2>/dev/null)
    read -r -a c < /proc/stat
    tz=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
    echo "$(date +%s.%N),$(( f[13] + f[14] )),${rss:-0},${hwm:-0},$(( c[1]+c[2]+c[3]+c[6]+c[7]+c[8] )),$(( c[1]+c[2]+c[3]+c[4]+c[5]+c[6]+c[7]+c[8] )),${tz:-0}"
    sleep 1
  done
}

SRV=""; SRV_RC=""; WAS_ACTIVE=""; RESTORED=0
stop_server() {
  [[ -n "$SRV" ]] || return 0
  if alive "$SRV"; then
    kill -INT "$SRV"
    wait_gone "$SRV" 30 || { echo "WARNING: server did not stop in 30 s, killed"; kill -KILL "$SRV"; }
  fi
  wait "$SRV" 2>/dev/null; SRV_RC=$?
  SRV=""
}
restore() {
  [[ $RESTORED -eq 1 ]] && return
  RESTORED=1
  pidof voxl-replay >/dev/null && { kill -INT $(pidof voxl-replay); sleep 1; }
  stop_server
  if [[ -n "$WAS_ACTIVE" ]]; then echo "starting again:$WAS_ACTIVE"; systemctl start $WAS_ACTIVE; fi
}

# run_once DIR LOG SEED BACKEND LABEL: one run in DIR (configs already written). Returns 0 with DONE
# written, or 1 with the reason in FAIL after an infrastructure failure, which the caller repeats.
FAIL=""
run_once() {
  local d=$1 l=$2 k=$3 b=$4 s w rc early frames
  for s in $SVCS; do systemctl is-active --quiet "$s" && die "$s became active again during the campaign"; done
  sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
  # the drop also evicts the code of the server and of voxl-replay: read it back, so that a run starts
  # as warm as runs 1 to 47 did (cold, the first runs after the drop lost 4 to 7 frames at the start)
  cat $WARM > /dev/null
  echo "$5 $l $(tag "$b") seed $k  $(date +%T)"
  stdbuf -oL -eL "$BIN" -c "$d/server.yaml" -v > "$d/server.log" 2>&1 &
  SRV=$!
  sample "$SRV" > "$d/proc.csv" &
  for ((w = 0; w < 60; w++)); do
    grep -q 'running, publishing' "$d/server.log" && break
    alive "$SRV" || break
    sleep 0.5
  done
  if ! grep -q 'running, publishing' "$d/server.log"; then
    stop_server; wait; FAIL="the server did not start (server.log)"; return 1
  fi
  tail -n +1 --pid="$SRV" -f "$d/server.log" | grep --line-buffered '^\[status\]' &

  echo "replay_start $(date +%s.%N)" > "$d/times.txt"
  timeout --foreground -s INT -k 30 "$REPLAY_TIMEOUT" \
    voxl-replay -p "$(logdir "$l")/" -i imu_apps -i tracking_front_misp_norm < /dev/null > "$d/replay.log" 2>&1
  rc=$?
  echo "replay_end $(date +%s.%N) rc $rc" >> "$d/times.txt"
  sleep 3
  early=0; alive "$SRV" || { early=1; echo "WARNING: the server exited before the replay ended"; }
  stop_server
  wait   # the sampler and the status tail end with the server

  [[ $rc -eq 124 || $rc -eq 137 ]] && { FAIL="voxl-replay hung, stopped after $REPLAY_TIMEOUT s"; return 1; }
  frames=$(ls "$d"/srv/*/frames.csv 2>/dev/null | head -1)
  [[ -n "$frames" ]] || { FAIL="no frames.csv under srv/ (replay rc $rc)"; return 1; }
  echo "  -> $(grep '^\[status\]' "$d/server.log" | tail -1 | cut -d'|' -f2 | xargs) | replay rc $rc, server rc $SRV_RC"
  grep -qE 'PIPE NOT CREATED|failed to set pipe size' "$d/replay.log" \
    && { FAIL="voxl-replay could not create a pipe (replay.log)"; return 1; }
  [[ $(wc -l < "$frames") -gt 1 ]] || { FAIL="the server received 0 frames"; return 1; }
  echo "replay_rc $rc server_rc $SRV_RC server_exited_early $early" > "$d/DONE"
}

if [[ $DRY -eq 0 ]]; then
  {
    echo "# board campaign $(date -Is)"
    echo "server md5 $(md5sum < "$BIN" | cut -c1-32)"
    echo "commit $(git -C "$REPO" log --oneline -1 2>/dev/null)"
    for f in $(for l in $LOG_LIST; do ecfg "$l"; scfg "$l"; done | sort -u); do echo "config $(md5sum "$f")"; done
    echo "voxl-logger $(dpkg-query -W -f='${Version}' voxl-logger 2>/dev/null)"
    echo "governor $(sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c | xargs)"
    echo "clk_tck $(getconf CLK_TCK)"
    echo "kernel $(uname -r), $(nproc) cores, $(free -m | awk '/^Mem/ {print $2}') MB"
    echo "logs $LOG_LIST | seeds $SEEDS | backends $BACKENDS | cooldown $COOLDOWN s | opencv_threads ${OCV:-as in the server config} | covariance ${COV:-as in the server config}"
    echo "warm after each drop: $(wc -w <<< "$WARM") files ($(cat $WARM | wc -c) bytes), the server, voxl-replay and their libraries"
    echo "per run: drop_caches 3 and compact_memory before the server starts, up to 3 attempts after an infrastructure failure (failed attempts in ${FAILED})"
  } >> "$OUT/campaign.txt"
  for s in $SVCS; do systemctl is-active --quiet "$s" && WAS_ACTIVE="$WAS_ACTIVE $s"; done
  echo "services running before:${WAS_ACTIVE:- none}" | tee -a "$OUT/campaign.txt"
  trap restore EXIT
  trap 'echo; echo "interrupted"; exit 130' INT TERM HUP
  for s in $SVCS; do systemctl stop "$s" 2>/dev/null; done
  for s in $SVCS; do systemctl is-active --quiet "$s" && die "$s is still active"; done
fi

N=$(( $(wc -w <<< "$SEEDS") * $(wc -w <<< "$LOG_LIST") * $(wc -w <<< "$BACKENDS") )); i=0
for k in $SEEDS; do
  order="$BACKENDS"; (( k % 2 == 1 )) && order=$(tr ' ' '\n' <<< "$BACKENDS" | tac | xargs)
  for l in $LOG_LIST; do
   for b in $order; do
    i=$((i + 1)); d="$OUT/${l}_$(tag "$b")_s$k"
    if [[ -f "$d/DONE" ]]; then echo "[$i/$N] $l $(tag "$b") seed $k: done before, skipped"; continue; fi
    for a in 1 2 3; do
      rm -rf "$d"; mkdir -p "$d/srv"
      make_cfgs "$d" "$l" "$k" "$b" || die "config rewrite check failed in $d"
      [[ $DRY -eq 1 ]] && break
      run_once "$d" "$l" "$k" "$b" "[$i/$N]" && break
      echo "WARNING: [$i/$N] $l $(tag "$b") seed $k, attempt $a of 3 failed: $FAIL" | tee -a "$OUT/campaign.txt"
      mkdir -p "$FAILED"; mv "$d" "$FAILED/$(basename "$d")_attempt${a}_$(date +%H%M%S)"
      [[ $a -eq 3 ]] && die "[$i/$N] $l $(tag "$b") seed $k failed 3 times, attempts in $FAILED"
      sleep "$COOLDOWN"
    done
    if [[ $DRY -eq 1 ]]; then echo "[$i/$N] $l $(tag "$b") seed $k: configs checked in $d"; continue; fi
    [[ $i -lt $N ]] && sleep "$COOLDOWN"
   done
  done
done
[[ $DRY -eq 1 ]] && { echo "dry run: configs only, nothing run"; exit 0; }
echo "campaign finished $(date -Is): $OUT" | tee -a "$OUT/campaign.txt"
