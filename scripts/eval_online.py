#!/usr/bin/env python3
"""Score an online run (a log folder of voxl-gtsam-vio-server or gvio-engine-replay) against ground truth.

ATE: keyframe positions of session<k>/vio_results_est.tum, moved to absolute stamps with meta.txt's t0_ns,
associated to the nearest ground-truth sample within 20 ms, SE(3) Umeyama on positions, RMSE. That is the
MANTIS protocol of python/seedset_eval.py (mantis_gt_align), so the numbers compare with the offline
seed sets. Also: frames, drops, resets and latency from frames.csv and stats.txt.

usage: eval_online.py LOG_DIR --gt MAV0 [--session K] [--json OUT]
  MAV0: the dataset's mav0 with state_groundtruth_estimate0/data.csv on the same clock as the log
"""
import argparse
import json
import pathlib

import numpy as np


def umeyama_se3(P, G):
    mp, mg = P.mean(0), G.mean(0)
    H = (P - mp).T @ (G - mg)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mg - R @ mp


def associate(te, tg, max_diff):
    j = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
    j = np.where(np.abs(tg[j - 1] - te) < np.abs(tg[j] - te), j - 1, j)
    ok = np.abs(tg[j] - te) <= max_diff
    return np.flatnonzero(ok), j[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--gt", type=pathlib.Path, required=True)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()

    meta = (a.log / "meta.txt").read_text().splitlines()
    t0 = int(next(l.split()[1] for l in meta if l.startswith("t0_ns ")))
    est = np.loadtxt(a.log / f"session{a.session}" / "vio_results_est.tum", ndmin=2)
    te = t0 * 1e-9 + est[:, 0]
    g = np.genfromtxt(a.gt / "state_groundtruth_estimate0" / "data.csv", delimiter=",", comments="#")
    tg, pg = g[:, 0] * 1e-9, g[:, 1:4]
    o = np.argsort(tg)
    tg, pg = tg[o], pg[o]
    ie, ig = associate(te, tg, 0.02)
    P, G = est[ie, 1:4], pg[ig]
    R, t = umeyama_se3(P, G)
    ate = float(np.sqrt((np.linalg.norm(P @ R.T + t - G, axis=1) ** 2).mean()))

    fr = np.genfromtxt(a.log / "frames.csv", delimiter=",", names=True)
    ok = fr["state"] == 2
    kf = fr["is_kf"] == 1
    st = {}
    sp = a.log / "stats.txt"
    if sp.exists():
        for line in sp.read_text().splitlines():
            k, v = line.split()
            st[k] = float(v)
    pct = lambda x, q: float(np.percentile(x, q)) if len(x) else float("nan")
    first_ok = float((fr["t_ns"][ok][0] - fr["t_ns"][0]) * 1e-9) if ok.any() else float("nan")
    r = dict(log=str(a.log), session=a.session, ate_m=ate, n_kf=int(len(est)), n_assoc=int(len(ie)),
             frames_out=int(len(fr)), frames_ok=int(ok.sum()), first_ok_after_s=first_ok,
             sessions=int(fr["session"].max()) + 1,
             latency_p50_ms=pct(fr["latency_ms"][ok], 50), latency_p95_ms=pct(fr["latency_ms"][ok], 95),
             latency_max_ms=float(fr["latency_ms"][ok].max()) if ok.any() else float("nan"),
             kf_update_p50_ms=pct(fr["kf_update_ms"][ok & kf], 50), kf_update_p95_ms=pct(fr["kf_update_ms"][ok & kf], 95),
             **{k: st[k] for k in ("frames_in", "dropped_backlog", "dropped_no_imu", "resets", "wall_s") if k in st})
    for k, v in r.items():
        print(f"{k:18s} {v:.4f}" if isinstance(v, float) else f"{k:18s} {v}")
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
