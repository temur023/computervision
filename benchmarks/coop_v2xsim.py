"""Cooperative perception benchmark on V2X-Sim 2.0 mini (CARLA, Apache-2.0).

For each ego vehicle (camera A = its front camera) and every frame we compare what A knows
from its own camera with what it knows after Guardian fusion with other cameras:

* recall of road users inside A's field of view (<= range) - single camera vs fused
* recall restricted to objects that are *occluded* for A (ground truth from segmentation + depth)
* alert precision: share of external-observation alerts that correspond to a real object
* localisation error of fused positions against 3D ground truth
* compute latency per stage and message size

Usage: python -m benchmarks.coop_v2xsim --root /path/to/v2x_sim_mini [--range 40]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from swarmsight.datasets.v2xsim import V2XSim
from swarmsight.detector import Detections, load_default_detectors
from swarmsight.guardian import GuardianScene

GROUP_THR = {"person": 2.0, "bicycle": 2.0, "motorcycle": 2.0, "car": 3.0, "truck": 3.5, "bus": 4.0}
OCCLUDED_PX = 400  # fewer depth-consistent object pixels than this => object hidden from A
RSU = ["CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]


def detect_cached(ds, det, cache: Path, frame: int, ch: str, timings: list) -> Detections:
    f = cache / f"{ch}_{frame:03d}.npz"
    if f.exists():
        z = np.load(f)
        return Detections(z["xyxy"], z["conf"], z["cls"], det.names)
    im = cv2.imread(str(ds.image_path(frame, ch)))
    t = time.perf_counter()
    d = det(im)
    timings.append((time.perf_counter() - t) * 1e3)
    np.savez(f, xyxy=d.xyxy, conf=d.conf, cls=d.cls)
    return d


def match(gt_xy, est_xy, thr):
    if len(gt_xy) == 0 or len(est_xy) == 0:
        return []
    D = np.linalg.norm(gt_xy[:, None] - est_xy[None], axis=2)
    C = np.where(D <= thr[:, None], D, 1e6)
    r, c = linear_sum_assignment(C)
    return [(i, j, D[i, j]) for i, j in zip(r, c) if C[i, j] < 1e6]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def run(ds, det, ego: int, sources: list[str], rng: float, cache: Path, det_ms: list):
    A = f"CAM_FRONT_id_{ego}"
    scene = GuardianScene(ttl=1.0, alert_radius_m=rng)
    for c in [A] + sources:
        scene.add_camera(c, None)
    stats = {k: 0 for k in ["gt_fov", "gt_visible", "gt_occluded", "gt_occluded_vru", "single_hit", "single_hit_visible",
                            "fused_hit", "fused_hit_occluded", "fused_hit_occluded_vru", "alerts", "alerts_true",
                            "alerts_true_hidden", "alerts_vru", "alerts_vru_true", "frames"]}
    loc_single, loc_fused, fusion_ms, msg_bytes = [], [], [], []
    for f in range(len(ds)):
        t = ds.timestamp(f)
        t0 = time.perf_counter()
        for c in sources + [A]:
            d = detect_cached(ds, det, cache, f, c, det_ms)
            scene.ingest(c, d, t, ds.camera(f, c))
        view = scene.view_for(A, t)
        fusion_ms.append((time.perf_counter() - t0) * 1e3)
        if f < 5:  # tracker / fusion warm-up
            continue
        stats["frames"] += 1
        camA = ds.camera(f, A)
        vis = ds.visibility(f, A, camA, max_range=rng)
        ego_xy = ds.ego_position(f, ego)[:2]
        gts = [o for o in ds.ann[f] if vis[o.instance]["in_fov"] and np.linalg.norm(o.ground_xy - ego_xy) > 2.0]
        gt_xy = np.array([o.ground_xy for o in gts]).reshape(-1, 2)
        thr = np.array([GROUP_THR.get(o.cls, 3.0) for o in gts])
        occluded = np.array([vis[o.instance]["visible_px"] < OCCLUDED_PX for o in gts], bool)
        vru = np.array([o.cls in ("person", "bicycle", "motorcycle") for o in gts], bool)
        stats["gt_fov"] += len(gts)
        stats["gt_occluded"] += int(occluded.sum())
        stats["gt_visible"] += int((~occluded).sum())
        stats["gt_occluded_vru"] += int((occluded & vru).sum())

        # --- single camera: A's own tracks, matched in the image (IoU) ---
        own = view["own_tracks"]
        single_hit = np.zeros(len(gts), bool)
        if own and gts:
            C = np.ones((len(gts), len(own)))
            for i, o in enumerate(gts):
                box = vis[o.instance]["box"]
                if box is None:
                    continue
                for j, tr in enumerate(own):
                    C[i, j] = 1 - iou(box, tr["bbox"])
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] <= 0.7:
                    single_hit[i] = True
                    if "xy" in own[j]:
                        loc_single.append(float(np.linalg.norm(np.array(own[j]["xy"]) - gt_xy[i])))
        stats["single_hit"] += int(single_hit.sum())
        stats["single_hit_visible"] += int((single_hit & ~occluded).sum())

        # --- fused: own hits + external objects delivered by the network ---
        ext = [o for o in view["objects"] if o["status"] != "visible" and o["sources"]]
        ext_xy = np.array([o["xy"] for o in ext]).reshape(-1, 2)
        fused_hit = single_hit.copy()
        remaining = np.where(~single_hit)[0]
        for i, j, dist in match(gt_xy[remaining], ext_xy, thr[remaining]):
            fused_hit[remaining[i]] = True
        # localisation of fused global objects (all statuses) vs GT
        all_xy = np.array([o["xy"] for o in view["objects"]]).reshape(-1, 2)
        for i, j, dist in match(gt_xy, all_xy, thr):
            loc_fused.append(float(dist))
        stats["fused_hit"] += int(fused_hit.sum())
        stats["fused_hit_occluded"] += int((fused_hit & occluded).sum())
        stats["fused_hit_occluded_vru"] += int((fused_hit & occluded & vru).sum())

        # --- alert precision against all GT objects (any visibility, any distance) ---
        alerts = view["alerts"]
        all_gt = ds.ann[f]
        all_xy_gt = np.array([o.ground_xy for o in all_gt]).reshape(-1, 2)
        for a in alerts:
            msg_bytes.append(len(json.dumps(a)))
            stats["alerts"] += 1
            stats["alerts_vru"] += int(a["vru"])
            if len(all_xy_gt) == 0:
                continue
            d = np.linalg.norm(all_xy_gt - np.array(a["position_m"]), axis=1)
            k = int(d.argmin())
            if d[k] <= GROUP_THR.get(all_gt[k].cls, 3.0):
                stats["alerts_true"] += 1
                stats["alerts_vru_true"] += int(a["vru"])
                v = vis.get(all_gt[k].instance)
                if v is None or v["visible_px"] < OCCLUDED_PX or not v["in_fov"]:
                    stats["alerts_true_hidden"] += 1
    return stats, loc_single, loc_fused, fusion_ms, msg_bytes


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def summarize(stats, loc_single, loc_fused, fusion_ms, msg_bytes):
    s = stats
    return {
        "frames": s["frames"],
        "gt_objects_in_fov": s["gt_fov"],
        "gt_occluded_for_ego": s["gt_occluded"],
        "gt_occluded_vru": s["gt_occluded_vru"],
        "recall_single_camera_%": pct(s["single_hit"], s["gt_fov"]),
        "recall_single_camera_visible_only_%": pct(s["single_hit_visible"], s["gt_visible"]),
        "recall_fused_%": pct(s["fused_hit"], s["gt_fov"]),
        "occluded_recall_single_%": 0.0 if s["gt_occluded"] else None,
        "occluded_recall_fused_%": pct(s["fused_hit_occluded"], s["gt_occluded"]),
        "occluded_vru_recall_fused_%": pct(s["fused_hit_occluded_vru"], s["gt_occluded_vru"]),
        "alerts": s["alerts"],
        "alert_precision_%": pct(s["alerts_true"], s["alerts"]),
        "alert_true_and_hidden_%": pct(s["alerts_true_hidden"], s["alerts"]),
        "vru_alert_precision_%": pct(s["alerts_vru_true"], s["alerts_vru"]),
        "loc_err_single_m_median": round(float(np.median(loc_single)), 2) if loc_single else None,
        "loc_err_fused_m_median": round(float(np.median(loc_fused)), 2) if loc_fused else None,
        "loc_err_fused_m_p90": round(float(np.percentile(loc_fused, 90)), 2) if loc_fused else None,
        "fusion_frame_ms_p50": round(float(np.median(fusion_ms)), 1),
        "alert_msg_bytes_mean": round(float(np.mean(msg_bytes)), 0) if msg_bytes else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--range", type=float, default=40.0)
    ap.add_argument("--egos", default="1,2,4")
    ap.add_argument("--out", default="benchmarks/results/coop_v2xsim.json")
    args = ap.parse_args()
    ds = V2XSim(args.root)
    det = load_default_detectors()["road_users"]
    cache = Path(args.root) / "_det_cache"
    cache.mkdir(exist_ok=True)
    egos = [int(e) for e in args.egos.split(",")]
    det_ms: list[float] = []
    results = {"dataset": "V2X-Sim 2.0 mini, scene_5 (Town03 intersection), 100 frames @5 Hz",
               "detector": det.name, "range_m": args.range, "configs": {}}
    configs = {
        "rsu": lambda ego: RSU,
        "rsu+v2v": lambda ego: RSU + [f"CAM_FRONT_id_{e}" for e in egos if e != ego],
    }
    for name, srcs in configs.items():
        tot = None
        ls_all, lf_all, fm_all, mb_all = [], [], [], []
        per_ego = {}
        for ego in egos:
            st, ls, lf, fm, mb = run(ds, det, ego, srcs(ego), args.range, cache, det_ms)
            per_ego[f"vehicle_{ego}"] = summarize(st, ls, lf, fm, mb)
            tot = st if tot is None else {k: tot[k] + st[k] for k in tot}
            ls_all += ls
            lf_all += lf
            fm_all += fm
            mb_all += mb
        results["configs"][name] = {"overall": summarize(tot, ls_all, lf_all, fm_all, mb_all), "per_ego": per_ego}
    if det_ms:
        results["detector_ms_per_image_p50"] = round(float(np.median(det_ms)), 1)
        results["detector_ms_per_image_p95"] = round(float(np.percentile(det_ms, 95)), 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
