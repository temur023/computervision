"""Measure per-stage latency on this machine and convert it into cost per frame / camera / km.

Stages are timed on real inputs: bundled V2X-Sim demo frames (road users) and RDD2022 test images
(road damage).  Prices are explicit assumptions (list prices, stated in the output), the throughput is
measured.  Run it on an otherwise idle machine.

Usage: python -m benchmarks.speed_cost --rdd data/rdd_800/images/test [--damage-model models/road_damage_yolo11n.onnx]
"""
import argparse
import glob
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from swarmsight.detector import MODELS_DIR, ROAD_USER_CLASSES, YoloOnnxDetector
from swarmsight.guardian import GuardianScene
from swarmsight.datasets.v2xsim import FPS  # noqa: F401  (documentation: demo frames are 5 Hz)
from swarmsight.geometry import CameraGeometry

# (name, $/hour, basis). Prices are assumptions for the cost model, not measurements.
PRICES = [
    ("Budget cloud VM, 4 vCPU", 0.025, "assumption: ~€16/month list price for a 4 vCPU / 8 GB VM (Hetzner CPX31 class)"),
    ("Hyperscaler VM, 4 vCPU", 0.17, "assumption: on-demand list price of a 4 vCPU compute-optimised instance (AWS c7i.xlarge class)"),
]


def pct(xs):
    return {"p50": round(float(np.percentile(xs, 50)), 1), "p95": round(float(np.percentile(xs, 95)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd", required=True, help="directory of RDD2022 test images")
    ap.add_argument("--damage-model", default=str(MODELS_DIR / "road_damage_yolo11n.onnx"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="benchmarks/results/speed_cost.json")
    args = ap.parse_args()
    users = YoloOnnxDetector(MODELS_DIR / "yolo11n_coco.onnx", conf=0.3, classes=ROAD_USER_CLASSES)
    damage = YoloOnnxDetector(args.damage_model, conf=0.25)
    demo = sorted(glob.glob("demo/v2xsim/frames/*.jpg"))[: args.n]
    rdd = sorted(glob.glob(os.path.join(args.rdd, "*.jpg")))[: args.n]
    for p in demo[:5]:
        users(cv2.imread(p))
    for p in rdd[:5]:
        damage(cv2.imread(p))
    t_dec, t_users, t_dmg, t_both = [], [], [], []
    for p in demo:
        t0 = time.perf_counter()
        img = cv2.imread(p)
        t1 = time.perf_counter()
        users(img)
        t2 = time.perf_counter()
        t_dec.append((t1 - t0) * 1e3)
        t_users.append((t2 - t1) * 1e3)
    for p in rdd:
        img = cv2.imread(p)
        t0 = time.perf_counter()
        damage(img)
        t1 = time.perf_counter()
        users(img)
        t2 = time.perf_counter()
        t_dmg.append((t1 - t0) * 1e3)
        t_both.append((t2 - t0) * 1e3)
    # Guardian fusion + reasoning on the bundled scene (detections precomputed so only fusion is timed)
    scene = json.loads(Path("demo/v2xsim/scene.json").read_text())
    cams = ["CAM_FRONT_id_1", "CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]
    dets = {}
    for fr in scene["frames"][:30]:
        for c in cams:
            dets[(fr["frame"], c)] = users(cv2.imread(f"demo/v2xsim/frames/{fr['cams'][c]['img']}"))
    g = GuardianScene()
    for c in cams:
        g.add_camera(c, None)
    t_fuse = []
    for fr in scene["frames"][:30]:
        t0 = time.perf_counter()
        for c in cams[1:] + cams[:1]:
            cd = fr["cams"][c]
            g.ingest(c, dets[(fr["frame"], c)], fr["t"], CameraGeometry.from_KRt(cd["K"], cd["R"], cd["t"], cd["w"], cd["h"]))
        g.view_for(cams[0], fr["t"])
        t_fuse.append((time.perf_counter() - t0) * 1e3)

    scan_ms = float(np.median(t_both))
    fps = 1000.0 / scan_ms
    cost = []
    for name, usd_h, basis in PRICES:
        per_1000 = usd_h / (fps * 3600) * 1000
        cam_month = usd_h * (1.0 * 3600 * 10 * 30) / (fps * 3600)  # 1 analysed frame/s, 10 h/day, 30 days
        per_km = usd_h * (600 / fps) / 3600  # 5 frames/s at 30 km/h -> 600 frames per km
        cost.append({"name": name, "basis": basis, "usd_per_hour": usd_h, "fps": round(fps, 1),
                     "usd_per_1000_frames": round(per_1000, 5), "usd_per_camera_month": round(cam_month, 3),
                     "usd_per_km_scanned": round(per_km, 5)})
    res = {
        "hardware": f"{os.cpu_count()} vCPU {platform.machine()} · ONNX Runtime CPU · measured {time.strftime('%Y-%m-%d')}",
        "stages": {
            "JPEG decode (800x450)": pct(t_dec),
            "Road users, YOLO11n 640 px": pct(t_users),
            "Road damage, our YOLO11n 512 px": pct(t_dmg),
            "Scan: both detectors per frame": pct(t_both),
            "Guardian: tracking + fusion + reasoning, 5 cameras": pct(t_fuse),
        },
        "cost": cost,
        "note": ("Throughput is measured (single process, one frame at a time). Cost per camera-month assumes 1 analysed "
                 "frame per second for 10 hours a day; cost per km assumes 5 frames/s at 30 km/h. Prices are list-price "
                 "assumptions stated per row. Traffic, storage and moderation labour are in the business model, not here."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
