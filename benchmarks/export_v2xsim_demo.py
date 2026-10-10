"""Export a compact Guardian demo scene from V2X-Sim 2.0 mini (Apache-2.0) into demo/v2xsim/.

Writes downscaled RGB frames for six cameras (vehicle 1 and 4 front cameras, four road-side
cameras) plus per-frame calibration and 3D ground truth.  Only raw sensor data and calibration
are exported - all detections, tracks and alerts are computed live by the server.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.datasets.v2xsim import V2XSim

CAMS = ["CAM_FRONT_id_1", "CAM_FRONT_id_4", "CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]
LABELS = {"CAM_FRONT_id_1": "Vehicle 1 · front", "CAM_FRONT_id_4": "Vehicle 4 · front", "CAM_id_0_0": "RSU · north",
          "CAM_id_0_1": "RSU · east", "CAM_id_0_2": "RSU · south", "CAM_id_0_3": "RSU · west"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="demo/v2xsim")
    ap.add_argument("--start", type=int, default=15)
    ap.add_argument("--end", type=int, default=75)
    ap.add_argument("--width", type=int, default=800)
    args = ap.parse_args()
    ds = V2XSim(args.root)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for f in range(args.start, args.end):
        rec = {"frame": f, "t": round(ds.timestamp(f) - ds.timestamp(args.start), 3), "cams": {}, "gt": []}
        for c in CAMS:
            cam = ds.camera(f, c)
            s = args.width / cam.width
            im = cv2.imread(str(ds.image_path(f, c)))
            im = cv2.resize(im, (args.width, int(round(cam.height * s))), interpolation=cv2.INTER_AREA)
            name = f"{c}_{f:03d}.jpg"
            cv2.imwrite(str(out / "frames" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 72])
            K = cam.K.copy()
            K[:2] *= s
            rec["cams"][c] = {"img": name, "K": np.round(K, 4).tolist(), "R": np.round(cam.R, 6).tolist(),
                              "t": np.round(cam.t, 4).tolist(), "w": args.width, "h": int(round(cam.height * s))}
        for o in ds.ann[f]:
            yaw = float(np.arctan2(o.R[1, 0], o.R[0, 0]))
            rec["gt"].append({"id": o.instance[:8], "cls": o.cls, "xy": np.round(o.ground_xy, 2).tolist(),
                              "yaw": round(yaw, 3), "l": round(float(o.size[1]), 2), "w": round(float(o.size[0]), 2)})
        frames.append(rec)
    meta = {
        "name": "V2X-Sim 2.0 mini · scene_5 · Town03 intersection",
        "source": "https://huggingface.co/datasets/ai4ce-drive/V2X-Sim-2.0-mini",
        "license": "Apache-2.0",
        "citation": "Li et al., V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving, IEEE RA-L 2022",
        "fps": 5.0,
        "cameras": {c: LABELS[c] for c in CAMS},
        "default_ego": "CAM_FRONT_id_1",
        "frames": frames,
    }
    (out / "scene.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("exported", len(frames), "frames to", out)


if __name__ == "__main__":
    main()
