"""Clean schematic basemap of the demo intersection, built from the dataset's semantic segmentation.

Every camera's segmentation (road, lane marking, sidewalk, ground, terrain) is projected onto the
ground plane through its calibration, and each map pixel takes the class most cameras and frames
agree on.  Road users and their shadows never enter (they are not ground classes), so the map is
free of the smears a photographic mosaic shows.  Rendered in the console's dark palette.

Usage: python -m benchmarks.build_v2xsim_schematic --root /path/to/v2x_sim_mini
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.datasets.v2xsim import V2XSim

CLASSES = {7: "road", 6: "line", 8: "sidewalk", 14: "ground", 22: "terrain"}
PALETTE = {  # BGR
    0: (18, 15, 13),      # unobserved: panel colour #0d0f12
    7: (38, 33, 30),      # road #1e2126
    6: (150, 146, 140),   # lane marking #8c9296
    8: (56, 51, 47),      # sidewalk #2f333a
    14: (52, 48, 45),     # ground
    22: (40, 46, 38),     # terrain, slight green
}


def project(cam, GX, GY):
    H_w2i = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
    P = np.stack([GX, GY, np.ones_like(GX)], -1) @ H_w2i.T
    w = P[..., 2]
    u = (P[..., 0] / np.where(w > 1e-6, w, 1)).astype(np.float32)
    v = (P[..., 1] / np.where(w > 1e-6, w, 1)).astype(np.float32)
    valid = (w > 1e-6) & (u >= 0) & (u < cam.width - 1) & (v >= 0) & (v < cam.height - 1)
    u[~valid] = -1
    v[~valid] = -1
    return u, v, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="demo/v2xsim")
    ap.add_argument("--bbox", default="-42,98,42,176")
    ap.add_argument("--ppm", type=float, default=12.0)
    args = ap.parse_args()
    ds = V2XSim(args.root)
    x0, y0, x1, y1 = [float(v) for v in args.bbox.split(",")]
    Wm, Hm = int(round((x1 - x0) * args.ppm)), int(round((y1 - y0) * args.ppm))
    xs = x0 + (np.arange(Wm) + 0.5) / args.ppm
    ys = y1 - (np.arange(Hm) + 0.5) / args.ppm
    GX, GY = np.meshgrid(xs, ys)
    keys = list(CLASSES)
    votes = np.zeros((len(keys), Hm, Wm), np.float32)
    for f in range(0, len(ds), 2):
        for ch in ds.channels():
            if (f, ch) not in ds.sd:
                continue
            rsu = ch.startswith("CAM_id_0")
            cam = ds.camera(f, ch)
            cx, cy = cam.center_ground
            if not (x0 - 20 < cx < x1 + 20 and y0 - 20 < cy < y1 + 20):
                continue
            seg, _ = ds.seg_depth(f, ch)
            if seg is None:
                continue
            u, v, valid = project(cam, GX, GY)
            dist = np.hypot(GX - cx, GY - cy)
            reach = 60 if rsu else 18
            valid &= dist < reach
            if not valid.any():
                continue
            warped = cv2.remap(seg, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            wt = (valid / (dist**2 + 25.0)).astype(np.float32) * (1.0 if rsu else 0.35)
            for k, cls in enumerate(keys):
                if cls == 6 and not rsu:
                    # lane markings seen from 1.5 m smear into streaks with distance: keep them only up close
                    near = dist < 10
                    votes[k] += ((warped == cls) & near) * wt
                    votes[keys.index(7)] += ((warped == cls) & ~near) * wt
                    continue
                votes[k] += (warped == cls) * wt
    total = votes.sum(0)
    cls_map = np.zeros((Hm, Wm), np.uint8)
    best = np.argmax(votes, 0)
    known = total > total.max() * 1e-4
    cls_map[known] = np.array(keys, np.uint8)[best[known]]
    # tidy: majority filter removes speckle, keep lane markings crisp
    lines = cls_map == 6
    smooth = cv2.medianBlur(np.where(lines, 7, cls_map).astype(np.uint8), 5)
    cls_map = np.where(lines & (smooth == 7), 6, smooth).astype(np.uint8)
    # render at 2x for smooth edges
    up = cv2.resize(cls_map, (Wm * 2, Hm * 2), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((Hm * 2, Wm * 2, 3), np.uint8)
    for cls, col in PALETTE.items():
        img[up == cls] = col
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(args.out) / "map.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    sc_p = Path(args.out) / "scene.json"
    if sc_p.exists():
        sc = json.loads(sc_p.read_text())
        sc["map"] = {"file": "map.png", "bbox": [x0, y0, x1, y1], "px_per_m": args.ppm * 2,
                     "method": "majority vote of the dataset's ground segmentation projected through every camera's calibration"}
        sc_p.write_text(json.dumps(sc, separators=(",", ":")))
    share = {CLASSES[c]: round(float((cls_map == c).mean()) * 100, 1) for c in keys}
    print("schematic", Wm * 2, "x", Hm * 2, share)


if __name__ == "__main__":
    main()
