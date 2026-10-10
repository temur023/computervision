"""Bird's-eye orthophoto of the demo intersection, stitched from the four road-side cameras.

Each camera's ground-plane homography (K [r1 r2 t]) maps every map pixel to an image pixel.  We keep
only pixels the dataset's segmentation labels as ground (road, road line, sidewalk, ground, terrain),
take a per-pixel median over all frames with vehicles and pedestrians masked out (so traffic
disappears), and blend cameras with weights that prefer the closer, sharper view.  If the
calibrations were wrong, lane markings from different cameras would not line up: the mosaic is a
visual check of the geometry as much as a basemap.

Usage: python -m benchmarks.build_v2xsim_ortho --root /path/to/v2x_sim_mini
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.datasets.v2xsim import V2XSim

RSU = ["CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]
GROUND = [6, 7, 8, 14, 22]  # CARLA tags: road line, road, sidewalk, ground, terrain
MOVERS = [4, 10]  # pedestrian, vehicle


def background(ds: V2XSim, ch: str, frames, rows_per_chunk=90):
    """Per-pixel median over frames, ignoring pixels covered by road users; plus a ground mask."""
    imgs = np.stack([cv2.imread(str(ds.image_path(f, ch))) for f in frames])  # (F,H,W,3) uint8
    segs = np.stack([ds.seg_depth(f, ch)[0] for f in frames])  # (F,H,W)
    F, H, W, _ = imgs.shape
    out = np.zeros((H, W, 3), np.uint8)
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        x = imgs[:, r0:r1].astype(np.float32)
        moving = np.isin(segs[:, r0:r1], MOVERS)
        x[moving] = np.nan
        med = np.nanmedian(x, axis=0)
        fallback = np.median(imgs[:, r0:r1], axis=0)
        med = np.where(np.isnan(med), fallback, med)
        out[r0:r1] = np.clip(med, 0, 255).astype(np.uint8)
    # a pixel is ground if it is ground in most frames where it is not covered by a road user
    seg_ground = np.isin(segs, GROUND)
    seg_free = ~np.isin(segs, MOVERS)
    frac = seg_ground.sum(0) / np.maximum(1, seg_free.sum(0))
    return out, (frac > 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="demo/v2xsim")
    ap.add_argument("--bbox", default="-48,98,36,178", help="x0,y0,x1,y1 in metres")
    ap.add_argument("--ppm", type=float, default=12.0, help="pixels per metre")
    ap.add_argument("--max-range", type=float, default=60.0)
    args = ap.parse_args()
    ds = V2XSim(args.root)
    x0, y0, x1, y1 = [float(v) for v in args.bbox.split(",")]
    Wm, Hm = int(round((x1 - x0) * args.ppm)), int(round((y1 - y0) * args.ppm))
    xs = x0 + (np.arange(Wm) + 0.5) / args.ppm
    ys = y1 - (np.arange(Hm) + 0.5) / args.ppm  # row 0 = north edge
    GX, GY = np.meshgrid(xs, ys)
    acc = np.zeros((Hm, Wm, 3), np.float64)
    wsum = np.zeros((Hm, Wm), np.float64)
    frames = list(range(0, len(ds), 2))
    for ch in RSU:
        cam = ds.camera(0, ch)
        bg, ground = background(ds, ch, frames)
        H_w2i = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
        P = np.stack([GX, GY, np.ones_like(GX)], -1) @ H_w2i.T
        w = P[..., 2]
        u = (P[..., 0] / np.where(w > 1e-6, w, 1)).astype(np.float32)
        v = (P[..., 1] / np.where(w > 1e-6, w, 1)).astype(np.float32)
        valid = (w > 1e-6) & (u >= 0) & (u < cam.width - 1) & (v >= 0) & (v < cam.height - 1)
        u[~valid] = -1
        v[~valid] = -1
        col = cv2.remap(bg, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gmask = cv2.remap(ground * 255, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
        dist = np.hypot(GX - cam.center_ground[0], GY - cam.center_ground[1])
        wt = valid & gmask & (dist < args.max_range)
        weight = wt / (dist**2 + 25.0)
        # feather towards the image border to hide seams
        edge = np.minimum.reduce([u, v, cam.width - 1 - u, cam.height - 1 - v]).clip(0, 60) / 60.0
        weight = weight * edge
        acc += col.astype(np.float64) * weight[..., None]
        wsum += weight
        print(ch, "covers", round(float(wt.mean()) * 100, 1), "% of the map", flush=True)
    covered = wsum > 1e-9
    rgb = np.zeros((Hm, Wm, 3), np.uint8)
    rgb[covered] = (acc[covered] / wsum[covered, None]).clip(0, 255).astype(np.uint8)
    # map styling: desaturate and darken the ground so data drawn on top carries the colour
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    styled = 0.75 * gray[..., None] + 0.25 * rgb.astype(np.float32)
    styled = (styled * 0.62 + 6).clip(0, 255)
    alpha = cv2.GaussianBlur(covered.astype(np.float32), (0, 0), 2.0)
    base = np.array([18, 15, 13], np.float32)  # #0d0f12 in BGR
    out = styled * alpha[..., None] + base * (1 - alpha[..., None])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(args.out) / "ortho.jpg"), out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 86])
    cv2.imwrite(str(Path(args.out) / "ortho_rgb.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 86])
    scene_p = Path(args.out) / "scene.json"
    if scene_p.exists():
        sc = json.loads(scene_p.read_text())
        sc["ortho"] = {"file": "ortho.jpg", "bbox": [x0, y0, x1, y1], "px_per_m": args.ppm,
                       "method": "stitched from the 4 road-side cameras via their ground homographies; median over frames with road users masked"}
        scene_p.write_text(json.dumps(sc, separators=(",", ":")))
    print("ortho", Wm, "x", Hm, "->", args.out)


if __name__ == "__main__":
    main()
