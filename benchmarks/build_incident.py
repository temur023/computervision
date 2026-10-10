"""Package the Tashkent dashcam case for the Guardian page: anonymised frames, telemetry, calibration, annotations.

Input: the team's dashcam clip (Tashkent, 2026-07-16 18:27:53 local time).  Output (demo/incident/):
  frames/*.jpg  20.0-29.75 s of the clip at 8 frames/s, 960x540.  Faces (top of every person box) and plate bands
                (lower part of every vehicle box, full width) are blurred, using detections from the frame and its
                four neighbours so that a missed detection in one frame does not leak a face or a plate (hackathon
                rule: no faces or plates of people who did not consent).  The replay stops at the
                moment of impact: we show the conflict, not its outcome.
  scene.json    telemetry read from the dashcam overlay, camera calibration, our manual annotations of the
                motorcycle (the detector misses it: 25-50 px, motion blur), sources and assumptions.

Everything else (tracking the car ahead, positions, speeds, the CrossGuard counterfactual) is computed by
swarmsight.incident on the server.

Usage: python -m benchmarks.build_incident --video clip.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.detector import YoloOnnxDetector, MODELS_DIR
from swarmsight.privacy import _blur

FPS_SRC = 24.0
T0, T1, STEP = 20.0, 29.75, 0.125
OUT_W = 960

# Read by us from the dashcam's on-screen overlay, one line per second: video time (s), latitude, longitude, km/h.
# Video time 0.0 s = 18:27:53 on the overlay clock.
TELEMETRY = [
    (0, 41.32853, 69.35432, 0), (14, 41.32853, 69.35432, 0), (15, 41.32851, 69.35432, 0), (19, 41.32851, 69.35430, 0),
    (25, 41.32851, 69.35430, 0), (26, 41.32851, 69.35430, 3), (27, 41.32851, 69.35430, 6), (28, 41.32853, 69.35433, 9),
    (29, 41.32857, 69.35436, 16), (30, 41.32862, 69.35441, 22), (31, 41.32866, 69.35445, 27), (32, 41.32872, 69.35451, 29),
    (33, 41.32878, 69.35457, 24), (34, 41.32883, 69.35462, 17), (35, 41.32886, 69.35463, 9), (36, 41.32887, 69.35465, 5),
]

# Camera model.  Horizon and lane geometry measured on a frame where the car stands still (t = 10 s): the lane lines
# meet at row 426 (column 663); with a 3.5 m lane the camera height is 1.38 m.  Focal length 1350 px: at the moment
# of impact the car ahead is in the near carriageway of Do'rmon yo'li, about 40 m ahead of our car's GPS position on
# the OpenStreetMap plan of the crossing; its apparent width (61 px for a 1.8 m car) gives the same distance.
CALIBRATION = {"image_size": [1280, 720], "horizon_y": 426.0, "vanishing_x": 663.0, "focal_px": 1350.0,
               "camera_height_m": 1.38, "lane_width_m": 3.5, "uncertainty": "distances +-20%",
               "focal_from": "distance to the crossing on the OpenStreetMap plan at impact, checked with the car's apparent width"}

# Manual annotations (1280x720 pixels) of the first motorcycle, from the frames in which it is visible from this
# car.  Frames 706-708 it passes behind a pedestrian; frames 710 and 711 are duplicates in the source file.
MOTORCYCLE_1 = [
    (705, [247, 410, 282, 437]), (709, [440, 428, 486, 458]), (711, [488, 428, 530, 460]),
    (712, [527, 428, 575, 468]), (713, [581, 424, 627, 473]),
]
IMPACT_FRAME = 714  # the motorcycle strikes the left side of the car ahead
LIGHT_FRAME = 732  # the signal ahead is visible (green for this direction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="demo/incident")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    det = YoloOnnxDetector(str(MODELS_DIR / "yolo11n_coco.onnx"), conf=0.1)
    cap = cv2.VideoCapture(args.video)
    frames = []
    for k, t in enumerate(np.arange(T0, T1 + 1e-6, STEP)):
        fi = int(round(t * FPS_SRC))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            break
        safe = img.copy()
        for g in range(fi - 2, fi + 3):  # union of detections over neighbouring frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, g)
            ok2, nb = cap.read()
            if not ok2:
                continue
            d = det(nb)
            for i in range(len(d)):
                x1, y1, x2, y2 = d.xyxy[i]
                w, h = x2 - x1, y2 - y1
                if d.label(i) == "person":  # face: top of the person box
                    _blur(safe, x1 - 0.12 * w, y1 - 0.06 * h, x2 + 0.12 * w, y1 + 0.3 * h)
                elif d.label(i) in ("car", "truck", "bus"):  # plate band, full width (a box cut by the frame edge hides where the plate is)
                    _blur(safe, x1, y1 + 0.55 * h, x2, y2 + 0.04 * h)
        s = OUT_W / img.shape[1]
        name = f"f{fi:04d}.jpg"
        cv2.imwrite(str(out / "frames" / name), cv2.resize(safe, (OUT_W, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        frames.append({"t": round(float(t), 3), "src_frame": fi, "img": name})
    scene = {
        "name": "Tashkent crossing, 16 July 2026, 18:28",
        "source": "Dashcam footage recorded and provided by the SwarmSight team; people and plates blurred by us",
        "clip_seconds": [T0, T1], "fps": 1 / STEP, "frames": frames, "frame_scale": OUT_W / 1280,
        "telemetry": [{"t": t, "lat": la, "lon": lo, "kmh": v} for t, la, lo, v in TELEMETRY],
        "telemetry_note": "read by us from the dashcam overlay (GPS updates once a second, 5 decimals, about 1 m)",
        "calibration": CALIBRATION,
        "annotations": {"motorcycle_1": [{"t": round(fi / FPS_SRC, 3), "src_frame": fi, "box": b} for fi, b in MOTORCYCLE_1],
                        "note": "annotated by us frame by frame; the road-user detector does not find it (25-50 px, motion blur)"},
        "events": {"impact_t": round(IMPACT_FRAME / FPS_SRC, 3), "signal_visible_t": round(LIGHT_FRAME / FPS_SRC, 3)},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    print("frames", len(frames), "->", out)


if __name__ == "__main__":
    main()
