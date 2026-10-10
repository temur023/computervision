"""Blur every detected person and vehicle (whole box + margin) before images are published.

Mapillary blurs faces and plates, but not always: registration numbers painted on truck bodies and
some faces survive.  For images we redistribute (the Tashkent test gallery) we therefore blur whole
road users at a low detection threshold.  Labels are image-level road-surface labels, so blurring
vehicles and people does not change them.  Evaluation uses the original images, fetched locally.

Usage: python training/anonymize_dir.py --src data/tashkent/images --dst datasets/tashkent/images --ids-from datasets/tashkent/manifest.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.detector import MODELS_DIR, YoloOnnxDetector

BLUR = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def blur_box(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // 14), max(1, (y2 - y1) // 14)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR), (0, 0), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ids-from", required=True, help="manifest.json; only its test-split images are published")
    ap.add_argument("--conf", type=float, default=0.12)
    args = ap.parse_args()
    det = YoloOnnxDetector(MODELS_DIR / "yolo11n_coco.onnx", conf=args.conf, classes=sorted(BLUR))
    man = json.loads(Path(args.ids_from).read_text())
    ids = [m["id"] for m in man["images"] if m["split"] == "test"]
    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for i in ids:
        img = cv2.imread(str(Path(args.src) / f"{i}.jpg"))
        d = det(img)
        for b in d.xyxy:
            mx, my = 0.08 * (b[2] - b[0]), 0.08 * (b[3] - b[1])
            blur_box(img, b[0] - mx, b[1] - my, b[2] + mx, b[3] + my)
        n_boxes += len(d)
        cv2.imwrite(str(out / f"{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"anonymised {len(ids)} images, {n_boxes} road users blurred -> {out}")


if __name__ == "__main__":
    main()
