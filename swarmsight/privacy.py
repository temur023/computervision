"""Best-effort anonymisation of evidence snapshots before they are stored.

We do not ship a face or plate detector, so we blur conservatively from the road-user detector:
the head region of every person box and the lower-central band of every vehicle box (where
plates are mounted).  This is a data-minimisation measure, not a guarantee of anonymity (see
README, Privacy).  Production pilots should add dedicated face/plate detectors.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detections

VEHICLES = {"car", "bus", "truck", "motorcycle"}


def _blur(img: np.ndarray, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    k = max(3, (min(roi.shape[:2]) // 3) | 1)
    small = cv2.resize(roi, (max(1, roi.shape[1] // 8), max(1, roi.shape[0] // 8)), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST), (k, k), 0)


def anonymise(img: np.ndarray, det: Detections | None) -> np.ndarray:
    out = img.copy()
    if det is None:
        return out
    for i in range(len(det)):
        x1, y1, x2, y2 = det.xyxy[i]
        cls = det.label(i)
        w, h = x2 - x1, y2 - y1
        if cls == "person":
            _blur(out, x1 - 0.05 * w, y1 - 0.02 * h, x2 + 0.05 * w, y1 + 0.28 * h)
        elif cls in VEHICLES:
            _blur(out, x1 + 0.15 * w, y1 + 0.55 * h, x2 - 0.15 * w, y2)
    return out
