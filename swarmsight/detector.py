"""YOLO object detection with ONNX Runtime.

The models are trained/exported with Ultralytics (AGPL-3.0) and executed here with plain
ONNX Runtime + NumPy, so the deployed service needs no PyTorch.  Output tensor layout of an
Ultralytics YOLOv8/YOLO11 detection export is (1, 4 + num_classes, num_anchors) with boxes in
(cx, cy, w, h) input-pixel units and per-class sigmoid scores.
"""
from __future__ import annotations

import ast
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4) float32, original image pixels
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int32
    names: dict[int, str] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.conf)

    def subset(self, mask: np.ndarray) -> "Detections":
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask], self.names, self.timing_ms)

    def label(self, i: int) -> str:
        return self.names.get(int(self.cls[i]), str(int(self.cls[i])))

    def to_list(self) -> list[dict]:
        return [
            {
                "bbox": [round(float(v), 1) for v in self.xyxy[i]],
                "conf": round(float(self.conf[i]), 3),
                "cls": self.label(i),
            }
            for i in range(len(self))
        ]

    @staticmethod
    def empty(names=None) -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), names or {})


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square `size` (Ultralytics convention, pad=114)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain NumPy greedy NMS. boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


class YoloOnnxDetector:
    """Ultralytics-exported YOLO detection model executed with ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        classes: list[str] | None = None,
        threads: int | None = None,
        max_det: int = 300,
    ):
        self.model_path = str(model_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        n = threads or int(os.environ.get("SWARMSIGHT_THREADS", "0") or 0)
        if n > 0:
            so.intra_op_num_threads = n
        self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()} if "names" in meta else {}
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.class_filter = None
        if classes:
            wanted = set(classes)
            self.class_filter = np.array([i for i, n in self.names.items() if n in wanted], dtype=np.int32)

    @property
    def name(self) -> str:
        return Path(self.model_path).stem

    def __call__(self, img_bgr: np.ndarray, conf: float | None = None) -> Detections:
        conf_thr = self.conf if conf is None else conf
        t0 = time.perf_counter()
        lb, r, (padx, pady) = letterbox(img_bgr, self.imgsz)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4+nc, A)
        t2 = time.perf_counter()
        pred = out.T
        scores_all = pred[:, 4:]
        if self.class_filter is not None:
            mask_cls = np.full(scores_all.shape[1], False)
            mask_cls[self.class_filter] = True
            scores_all = np.where(mask_cls[None, :], scores_all, 0.0)
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores >= conf_thr
        pred, cls, scores = pred[keep], cls[keep], scores[keep]
        if len(scores):
            cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padx) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pady) / r
            H, W = img_bgr.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
            # class-aware NMS via coordinate offset trick
            offs = cls[:, None].astype(np.float32) * 7680.0
            idx = nms(boxes + offs, scores, self.iou)[: self.max_det]
            det = Detections(boxes[idx].astype(np.float32), scores[idx].astype(np.float32), cls[idx].astype(np.int32), self.names)
        else:
            det = Detections.empty(self.names)
        t3 = time.perf_counter()
        det.timing_ms = {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}
        return det


ROAD_USER_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]
VRU_CLASSES = {"person", "bicycle", "motorcycle"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}

MODELS_DIR = Path(os.environ.get("SWARMSIGHT_MODELS", Path(__file__).resolve().parent.parent / "models"))


def load_default_detectors(threads: int | None = None) -> dict[str, YoloOnnxDetector]:
    """Load the road-user (COCO) and road-damage (RDD2022) detectors if their weights are present."""
    dets: dict[str, YoloOnnxDetector] = {}
    coco = MODELS_DIR / "yolo11n_coco.onnx"
    if coco.exists():
        dets["road_users"] = YoloOnnxDetector(coco, conf=0.3, classes=ROAD_USER_CLASSES, threads=threads)
    dmg = MODELS_DIR / "road_damage_yolo11n.onnx"
    if dmg.exists():
        dets["road_damage"] = YoloOnnxDetector(dmg, conf=0.25, threads=threads)
    return dets
