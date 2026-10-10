"""Single-camera analysis ("Scan"): any photo or video from a phone, dashcam or fixed camera.

Outputs, per frame: road users (tracked), road damage (tracked) and dashcam warnings; per clip:
a summary and road-event observations for the trust engine, with anonymised evidence snapshots.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .detector import Detections, YoloOnnxDetector
from .guardian import VRU, compose_road_users
from .privacy import anonymise
from .tracker import ByteTracker, time_to_contact
from .trust import RoadObservation, damage_severity, severity_label

MAX_SIDE = 1280
TTC_WARN_S = 3.0
DAMAGE_COLORS = {"pothole": (40, 40, 230), "alligator_crack": (0, 140, 255), "transverse_crack": (0, 200, 255),
                 "longitudinal_crack": (0, 230, 230)}


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _in_corridor(box, w, h) -> bool:
    """Ego path of a forward-facing dashcam: central band of the lower part of the frame."""
    cx = (box[0] + box[2]) / 2
    return 0.3 * w <= cx <= 0.7 * w and box[3] >= 0.45 * h


class ScanPipeline:
    def __init__(self, detectors: dict[str, YoloOnnxDetector], media_dir: str | Path = "runtime/media"):
        self.users = detectors.get("road_users")
        self.damage = detectors.get("road_damage")
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_version(self) -> str:
        return "+".join(d.name for d in (self.users, self.damage) if d)

    def detect(self, img: np.ndarray) -> tuple[Detections, Detections, dict]:
        t0 = time.perf_counter()
        u = self.users(img) if self.users else Detections.empty()
        t1 = time.perf_counter()
        d = self.damage(img) if self.damage else Detections.empty()
        t2 = time.perf_counter()
        return u, d, {"road_users_ms": round((t1 - t0) * 1e3, 1), "road_damage_ms": round((t2 - t1) * 1e3, 1)}

    def _save_evidence(self, img, users: Detections, box, label: str) -> str:
        safe = anonymise(img, users)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(safe, (x1, y1), (x2, y2), DAMAGE_COLORS.get(label, (0, 255, 0)), 3)
        cv2.putText(safe, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        name = f"{uuid.uuid4().hex[:16]}.jpg"
        h, w = safe.shape[:2]
        s = 960 / max(h, w)
        if s < 1:
            safe = cv2.resize(safe, (int(w * s), int(h * s)))
        cv2.imwrite(str(self.media_dir / name), safe, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return name

    # ------------------------------------------------------------------ image
    def analyze_image(self, img: np.ndarray) -> dict:
        img = _resize(img)
        h, w = img.shape[:2]
        u, d, timing = self.detect(img)
        users = [{"cls": c, "bbox": [round(float(v), 1) for v in b], "conf": round(cf, 3)} for b, c, cf in compose_road_users(u)]
        damage = []
        for i in range(len(d)):
            cls = d.label(i)
            sev = damage_severity(cls, d.xyxy[i], w, h, float(d.conf[i]))
            damage.append({"cls": cls, "bbox": [round(float(v), 1) for v in d.xyxy[i]], "conf": round(float(d.conf[i]), 3),
                           "severity": sev, "severity_label": severity_label(sev)})
        observations = self._damage_observations(img, u, [(x["cls"], x["bbox"], x["conf"], x["severity"]) for x in damage])
        return {"kind": "image", "width": w, "height": h, "road_users": users, "road_damage": damage,
                "timing_ms": timing, "observations": observations, "decisions": self._decisions(damage, [], users)}

    def _damage_observations(self, img, users: Detections, items) -> list[dict]:
        """One observation per damage type per submission (best instance as evidence)."""
        by_type = defaultdict(list)
        for cls, box, conf, sev in items:
            by_type[cls].append((conf * (0.5 + sev), cls, box, conf, sev))
        obs = []
        for cls, lst in by_type.items():
            lst.sort(key=lambda x: -x[0])
            _, _, box, conf, sev = lst[0]
            media = self._save_evidence(img, users, box, cls)
            obs.append({"type": cls, "conf": round(float(conf), 3), "severity": float(max(x[4] for x in lst)),
                        "count": len(lst), "bbox": box, "media": media})
        return obs

    @staticmethod
    def _decisions(damage, warnings, users) -> list[dict]:
        out = []
        for t in sorted({x["cls"] for x in damage}):
            items = [x for x in damage if x["cls"] == t]
            sev = max(x["severity"] for x in items)
            act = "dispatch inspection" if sev >= 0.6 else "add to maintenance backlog" if sev >= 0.35 else "monitor"
            out.append({"kind": "road_damage", "type": t, "count": len(items), "severity": severity_label(sev),
                        "action": f"Create road event candidate -> {act}"})
        for w in warnings:
            out.append({**w, "kind": "driver_warning", "warning": w["kind"]})
        vru = [u for u in users if u["cls"] in VRU]
        if vru:
            out.append({"kind": "vru_presence", "count": len(vru), "action": "Vulnerable road users in view: reduce speed"})
        return out

    # ------------------------------------------------------------------ video
    def analyze_video(self, path: str | Path, sample_fps: float = 5.0, max_seconds: float = 60.0, on_frame=None,
                      should_stop=None) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("cannot decode video (supported: mp4/mov/webm/avi with H.264/H.265/VP8/VP9)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 < fps < 240.0):
            fps = 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / sample_fps)))
        max_frames = int(max_seconds * fps)
        utrack = ByteTracker(high_thr=0.45, low_thr=0.15, new_track_thr=0.5, track_buffer=10, min_hits=2)
        dtrack = ByteTracker(high_thr=0.35, low_thr=0.15, new_track_thr=0.35, match_thr=0.95, track_buffer=4, min_hits=2)
        group = {"person": 0, "cyclist": 1, "motorcyclist": 1, "bicycle": 1, "motorcycle": 1, "car": 2, "truck": 2, "bus": 2}
        gnames = {0: "person", 1: "two_wheeler", 2: "vehicle"}
        labels: dict[int, Counter] = defaultdict(Counter)
        damage_tracks: dict[int, tuple[str, float]] = {}  # track -> (class, max severity)
        best_by_type: dict[str, tuple] = {}  # class -> best evidence (keeps one frame per class in memory)
        warnings, frames, lat = [], [], []
        idx = 0
        w = h = None
        t_wall = time.perf_counter()
        while idx < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            if should_stop and should_stop():
                break
            t = idx / fps
            img = _resize(img)
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            u, d, timing = self.detect(img)
            users = compose_road_users(u)
            ud = Detections(np.array([x[0] for x in users], np.float32).reshape(-1, 4), np.array([x[2] for x in users], np.float32),
                            np.array([group.get(x[1], 2) for x in users], np.int32), gnames)
            utr = utrack.update(ud, t)
            frame_users = []
            for tr in utr:
                box = tr.history[-1][1]
                best = max(range(len(users)), key=lambda i: _iou(users[i][0], box), default=None)
                if best is not None:
                    labels[tr.track_id][users[best][1]] += 1
                cls = labels[tr.track_id].most_common(1)[0][0] if labels[tr.track_id] else gnames[tr.cls]
                rec = {"id": tr.track_id, "cls": cls, "bbox": [round(float(v), 1) for v in box], "conf": round(tr.conf, 2)}
                if _in_corridor(box, w, h):
                    ttc = time_to_contact(tr)
                    if ttc is not None and ttc < 10:
                        rec["ttc"] = round(ttc, 2)
                    vru_near = cls in VRU and (box[3] - box[1]) > 0.12 * h
                    if (ttc is not None and ttc < TTC_WARN_S) or vru_near:
                        kind = "vru_in_path" if vru_near else "forward_collision_risk"
                        if not any(x["track"] == tr.track_id and x["kind"] == kind for x in warnings):
                            warnings.append({"kind": kind, "t": round(t, 2), "track": tr.track_id, "cls": cls,
                                             "ttc_s": rec.get("ttc"),
                                             "action": "Alert driver / log risk event for fleet safety review"})
                        rec["warn"] = kind
                frame_users.append(rec)
            dtr = dtrack.update(d, t)
            frame_damage = []
            for tr in dtr:
                box = tr.history[-1][1]
                cls = d.names.get(tr.cls, str(tr.cls))
                sev = damage_severity(cls, box, w, h, tr.conf)
                frame_damage.append({"id": tr.track_id, "cls": cls, "bbox": [round(float(v), 1) for v in box],
                                     "conf": round(tr.conf, 2), "severity": sev})
                prev = damage_tracks.get(tr.track_id, (cls, 0.0))
                damage_tracks[tr.track_id] = (cls, max(prev[1], sev))
                score = tr.conf * (0.5 + sev)
                if cls not in best_by_type or score > best_by_type[cls][0]:
                    best_by_type[cls] = (score, box.copy(), tr.conf, sev, img.copy(), u)
            dt = (time.perf_counter() - t0) * 1e3
            lat.append(dt)
            fr = {"t": round(t, 3), "w": w, "h": h, "users": frame_users, "damage": frame_damage, "ms": round(dt, 1), **timing}
            frames.append(fr)
            if on_frame:
                on_frame(fr, idx, min(n_total, max_frames) if n_total else 0)
            idx += 1
        cap.release()
        if w is None:
            raise ValueError("no frames could be decoded")
        # damage: one observation per type with the best instance as evidence
        observations, damage_summary = [], []
        for cls, (_, box, conf, sev, img, u) in best_by_type.items():
            sevs = [s for c, s in damage_tracks.values() if c == cls]
            media = self._save_evidence(img, u, box, cls)
            msev = max(sevs)
            observations.append({"type": cls, "conf": round(float(conf), 3), "severity": float(msev), "count": len(sevs),
                                 "bbox": [round(float(v), 1) for v in box], "media": media})
            damage_summary.append({"cls": cls, "count": len(sevs), "severity": msev})
        user_counts = Counter(labels[k].most_common(1)[0][0] for k in labels if labels[k])
        elapsed = time.perf_counter() - t_wall
        dmg_for_decisions = [{"cls": x["cls"], "severity": x["severity"]} for x in damage_summary for _ in range(x["count"])]
        return {
            "kind": "video", "width": w, "height": h, "fps": fps, "sample_fps": fps / step, "frames": frames,
            "duration_s": round(idx / fps, 2), "road_user_tracks": dict(user_counts), "road_damage": damage_summary,
            "warnings": warnings, "observations": observations,
            "decisions": self._decisions(dmg_for_decisions, warnings, []),
            "perf": {"frames": len(frames), "wall_s": round(elapsed, 2), "ms_per_frame_p50": round(float(np.median(lat)), 1),
                     "ms_per_frame_p95": round(float(np.percentile(lat, 95)), 1)},
        }


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def to_road_observation(o: dict, source_id: str, lat=None, lon=None, gps_acc=None, observed_at=None, model_version="") -> RoadObservation:
    return RoadObservation(type=o["type"], conf=o["conf"], source_id=source_id, lat=lat, lon=lon, gps_accuracy_m=gps_acc,
                           severity=o["severity"], observed_at=observed_at, model_version=model_version, media=o.get("media"),
                           bbox=o.get("bbox"), extra={"count": o.get("count", 1)})
