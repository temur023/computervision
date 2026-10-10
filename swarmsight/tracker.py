"""ByteTrack-style multi-object tracker (Zhang et al., 2022), re-implemented in NumPy/SciPy.

Two-stage association: confident detections are matched to all live tracks first, then
low-confidence detections rescue tracks that would otherwise be lost (typical under partial
occlusion).  Matching is class-aware and uses IoU on Kalman-predicted boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detections


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) as in SORT/DeepSORT/ByteTrack."""

    _std_pos = 1.0 / 20
    _std_vel = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.Hm = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        S = self.Hm @ cov @ self.Hm.T + R
        K = cov @ self.Hm.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.Hm @ mean)
        cov = (np.eye(8) - K @ self.Hm) @ cov
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod((br - tl).clip(0), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    cls: int
    mean: np.ndarray
    cov: np.ndarray
    conf: float
    start_frame: int
    last_frame: int
    hits: int = 1
    state: str = "tentative"  # tentative | tracked | lost | removed
    history: list = field(default_factory=list)  # (t, xyxy) for speed / TTC estimation

    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


class ByteTracker:
    def __init__(self, high_thr=0.5, low_thr=0.1, new_track_thr=0.55, match_thr=0.8, track_buffer=30,
                 min_hits=2, history_len=30):
        self.high_thr, self.low_thr, self.new_track_thr = high_thr, low_thr, new_track_thr
        self.match_thr = match_thr  # max cost (1 - IoU) accepted
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.history_len = history_len
        self.kf = KalmanXYAH()
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1

    def _match(self, tracks, boxes, classes, scores, thr):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = np.array([t.xyxy for t in tracks])
        iou = iou_matrix(tb, boxes)
        cost = 1 - iou * scores[None, :]  # ByteTrack "fuse score"
        tc = np.array([t.cls for t in tracks])
        cost[tc[:, None] != classes[None, :]] = 1e6
        r, c = linear_sum_assignment(cost)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(boxes)))
        for i, j in zip(r, c):
            if cost[i, j] <= thr:
                matches.append((i, j))
                ut.discard(i)
                ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def update(self, det: Detections, t: float | None = None) -> list[Track]:
        """Advance one frame. Returns confirmed tracks updated in this frame."""
        self.frame += 1
        t = float(self.frame) if t is None else t
        for tr in self.tracks:
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov)
        boxes, scores, classes = det.xyxy.astype(np.float64), det.conf.astype(np.float64), det.cls
        hi = scores >= self.high_thr
        lo = (scores >= self.low_thr) & ~hi
        active = [tr for tr in self.tracks if tr.state in ("tracked", "tentative", "lost")]

        # stage 1: high-confidence detections vs all active tracks
        hi_idx = np.where(hi)[0]
        m1, ut1, ud1 = self._match(active, boxes[hi_idx], classes[hi_idx], scores[hi_idx], self.match_thr)
        updated = []
        for i, j in m1:
            self._apply(active[i], boxes[hi_idx[j]], scores[hi_idx[j]], t)
            updated.append(active[i])
        # stage 2: low-confidence detections vs remaining tracked tracks (not tentative)
        rem = [active[i] for i in ut1 if active[i].state in ("tracked", "lost")]
        lo_idx = np.where(lo)[0]
        m2, ut2, _ = self._match(rem, boxes[lo_idx], classes[lo_idx], np.ones(len(lo_idx)), 0.5)
        for i, j in m2:
            self._apply(rem[i], boxes[lo_idx[j]], scores[lo_idx[j]], t)
            updated.append(rem[i])
        # unmatched tracks
        upd_ids = {id(x) for x in updated}
        for tr in active:
            if id(tr) in upd_ids:
                continue
            if tr.state == "tentative":
                tr.state = "removed"
            elif self.frame - tr.last_frame > self.track_buffer:
                tr.state = "removed"
            else:
                tr.state = "lost"
        # new tracks from unmatched confident detections
        for j in ud1:
            k = hi_idx[j]
            if scores[k] < self.new_track_thr:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(boxes[k]))
            tr = Track(self._next_id, int(classes[k]), mean, cov, float(scores[k]), self.frame, self.frame)
            tr.history.append((t, boxes[k].copy()))
            if self.min_hits <= 1:
                tr.state = "tracked"
            self._next_id += 1
            self.tracks.append(tr)
            if tr.state == "tracked":
                updated.append(tr)
        self.tracks = [tr for tr in self.tracks if tr.state != "removed"]
        return [tr for tr in updated if tr.state == "tracked"]

    def _apply(self, tr: Track, box, score, t):
        tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, xyxy_to_xyah(box))
        tr.conf = float(score)
        tr.last_frame = self.frame
        tr.hits += 1
        if tr.state in ("tentative", "lost") and tr.hits >= self.min_hits:
            tr.state = "tracked"
        tr.history.append((t, np.asarray(box, dtype=np.float64).copy()))
        if len(tr.history) > self.history_len:
            tr.history.pop(0)


def time_to_contact(track: Track, min_span: float = 0.4) -> float | None:
    """Monocular time-to-contact from bounding-box scale expansion: TTC = s / (ds/dt).

    Uses a least-squares fit of log(height) over the track history; no calibration needed.
    Returns seconds, or None if the object is not approaching or history is too short.
    """
    if len(track.history) < 4:
        return None
    ts = np.array([h[0] for h in track.history])
    if ts[-1] - ts[0] < min_span:
        return None
    hs = np.array([max(h[1][3] - h[1][1], 1.0) for h in track.history])
    ws = np.array([max(h[1][2] - h[1][0], 1.0) for h in track.history])
    s = np.log(np.sqrt(hs * ws))
    A = np.vstack([ts - ts[-1], np.ones_like(ts)]).T
    (rate, _), *_ = np.linalg.lstsq(A, s, rcond=None)  # d log(s)/dt
    if rate <= 1e-3:
        return None
    return float(1.0 / rate)
