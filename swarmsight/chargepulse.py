"""ChargePulse: visual occupancy of EV charging bays.

Bays are polygons drawn once per camera.  A bay is visually occupied when a vehicle's ground
contact region (the lower part of its box) covers enough of the bay.  States change only after
`hold` consecutive agreeing samples, so a car driving through does not flip the bay.

This reports *visual occupancy* only.  Whether a car is actually charging, or a connector is
faulty, needs operator telemetry (e.g. OCPP) - the API keeps the two strictly apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detections

VEHICLE = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Bay:
    bay_id: str
    polygon: np.ndarray  # (N,2) image px
    state: str = "unknown"
    since: float = 0.0
    pending: str | None = None
    pending_n: int = 0
    vehicle_cls: str | None = None
    timeline: list = field(default_factory=list)  # (t_start, t_end, state)


def _poly_mask(poly: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m


class ChargePulse:
    def __init__(self, bays: dict[str, list], img_w: int, img_h: int, cover_thr: float = 0.25, hold: int = 3):
        self.bays = {k: Bay(k, np.asarray(v, float)) for k, v in bays.items()}
        self.shape = (img_h, img_w)
        self.masks = {k: _poly_mask(b.polygon, self.shape) for k, b in self.bays.items()}
        self.areas = {k: max(1, int(m.sum())) for k, m in self.masks.items()}
        self.cover_thr = cover_thr
        self.hold = hold
        self.events: list[dict] = []

    def update(self, det: Detections, t: float) -> dict:
        occ_now: dict[str, tuple[float, str]] = {}
        for i in range(len(det)):
            cls = det.label(i)
            if cls not in VEHICLE:
                continue
            x1, y1, x2, y2 = det.xyxy[i]
            # ground-contact region: bottom 40% of the box
            gy1 = y1 + 0.6 * (y2 - y1)
            box = np.zeros(self.shape, np.uint8)
            box[int(max(0, gy1)):int(min(self.shape[0], y2)), int(max(0, x1)):int(min(self.shape[1], x2))] = 1
            for k, m in self.masks.items():
                cover = float((box & m).sum()) / self.areas[k]
                if cover > occ_now.get(k, (0.0, ""))[0]:
                    occ_now[k] = (cover, cls)
        out = {}
        for k, b in self.bays.items():
            cover, cls = occ_now.get(k, (0.0, None))
            obs = "occupied" if cover >= self.cover_thr else "free"
            if b.state == "unknown":
                b.state, b.since, b.vehicle_cls = obs, t, cls
                b.timeline.append([t, t, obs])
                self.events.append({"t": t, "bay": k, "state": obs, "vehicle_cls": cls, "initial": True})
            elif obs != b.state:
                if b.pending == obs:
                    b.pending_n += 1
                else:
                    b.pending, b.pending_n = obs, 1
                if b.pending_n >= self.hold:
                    b.state, b.since, b.vehicle_cls = obs, t, cls
                    b.pending, b.pending_n = None, 0
                    b.timeline.append([t, t, obs])
                    self.events.append({"t": t, "bay": k, "state": obs, "vehicle_cls": cls, "initial": False})
            else:
                b.pending, b.pending_n = None, 0
            b.timeline[-1][1] = t
            out[k] = {"state": b.state, "since": b.since, "cover": round(cover, 2), "vehicle_cls": b.vehicle_cls}
        return out

    def summary(self) -> dict:
        res = {}
        for k, b in self.bays.items():
            total = sum(e - s for s, e, _ in b.timeline) or 1e-9
            occ = sum(e - s for s, e, st in b.timeline if st == "occupied")
            res[k] = {"state": b.state, "since": b.since, "utilisation": round(occ / total, 3),
                      "timeline": [[round(s, 2), round(e, 2), st] for s, e, st in b.timeline]}
        return res
