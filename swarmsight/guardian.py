"""Guardian: cooperative perception across independent cameras.

Pipeline per incoming frame of camera c:
  detect -> compose riders -> track (ByteTrack) -> ground projection with covariance
  -> associate to global objects (sticky bindings + gated Hungarian on Mahalanobis distance)
  -> Kalman fusion in the ground plane -> per-subscriber visibility reasoning:
       visible | occluded (by which object) | undetected | out_of_view
  -> external-observation messages with source, age, TTL and uncertainty radius.

The output is advisory perception sharing for research and pilots, not a braking or steering
signal (see README, "Safety position").
"""
from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .conflict import Agent, ConflictMonitor, assess_all, assess_for
from .detector import Detections
from .geometry import CameraGeometry, mahalanobis2, uncertainty_radius
from .tracker import ByteTracker

GROUP = {
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "cyclist": "two_wheeler",
    "motorcyclist": "two_wheeler",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
}
GROUP_IDS = {"person": 0, "two_wheeler": 1, "vehicle": 2}
VRU = {"person", "cyclist", "motorcyclist", "bicycle", "motorcycle"}
SHAPE_CLASS = {"cyclist": "bicycle", "motorcyclist": "motorcycle"}  # for ghost boxes / footprint
MIN_GATE_M = {"person": 0.8, "two_wheeler": 1.2, "vehicle": 2.0}
CLASS_PRIORITY = {"person": 1.0, "cyclist": 0.95, "motorcyclist": 0.95, "bicycle": 0.6, "motorcycle": 0.7,
                  "car": 0.4, "truck": 0.4, "bus": 0.4}


def compose_road_users(det: Detections, min_conf: float = 0.25) -> list[tuple[np.ndarray, str, float]]:
    """Merge person + bicycle/motorcycle boxes into a single rider object.

    COCO detectors report a cyclist as two boxes; for collective perception a rider is one road
    user whose ground contact is the wheel, not the feet.
    """
    items = [(det.xyxy[i].astype(float), det.label(i), float(det.conf[i])) for i in range(len(det)) if det.conf[i] >= min_conf]
    items = [it for it in items if it[1] in GROUP]
    persons = [i for i, it in enumerate(items) if it[1] == "person"]
    bikes = [i for i, it in enumerate(items) if it[1] in ("bicycle", "motorcycle")]
    used, out = set(), []
    for b in bikes:
        bb = items[b][0]
        best, best_score = None, 0.0
        for p in persons:
            if p in used:
                continue
            pb = items[p][0]
            ix = max(0, min(pb[2], bb[2]) - max(pb[0], bb[0]))
            overlap_x = ix / max(1.0, min(pb[2] - pb[0], bb[2] - bb[0]))
            feet_in = bb[1] - 0.1 * (bb[3] - bb[1]) <= pb[3] <= bb[3] + 0.1 * (bb[3] - bb[1])
            if overlap_x > 0.5 and feet_in and pb[1] < bb[1]:
                score = overlap_x
                if score > best_score:
                    best, best_score = p, score
        if best is not None:
            pb = items[best][0]
            union = np.array([min(pb[0], bb[0]), min(pb[1], bb[1]), max(pb[2], bb[2]), bb[3]])
            label = "cyclist" if items[b][1] == "bicycle" else "motorcyclist"
            out.append((union, label, max(items[b][2], items[best][2])))
            used.update({b, best})
    for i, it in enumerate(items):
        if i not in used:
            out.append(it)
    # suppress duplicate vehicle boxes of different fine classes (car vs truck on the same object)
    out.sort(key=lambda x: -x[2])
    keep: list[tuple[np.ndarray, str, float]] = []
    for it in out:
        dup = False
        for k in keep:
            if GROUP[k[1]] == GROUP[it[1]] and _iou(k[0], it[0]) > 0.7:
                dup = True
                break
        if not dup:
            keep.append(it)
    return keep


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _coverage(box, by) -> float:
    """Fraction of `box` covered by `by`."""
    ix = max(0.0, min(box[2], by[2]) - max(box[0], by[0]))
    iy = max(0.0, min(box[3], by[3]) - max(box[1], by[1]))
    return ix * iy / max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))


@dataclass
class Observation:
    cam: str
    local_id: int
    cls: str
    conf: float
    bbox: list
    xy: np.ndarray
    cov: np.ndarray
    t: float

    @property
    def group(self) -> str:
        return GROUP.get(self.cls, self.cls)


class CameraNode:
    """Per-camera perception: rider composition, tracking, ground projection."""

    def __init__(self, cam_id: str, geometry: CameraGeometry | None, track_buffer: int = 15, mount: str = "fixed"):
        self.cam_id = cam_id
        self.geometry = geometry
        self.mount = mount  # "vehicle": a car's own camera (CrossGuard for that car); "fixed": road-side camera
        self.tracker = ByteTracker(high_thr=0.45, low_thr=0.15, new_track_thr=0.5, track_buffer=track_buffer, min_hits=1)
        self.labels: dict[int, Counter] = {}
        self.last_tracks: list[dict] = []
        self.last_t = -1.0
        self.frames = 0

    def process(self, det: Detections, t: float, geometry: CameraGeometry | None = None) -> list[Observation]:
        if geometry is not None:
            self.geometry = geometry  # moving camera: per-frame pose
        users = compose_road_users(det)
        group_names = {v: k for k, v in GROUP_IDS.items()}
        fine = [u[1] for u in users]
        d = Detections(
            np.array([u[0] for u in users], np.float32).reshape(-1, 4),
            np.array([u[2] for u in users], np.float32),
            np.array([GROUP_IDS[GROUP[u[1]]] for u in users], np.int32),
            group_names,
        )
        tracks = self.tracker.update(d, t)
        # vote the fine label of each track from the detection it was matched to (highest IoU)
        obs, self.last_tracks = [], []
        for tr in tracks:
            box = tr.history[-1][1]
            best = max(range(len(users)), key=lambda i: _iou(users[i][0], box), default=None)
            if best is not None and _iou(users[best][0], box) > 0.5:
                self.labels.setdefault(tr.track_id, Counter())[fine[best]] += users[best][2]
            label = self.labels.get(tr.track_id, Counter({group_names[tr.cls]: 1})).most_common(1)[0][0]
            rec = {"id": tr.track_id, "cls": label, "conf": round(tr.conf, 3), "bbox": [round(float(v), 1) for v in box]}
            self.last_tracks.append(rec)
            if self.geometry is None:
                continue
            gp = self.geometry.ground_point(box, SHAPE_CLASS.get(label, label))
            if gp is None:
                continue
            xy, cov = gp
            rec["xy"] = [round(float(xy[0]), 2), round(float(xy[1]), 2)]
            obs.append(Observation(self.cam_id, tr.track_id, label, tr.conf, rec["bbox"], xy, cov, t))
        live = {tr.track_id for tr in self.tracker.tracks}
        self.labels = {k: v for k, v in self.labels.items() if k in live}
        self.last_t = t
        self.frames += 1
        return obs


@dataclass
class GlobalObject:
    gid: int
    group: str
    x: np.ndarray  # (4,) x, y, vx, vy
    P: np.ndarray  # (4,4)
    t: float  # time of last state update
    created: float
    labels: Counter = field(default_factory=Counter)
    members: dict = field(default_factory=dict)  # cam -> local track id
    seen: dict = field(default_factory=dict)  # cam -> last time observed
    conf: float = 0.0
    trail: list = field(default_factory=list)
    hist: list = field(default_factory=list)  # [(t, x, y, vx, vy)] for motion prediction
    n_updates: int = 0

    @property
    def cls(self) -> str:
        return self.labels.most_common(1)[0][0] if self.labels else self.group

    @property
    def xy(self) -> np.ndarray:
        return self.x[:2]

    @property
    def cov(self) -> np.ndarray:
        return self.P[:2, :2]


class CooperativeFusion:
    """Track-level fusion of observations from many cameras in the shared ground plane."""

    def __init__(self, ttl: float = 1.0, gate: float = 9.21, sticky_gate: float = 18.4, accel_std: float = 2.0):
        self.ttl = ttl
        self.gate = gate  # chi-square 99% (2 dof)
        self.sticky_gate = sticky_gate
        self.accel_std = accel_std
        self.objects: dict[int, GlobalObject] = {}
        self.bindings: dict[tuple[str, int], int] = {}
        self._ids = itertools.count(1)

    def _predict(self, g: GlobalObject, t: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(0.0, t - g.t)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.accel_std**2
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        return F @ g.x, F @ g.P @ F.T + G @ G.T * q

    def _d2(self, g: GlobalObject, o: Observation) -> float:
        x, P = self._predict(g, o.t)
        S = P[:2, :2] + o.cov + np.eye(2) * MIN_GATE_M.get(o.group, 1.0) ** 2 / 9.0
        return mahalanobis2(o.xy - x[:2], S)

    def _update(self, g: GlobalObject, o: Observation):
        x, P = self._predict(g, o.t)
        Hm = np.eye(2, 4)
        S = Hm @ P @ Hm.T + o.cov
        K = P @ Hm.T @ np.linalg.inv(S)
        g.x = x + K @ (o.xy - Hm @ x)
        g.P = (np.eye(4) - K @ Hm) @ P
        g.t = max(g.t, o.t)
        g.labels[o.cls] += o.conf
        g.members[o.cam] = o.local_id
        g.seen[o.cam] = o.t
        g.conf = max(g.conf * 0.9, o.conf)
        g.n_updates += 1
        self.bindings[(o.cam, o.local_id)] = g.gid

    def _new(self, o: Observation) -> GlobalObject:
        P = np.zeros((4, 4))
        P[:2, :2] = o.cov
        P[2:, 2:] = np.eye(2) * 4.0
        g = GlobalObject(next(self._ids), o.group, np.r_[o.xy, 0.0, 0.0], P, o.t, o.t)
        self.objects[g.gid] = g
        self._update(g, o)
        return g

    def _merge(self, t: float):
        objs = sorted(self.objects.values(), key=lambda g: g.created)
        removed = set()
        for i, a in enumerate(objs):
            if a.gid in removed:
                continue
            for b in objs[i + 1:]:
                if b.gid in removed or b.group != a.group:
                    continue
                # never merge two objects one camera currently sees as distinct tracks
                shared = [c for c in set(a.seen) & set(b.seen) if t - a.seen[c] < 0.5 and t - b.seen[c] < 0.5]
                if shared:
                    continue
                xa, Pa = self._predict(a, t)
                xb, Pb = self._predict(b, t)
                S = Pa[:2, :2] + Pb[:2, :2] + np.eye(2) * MIN_GATE_M.get(a.group, 1.0) ** 2 / 9.0
                if mahalanobis2(xa[:2] - xb[:2], S) > self.gate:
                    continue
                # information-weighted combination of the two state estimates
                Ia, Ib = np.linalg.inv(Pa + np.eye(4) * 1e-6), np.linalg.inv(Pb + np.eye(4) * 1e-6)
                a.P = np.linalg.inv(Ia + Ib)
                a.x = a.P @ (Ia @ xa + Ib @ xb)
                a.t = max(a.t, b.t)
                a.labels.update(b.labels)
                a.members.update(b.members)
                a.seen.update(b.seen)
                a.conf = max(a.conf, b.conf)
                a.n_updates += b.n_updates
                for k, v in list(self.bindings.items()):
                    if v == b.gid:
                        self.bindings[k] = a.gid
                removed.add(b.gid)
        for gid in removed:
            self.objects.pop(gid, None)

    def step(self, obs: list[Observation], t: float) -> list[GlobalObject]:
        assigned: dict[int, set] = {}  # gid -> cams assigned in this step
        pending: list[Observation] = []
        # 1) sticky bindings from earlier frames
        for o in obs:
            gid = self.bindings.get((o.cam, o.local_id))
            g = self.objects.get(gid) if gid is not None else None
            if g is not None and g.group == o.group and o.cam not in assigned.get(gid, set()) and self._d2(g, o) < self.sticky_gate:
                self._update(g, o)
                assigned.setdefault(gid, set()).add(o.cam)
            else:
                self.bindings.pop((o.cam, o.local_id), None)
                pending.append(o)
        # 2) gated Hungarian per camera; objects created by earlier cameras are candidates too
        by_cam: dict[str, list[Observation]] = {}
        for o in pending:
            by_cam.setdefault(o.cam, []).append(o)
        for cam, olist in by_cam.items():
            cands = [g for g in self.objects.values() if cam not in assigned.get(g.gid, set())
                     and (cam not in g.members or g.seen.get(cam, -1e9) < t - 0.5)]
            if cands:
                C = np.full((len(olist), len(cands)), 1e6)
                for i, o in enumerate(olist):
                    for j, g in enumerate(cands):
                        if g.group == o.group:
                            C[i, j] = self._d2(g, o)
                r, c = linear_sum_assignment(C)
                matched = set()
                for i, j in zip(r, c):
                    if C[i, j] < self.gate:
                        self._update(cands[j], olist[i])
                        assigned.setdefault(cands[j].gid, set()).add(cam)
                        matched.add(i)
                rest = [o for i, o in enumerate(olist) if i not in matched]
            else:
                rest = olist
            for o in rest:
                g = self._new(o)
                assigned.setdefault(g.gid, set()).add(cam)
        # 3) merge duplicates (objects first seen separately by different cameras)
        self._merge(t)
        # 4) expire
        for gid in [gid for gid, g in self.objects.items() if t - g.t > self.ttl]:
            g = self.objects.pop(gid)
            for k in [k for k, v in self.bindings.items() if v == gid]:
                self.bindings.pop(k)
        for g in self.objects.values():
            if g.gid in assigned:
                g.trail.append((round(float(g.x[0]), 2), round(float(g.x[1]), 2)))
                g.trail = g.trail[-25:]
                g.hist.append((t, float(g.x[0]), float(g.x[1]), float(g.x[2]), float(g.x[3])))
                g.hist = g.hist[-15:]
        return list(self.objects.values())


class GuardianScene:
    """Several camera nodes + fusion + per-subscriber reasoning."""

    def __init__(self, ttl: float = 1.0, alert_radius_m: float = 45.0, ego_offset_m: float = 1.5):
        self.nodes: dict[str, CameraNode] = {}
        self.fusion = CooperativeFusion(ttl=ttl)
        self.alert_radius_m = alert_radius_m
        self.ttl = ttl
        self.timing: dict[str, float] = {}
        self.ego_offset_m = ego_offset_m  # vehicle centre lies this far behind a forward-facing camera
        self.ego_hist: dict[str, list] = {}
        self.monitors: dict[str, ConflictMonitor] = {}

    def ego_agent(self, cam_id: str, t: float) -> Agent | None:
        """The subscriber's own vehicle: position and heading from its camera pose, velocity from the pose history."""
        geo = self.nodes[cam_id].geometry
        if geo is None or geo.R is None:
            return None
        fwd = geo.R.T @ np.array([0.0, 0.0, 1.0])
        hd = float(np.arctan2(fwd[1], fwd[0]))
        pos = geo.center_ground - self.ego_offset_m * np.array([np.cos(hd), np.sin(hd)])
        h = self.ego_hist.setdefault(cam_id, [])
        if not h or t > h[-1][0]:
            h.append((t, float(pos[0]), float(pos[1])))
            del h[:-12]
        v = np.zeros(2)
        hist = []
        for (t0, x0, y0), (t1, x1, y1) in zip(h, h[1:]):
            vi = np.array([x1 - x0, y1 - y0]) / max(1e-3, t1 - t0)
            hist.append((t1, x1, y1, float(vi[0]), float(vi[1])))
        recent = [r for r in hist if r[0] >= t - 0.6]
        if recent:
            v = np.mean([[r[3], r[4]] for r in recent], axis=0)
        return Agent("ego", "car", pos, v, 0.3, 0.3, hist, ego=True, heading=hd)

    def _agents(self, t: float, objects: list[dict], exclude_near=None) -> list[Agent]:
        by_gid = {o["gid"] for o in objects}
        out = []
        for g in self.fusion.objects.values():
            if g.gid not in by_gid:
                continue
            if exclude_near is not None and g.group == "vehicle" and np.linalg.norm(g.xy - exclude_near) < 3.0:
                continue  # our own car as the other cameras see it
            Pv = g.P[2:, 2:]
            out.append(Agent(str(g.gid), g.cls, g.xy.copy(), g.x[2:].copy(), float(np.sqrt(max(np.trace(g.cov) / 2, 0.04))),
                             float(np.sqrt(max(np.trace(Pv) / 2, 0.04))), list(g.hist), age_s=t - g.created))
        return out

    def conflicts_between(self, cam_id: str, t: float, objects: list[dict]) -> dict:
        """CrossGuard for a road-side camera: predicted conflicts between every pair of road users it knows about."""
        t0 = time.perf_counter()
        res = assess_all(self._agents(t, objects), seed=int(round(t * 10)))
        for r in res:
            r["gids"] = [int(x) for x in r["pair"]]
        alerts = self.monitors.setdefault(cam_id, ConflictMonitor()).update(res, t)
        self.timing["conflict_ms"] = (time.perf_counter() - t0) * 1e3
        return {"ego": None, "assessments": res, "alerts": alerts}

    def conflicts_for(self, cam_id: str, t: float, objects: list[dict]) -> dict:
        """CrossGuard for one subscriber: predicted conflicts between its vehicle and every fused road user."""
        ego = self.ego_agent(cam_id, t)
        if ego is None:
            return {"ego": None, "assessments": [], "alerts": []}
        by_gid = {o["gid"]: o for o in objects}
        others = self._agents(t, objects, exclude_near=ego.xy)
        t0 = time.perf_counter()
        res = assess_for(ego, others, seed=int(round(t * 10)))
        for r in res:
            o = by_gid.get(int(r["pair"][1]))
            r["gid"] = int(r["pair"][1])
            r["gids"] = [int(r["pair"][1])]
            r["status"] = o.get("status") if o else None
            if o and o.get("status") == "occluded" and o.get("occluded_by"):
                r["reason"] = r["reason"].replace(":", f", hidden behind a {o['occluded_by']['cls']}:", 1)
        mon = self.monitors.setdefault(cam_id, ConflictMonitor())
        alerts = mon.update(res, t)
        self.timing["conflict_ms"] = (time.perf_counter() - t0) * 1e3
        s = float(np.hypot(*ego.v))
        return {"ego": {"xy": [round(float(ego.xy[0]), 2), round(float(ego.xy[1]), 2)], "heading": round(ego.heading, 4),
                        "speed_kmh": round(s * 3.6, 1), "mode": "moving" if s >= 1.0 else "stopped"},
                "assessments": res, "alerts": alerts}

    def add_camera(self, cam_id: str, geometry: CameraGeometry | None, mount: str = "fixed"):
        self.nodes[cam_id] = CameraNode(cam_id, geometry, mount=mount)

    def ingest(self, cam_id: str, det: Detections, t: float, geometry: CameraGeometry | None = None) -> list[Observation]:
        t0 = time.perf_counter()
        obs = self.nodes[cam_id].process(det, t, geometry)
        t1 = time.perf_counter()
        self.fusion.step(obs, t)
        t2 = time.perf_counter()
        self.timing = {"track_project_ms": (t1 - t0) * 1e3, "fusion_ms": (t2 - t1) * 1e3}
        return obs

    def view_for(self, cam_id: str, t: float, recent: float = 0.35) -> dict:
        """What camera `cam_id` sees by itself, and what the network adds for it."""
        node = self.nodes[cam_id]
        geo = node.geometry
        own = node.last_tracks
        objects, alerts = [], []
        for g in self.fusion.objects.values():
            age = t - g.t
            if age > self.ttl:
                continue
            sources = sorted(c for c, ts in g.seen.items() if t - ts <= self.ttl)
            seen_here = cam_id in g.seen and t - g.seen[cam_id] <= recent
            rec = {
                "gid": g.gid,
                "cls": g.cls,
                "xy": [round(float(g.x[0]), 2), round(float(g.x[1]), 2)],
                "v": [round(float(g.x[2]), 2), round(float(g.x[3]), 2)],
                "r2s": round(uncertainty_radius(g.cov), 2),
                "sources": sources,
                "age_ms": round(age * 1e3),
                "conf": round(g.conf, 2),
                "trail": g.trail[-12:],
                "members": {c: int(lid) for c, lid in g.members.items() if c in sources},
            }
            if seen_here:
                rec["status"] = "visible"
                rec["local_id"] = g.members.get(cam_id)
            elif geo is None:
                rec["status"] = "uncalibrated"
            else:
                shape_cls = SHAPE_CLASS.get(g.cls, g.cls)
                dist = geo.ground_distance(g.xy)
                rec["dist_m"] = round(dist, 1)
                if geo.in_view(g.xy):
                    ghost = geo.ghost_box(g.xy, shape_cls)
                    rec["ghost_box"] = [round(v, 1) for v in ghost] if ghost else None
                    occ = None
                    # image-space check: this camera may track the object but its own ground
                    # estimate was too uncertain to associate (e.g. truncated at the image edge)
                    same = [tr for tr in own if GROUP.get(tr["cls"]) == g.group and ghost
                            and (_iou(ghost, tr["bbox"]) > 0.3 or _coverage(tr["bbox"], ghost) > 0.7)]
                    if same:
                        rec["status"] = "visible"
                        rec["local_id"] = same[0]["id"]
                        rec["match"] = "image"
                        objects.append(rec)
                        continue
                    if ghost:
                        best = 0.0
                        for tr in own:
                            cov = _coverage(ghost, tr["bbox"])
                            if cov < 0.3:
                                continue
                            gp = geo.ground_point(tr["bbox"], SHAPE_CLASS.get(tr["cls"], tr["cls"]))
                            nearer = gp is None or geo.ground_distance(gp[0]) < dist - 0.5
                            if nearer and cov > best:
                                best, occ = cov, tr
                    if occ is not None:
                        rec["status"] = "occluded"
                        rec["occluded_by"] = {"cls": occ["cls"], "id": occ["id"], "coverage": round(best, 2)}
                    else:
                        rec["status"] = "undetected"
                else:
                    rec["status"] = "out_of_view"
                if rec["status"] in ("occluded", "undetected") and dist <= self.alert_radius_m and sources:
                    pr = CLASS_PRIORITY.get(g.cls, 0.3) * (1 - dist / self.alert_radius_m) * (1.3 if rec["status"] == "occluded" else 1.0)
                    alerts.append({
                        "type": "external_observation",
                        "gid": g.gid,
                        "cls": g.cls,
                        "status": rec["status"],
                        "occluded_by": rec.get("occluded_by"),
                        "position_m": rec["xy"],
                        "uncertainty_m": rec["r2s"],
                        "distance_m": rec["dist_m"],
                        "sources": sources,
                        "age_ms": rec["age_ms"],
                        "ttl_ms": round(self.ttl * 1e3),
                        "priority": round(pr, 3),
                        "vru": g.cls in VRU,
                    })
            objects.append(rec)
        alerts.sort(key=lambda a: -a["priority"])
        if geo is None:
            cg = {"ego": None, "assessments": [], "alerts": []}
        elif node.mount == "vehicle":
            cg = self.conflicts_for(cam_id, t, objects)
        else:
            cg = self.conflicts_between(cam_id, t, objects)
        risk = {}
        for r in cg["assessments"]:
            for gid in r["gids"]:
                if risk.get(gid) != "high":
                    risk[gid] = r["level"]
        for o in objects:
            if o["gid"] in risk:
                o["risk"] = risk[o["gid"]]
        return {
            "camera": cam_id,
            "crossguard": cg,
            "geometry": geo.describe() if geo else None,
            "own_tracks": own,
            "objects": objects,
            "alerts": alerts,
            "timing": self.timing,
        }
