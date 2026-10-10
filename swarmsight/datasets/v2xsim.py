"""Reader for V2X-Sim 2.0 (nuScenes-format, CARLA-rendered, Apache-2.0).

Li et al., "V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous
Driving", IEEE RA-L 2022.  We use the public `v2x_sim_2.0_mini` release: one Town03 intersection
(scene_5), 100 frames, a road-side unit (agent 0, four cameras at 7 m) and five vehicles
(agents 1-5, six cameras each), with RGB, semantic segmentation and depth per camera.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..geometry import CameraGeometry

FPS = 5.0  # V2X-Sim samples are 0.2 s apart

CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.emergency.police": "car",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "person",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
}
# CARLA semantic tags used by V2X-Sim segmentation maps
SEG_PEDESTRIAN, SEG_VEHICLE = 4, 10


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class GTObject:
    instance: str
    cls: str
    center: np.ndarray  # (3,) world
    size: np.ndarray  # (w, l, h) nuScenes order
    R: np.ndarray  # box orientation

    def corners(self) -> np.ndarray:
        w, l, h = self.size
        xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
        ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
        zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
        return (self.R @ np.vstack([xs, ys, zs])).T + self.center

    @property
    def ground_xy(self) -> np.ndarray:
        return self.center[:2].copy()


class V2XSim:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = self.root / "v2.0-mini"
        L = lambda name: json.loads((meta / f"{name}.json").read_text())
        cat = {c["token"]: c["name"] for c in L("category")}
        self.instance_cls = {i["token"]: cat[i["category_token"]] for i in L("instance")}
        sensors = {s["token"]: s["channel"] for s in L("sensor")}
        calib = {c["token"]: c for c in L("calibrated_sensor")}
        poses = {e["token"]: e for e in L("ego_pose")}
        self.samples = sorted(L("sample"), key=lambda s: s["timestamp"])
        self.sample_index = {s["token"]: i for i, s in enumerate(self.samples)}
        # sample_data indexed by (frame, channel)
        self.sd: dict[tuple[int, str], dict] = {}
        for d in L("sample_data"):
            cs = calib[d["calibrated_sensor_token"]]
            ch = sensors[cs["sensor_token"]]
            fi = self.sample_index.get(d["sample_token"])
            if fi is None:
                continue
            self.sd[(fi, ch)] = {"file": d["filename"], "calib": cs, "pose": poses[d["ego_pose_token"]],
                                 "w": d.get("width", 1600), "h": d.get("height", 900), "ts": d["timestamp"]}
        self.ann: dict[int, list[GTObject]] = {i: [] for i in range(len(self.samples))}
        for a in L("sample_annotation"):
            fi = self.sample_index[a["sample_token"]]
            name = self.instance_cls[a["instance_token"]]
            self.ann[fi].append(GTObject(a["instance_token"], CATEGORY_MAP.get(name, name), np.array(a["translation"]),
                                         np.array(a["size"]), quat_to_R(a["rotation"])))

    def __len__(self):
        return len(self.samples)

    def channels(self) -> list[str]:
        return sorted({ch for (_, ch) in self.sd if ch.startswith("CAM")})

    def image_path(self, frame: int, channel: str) -> Path:
        return self.root / self.sd[(frame, channel)]["file"]

    def timestamp(self, frame: int) -> float:
        return frame / FPS

    def camera(self, frame: int, channel: str) -> CameraGeometry:
        d = self.sd[(frame, channel)]
        cs, pose = d["calib"], d["pose"]
        R_cs, t_cs = quat_to_R(cs["rotation"]), np.array(cs["translation"])
        R_ego, t_ego = quat_to_R(pose["rotation"]), np.array(pose["translation"])
        # world -> ego -> sensor
        R = R_cs.T @ R_ego.T
        t = -R_cs.T @ (R_ego.T @ t_ego) - R_cs.T @ t_cs
        return CameraGeometry.from_KRt(np.array(cs["camera_intrinsic"]), R, t, d["w"], d["h"], source="V2X-Sim calibration")

    def ego_position(self, frame: int, agent: int) -> np.ndarray:
        ch = "CAM_id_0_0" if agent == 0 else f"CAM_FRONT_id_{agent}"
        return np.array(self.sd[(frame, ch)]["pose"]["translation"])

    def seg_depth(self, frame: int, channel: str):
        seg_ch = channel.replace("CAM", "SEG", 1)
        dep_ch = channel.replace("CAM", "DEP", 1)
        seg = dep = None
        if (frame, seg_ch) in self.sd:
            p = self.root / self.sd[(frame, seg_ch)]["file"]
            if p.exists():
                seg = np.load(p)["arr_0"]
        if (frame, dep_ch) in self.sd:
            p = self.root / self.sd[(frame, dep_ch)]["file"]
            if p.exists():
                dep = np.load(p)["arr_0"] * 1000.0  # CARLA normalised depth, far plane 1000 m
        return seg, dep

    def visibility(self, frame: int, channel: str, cam: CameraGeometry | None = None, max_range: float = 60.0) -> dict[str, dict]:
        """Per-object ground-truth visibility in one camera from segmentation + depth.

        An object is `in_fov` if its 3D box centre projects inside the image within `max_range`;
        `visible_px` counts pixels inside its projected box that carry a vehicle/pedestrian tag
        and whose depth agrees with the box (so pixels of an occluding car in front do not count).
        """
        cam = cam or self.camera(frame, channel)
        seg, dep = self.seg_depth(frame, channel)
        out = {}
        for o in self.ann[frame]:
            ctr_px, ctr_depth = cam.project3d(o.center[None])
            dist = cam.ground_distance(o.ground_xy)
            in_fov = bool(ctr_depth[0] > 0.5 and 0 <= ctr_px[0, 0] < cam.width and 0 <= ctr_px[0, 1] < cam.height and dist <= max_range)
            rec = {"in_fov": in_fov, "dist": dist, "visible_px": 0, "box": None}
            if ctr_depth[0] > 0.5:
                px, d = cam.project3d(o.corners())
                if (d > 0.3).all():
                    x1, y1 = np.floor(px.min(0)).astype(int)
                    x2, y2 = np.ceil(px.max(0)).astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, cam.width), min(y2, cam.height)
                    if x2 > x1 and y2 > y1:
                        rec["box"] = [x1, y1, x2, y2]
                        if seg is not None and dep is not None:
                            s = seg[y1:y2, x1:x2]
                            z = dep[y1:y2, x1:x2]
                            # planar depth vs euclidean range: CARLA depth is planar (camera z)
                            dmin, dmax = d.min() - 0.6, d.max() + 0.6
                            m = ((s == SEG_VEHICLE) | (s == SEG_PEDESTRIAN)) & (z >= dmin) & (z <= dmax)
                            rec["visible_px"] = int(m.sum())
                            rec["box_area"] = int((x2 - x1) * (y2 - y1))
            out[o.instance] = rec
        return out
