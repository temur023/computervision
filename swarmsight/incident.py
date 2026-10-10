"""Reconstruction of a real crossing from one dashcam, and what CrossGuard would have done with a camera at it.

Inputs (demo/incident/scene.json, built by benchmarks/build_incident.py): anonymised frames, the car's GPS and
speed read from the dashcam overlay, the camera calibration, and our annotations of the motorcycle that the detector
cannot see.  Everything below is computed here:

1. Our car (the dashcam car): straight path on the GPS bearing, distance integrated from the overlay speed.
2. The car ahead: road-user detector + ByteTrack on every frame; ground contact point -> metres with the calibrated
   camera (horizon row, focal length, camera height) -> map coordinates with our car's pose.
3. The motorcycle: its speed from the annotated boxes (image motion scaled by its apparent size: a rider on a
   motorcycle is about 1.5 m tall), its path through the impact point, perpendicular to our road (assumption).
4. What this car's own camera allowed: the motorcycle was visible for the last `visible_s` seconds only.
5. Counterfactual: a camera at the crossing (or connected cars) that sees the cross street up to 40-150 m.
   CrossGuard runs on the car ahead (tracked) and the motorcycle as that camera would have tracked it; for each
   range: the first warning, and whether a driver reacting in 1.0 s (or 1.5 s) and braking at 7 m/s² keeps the car
   out of the motorcycle's path at the moment it crosses.

The reconstruction is approximate (single wide-angle camera, 1 Hz GPS): positions +-15%, motorcycle speed +-20%.
The output says which numbers are measured, estimated or assumed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .conflict import Agent, ConflictMonitor, assess, explain
from .guardian import CameraNode

R_EARTH = 6378137.0
RIDER_HEIGHT_M = 1.5
CAR_HALF_LENGTH = 2.3
CROSSING_CAMERA_RANGES = (40.0, 60.0, 80.0, 100.0, 120.0, 150.0)
TRACK_CONFIRM_S = 0.4  # a new track needs a few frames before it is trusted
REACTION_S = (1.0, 1.5)
DECEL = 7.0  # m/s², dry asphalt with ABS


def _local(lat, lon, lat0, lon0):
    return np.array([math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0)), math.radians(lat - lat0) * R_EARTH])


def _latlon(e, n, lat0, lon0):
    return [round(lat0 + math.degrees(n / R_EARTH), 7), round(lon0 + math.degrees(e / (R_EARTH * math.cos(math.radians(lat0)))), 7)]


class EgoPath:
    """Our car: GPS bearing for the direction, overlay speed integrated for the distance."""

    def __init__(self, telemetry: list[dict]):
        tel = sorted(telemetry, key=lambda r: r["t"])
        self.t = np.array([r["t"] for r in tel], float)
        self.v = np.array([r["kmh"] for r in tel], float) / 3.6
        stopped = [r for r in tel if r["kmh"] == 0]
        last_stop = stopped[-1]
        self.lat0 = float(np.mean([r["lat"] for r in stopped if r["t"] >= last_stop["t"] - 6]))
        self.lon0 = float(np.mean([r["lon"] for r in stopped if r["t"] >= last_stop["t"] - 6]))
        moving = [r for r in tel if r["kmh"] > 0]
        a, b = moving[1], tel[-1]  # first fix that has moved, last fix
        d = _local(b["lat"], b["lon"], a["lat"], a["lon"])
        self.bearing = math.atan2(d[0], d[1])  # clockwise from north
        self.dir = np.array([math.sin(self.bearing), math.cos(self.bearing)])
        self.right = np.array([math.cos(self.bearing), -math.sin(self.bearing)])
        ts = np.arange(self.t[0], self.t[-1] + 1e-9, 0.01)
        vs = np.interp(ts, self.t, self.v)
        self._ts, self._s = ts, np.concatenate([[0.0], np.cumsum(0.5 * (vs[1:] + vs[:-1]) * 0.01)])
        gps_dist = float(np.linalg.norm(d))
        self.check = {"gps_distance_m": round(gps_dist, 1),
                      "speed_integrated_m": round(float(np.interp(b["t"], ts, self._s) - np.interp(a["t"], ts, self._s)), 1)}

    def speed(self, t: float) -> float:
        return float(np.interp(t, self.t, self.v))

    def pos(self, t: float) -> np.ndarray:
        return self.dir * float(np.interp(t, self._ts, self._s))

    def latlon(self, xy) -> list[float]:
        return _latlon(float(xy[0]), float(xy[1]), self.lat0, self.lon0)


def _ground(box, cal) -> tuple[float, float] | None:
    """Forward and right distance (m) of the ground contact point of a 1280x720 box."""
    yb = box[3]
    dy = yb - cal["horizon_y"]
    if dy < 6:
        return None
    z = cal["focal_px"] * cal["camera_height_m"] / dy
    x = ((box[0] + box[2]) / 2 - cal["vanishing_x"]) * cal["camera_height_m"] / dy
    return z, x


def reconstruct(scene_dir: str | Path, detector) -> dict:
    scene_dir = Path(scene_dir)
    sc = json.loads((scene_dir / "scene.json").read_text())
    cal = sc["calibration"]
    k = sc["frame_scale"]
    ego = EgoPath(sc["telemetry"])
    t_imp = sc["events"]["impact_t"]
    node = CameraNode("dashcam", None, track_buffer=10)
    frames, per_track = [], {}
    for fr in sc["frames"]:
        img = cv2.imread(str(scene_dir / "frames" / fr["img"]))
        node.process(detector(img), fr["t"])
        out = []
        for tr in node.last_tracks:
            out.append({"id": int(tr["id"]), "cls": tr["cls"], "bbox": tr["bbox"]})
            if tr["cls"] in ("car", "truck", "bus"):
                per_track.setdefault(int(tr["id"]), []).append((fr["t"], [v / k for v in tr["bbox"]]))
        frames.append({"t": fr["t"], "img": fr["img"], "tracks": out})

    # the car ahead: the vehicle track still followed just before impact, nearest to the centre of our lane;
    # earlier tracks of the same car (the tracker loses it while it passes the traffic officer) are chained on
    def lead_score(item):
        tid, obs = item
        late = [o for o in obs if o[0] >= t_imp - 0.5]
        if not late:
            return -1
        b = late[-1][1]
        return len(obs) - abs((b[0] + b[2]) / 2 - cal["vanishing_x"]) / 50
    lead_id, chain = max(per_track.items(), key=lead_score)
    chain = list(chain)
    used = {lead_id}
    while True:
        t0c, b0 = chain[0]
        c0 = np.array([(b0[0] + b0[2]) / 2, b0[3]])
        cands = [(tid, obs) for tid, obs in per_track.items() if tid not in used and obs[0][0] < t0c and t0c - obs[-1][0] <= 1.0
                 and np.linalg.norm(np.array([(obs[-1][1][0] + obs[-1][1][2]) / 2, obs[-1][1][3]]) - c0) < 160]
        if not cands:
            break
        tid, obs = max(cands, key=lambda x: len(x[1]))
        used.add(tid)
        chain = [o for o in obs if o[0] < t0c] + chain
    lead = []
    for t, b in chain:
        g = _ground(b, cal)
        if g is None:
            continue
        z, x = g
        rear = ego.pos(t) + ego.dir * z + ego.right * x
        lead.append((t, rear + ego.dir * CAR_HALF_LENGTH))  # centre of the car, half a length beyond its rear
    lt = np.array([p[0] for p in lead])
    lp = np.array([p[1] for p in lead])
    if len(lp) >= 7:  # centred moving average over 7 frames (0.9 s): monocular positions are noisy
        ker = np.ones(7) / 7
        pad = np.pad(lp, ((3, 3), (0, 0)), mode="edge")
        lp = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(2)], 1)
    late = lt >= t_imp - 1.2
    vel = np.array([np.polyfit(lt[late], lp[late, 0], 1)[0], np.polyfit(lt[late], lp[late, 1], 1)[0]])
    lead_speed = float(np.linalg.norm(vel))
    lead_at = lambda t: np.array([np.interp(t, lt, lp[:, 0]), np.interp(t, lt, lp[:, 1])]) if t <= lt[-1] else lp[-1] + vel * (t - lt[-1])
    lead_vel = lambda t: (lead_at(t + 0.25) - lead_at(t - 0.25)) / 0.5 if lt[0] + 0.25 <= t else np.zeros(2)

    # the motorcycle: speed from the annotated boxes, path through the impact point
    ann = sc["annotations"]["motorcycle_1"]
    ta = np.array([a["t"] for a in ann])
    xc = np.array([(a["box"][0] + a["box"][2]) / 2 for a in ann])
    hpx = np.array([a["box"][3] - a["box"][1] for a in ann])
    px_per_s = float(np.polyfit(ta, xc, 1)[0])
    m_per_px = RIDER_HEIGHT_M / float(np.mean(hpx[-3:]))
    v_moto = px_per_s * m_per_px
    m_dir = ego.right  # crossing from the left, perpendicular to our road (assumption)
    impact = lead_at(t_imp) - ego.right * 0.9  # the motorcycle struck the left side of the car ahead
    moto_at = lambda t: impact - m_dir * v_moto * (t_imp - t)
    visible_from = float(ta[0])

    # counterfactual: a camera at the crossing (or connected cars) watching the cross street up to `rng` metres
    hd = math.atan2(ego.dir[1], ego.dir[0])
    s_line = float(np.dot(impact, ego.dir))  # where the motorcycle's line crosses the path of the car ahead

    def scenario(rng: float) -> dict:
        t_detect = t_imp - rng / v_moto
        t_track = t_detect + TRACK_CONFIRM_S
        mon = ConflictMonitor(raise_n=2)
        t_warn, first, timeline = None, None, []
        for t in np.round(np.arange(math.ceil(t_track * 10) / 10, t_imp + 1e-6, 0.1), 2):  # one time grid for every range
            age = float(t - t_detect)
            lh = [(s, *lead_at(s), *lead_vel(s)) for s in np.arange(t - 1.0, t + 1e-6, 0.2)]
            mh = [(s, *moto_at(s), *(m_dir * v_moto)) for s in np.arange(max(t_detect, t - 1.0), t + 1e-6, 0.2)]
            car = Agent("car", "car", lead_at(t), lead_vel(t), 0.5, 0.6, lh, ego=True, heading=hd)
            moto = Agent("moto", "motorcycle", moto_at(t), m_dir * v_moto, 0.8, 1.0 if age >= TRACK_CONFIRM_S else 2.0, mh, age_s=age)
            r = assess(car, moto, seed=int(t * 10))
            al = mon.update([r] if r and r["level"] != "low" else [], float(t))
            timeline.append({"t": float(t), "level": r["level"] if r else "low", "p": r["p"] if r else 0.0, "ttc_s": r["ttc_s"] if r else None})
            if al and t_warn is None:
                t_warn, first = float(t), r
                first["reason"], first["action"] = explain(r, car, moto)
        out = {"range_m": rng, "t_detect": round(t_detect, 2), "t_track": round(t_track, 2), "timeline": timeline, "t_warn": None}
        if t_warn is None:
            return out
        v_warn = float(np.linalg.norm(lead_vel(t_warn)))
        v0 = float(np.linalg.norm(vel))  # speed when it entered the crossing
        res = []
        along = lambda t: float(np.dot(lead_at(t), ego.dir))
        for rs in REACTION_S:
            tb0 = t_warn + rs  # braking starts after the driver reacts; until then the car does what it really did
            if tb0 >= t_imp:
                deficit = 0.0
            else:
                vb = float(np.dot(lead_vel(tb0), ego.dir))
                tau = min(t_imp - tb0, max(vb, 0.0) / DECEL)
                deficit = along(t_imp) - (along(tb0) + max(vb, 0.0) * tau - 0.5 * DECEL * tau * tau)
            front = s_line + CAR_HALF_LENGTH - deficit  # front bumper along the path when the motorcycle crosses
            clear = (s_line - 0.5) - front  # >0: the front stays this far short of the motorcycle's line
            pos = lead_at(t_imp) - ego.dir * deficit
            res.append({"reaction_s": rs, "clear_m": round(clear, 1), "avoided": clear > 0, "car_at": ego.latlon(pos)})
        out.update({"t_warn": round(t_warn, 2), "lead_s": round(t_imp - t_warn, 2), "car_speed_kmh": round(v_warn * 3.6),
                    "car_mode": first["hyp"][0], "action": first.get("action"),
                    "warning": {"level": first["level"], "p": first["p"], "ttc_s": first["ttc_s"], "reason": first["reason"]},
                    "braking": res})
        return out

    scenarios = [scenario(r) for r in CROSSING_CAMERA_RANGES]
    need = next((sc_["range_m"] for sc_ in scenarios if sc_["t_warn"] is not None and sc_["braking"][0]["avoided"]), None)
    out_cf = {"scenarios": scenarios, "min_range_avoided_m": need, "decel_ms2": DECEL}

    def ll(xy):
        return ego.latlon(xy)
    for f in frames:
        t = f["t"]
        f["ego"] = ll(ego.pos(t))
        f["ego_kmh"] = round(ego.speed(t) * 3.6)
        f["lead"] = ll(lead_at(t)) if lt[0] <= t <= t_imp else None
        a = next((a for a in ann if abs(a["t"] - t) < 0.07), None)
        f["moto_box"] = [round(v * k, 1) for v in a["box"]] if a else None
    paths = {
        "ego": [ll(ego.pos(t)) for t in np.arange(sc["clip_seconds"][0], t_imp + 0.75, 0.25)],
        "lead": [ll(p) for p in lp[lt <= t_imp]] + [ll(lead_at(t_imp))],
        "moto_seen": [ll(moto_at(t)) for t in np.linspace(visible_from, t_imp, 6)],
        "moto_before": [ll(moto_at(t)) for t in np.linspace(t_imp - 3.0, visible_from, 12)],
        "camera_cover": {str(int(r)): [ll(impact - m_dir * d) for d in (0.0, r)] for r in CROSSING_CAMERA_RANGES},
    }
    return {
        "name": sc["name"], "source": sc["source"], "frames": frames, "fps": sc["fps"], "impact_t": t_imp,
        "impact": ll(impact), "paths": paths, "bearing_deg": round(math.degrees(ego.bearing), 1),
        "motorcycle": {"speed_kmh": round(v_moto * 3.6), "range_kmh": [round(v_moto * 3.6 * 0.8), round(v_moto * 3.6 * 1.2)],
                       "visible_from_t": visible_from, "visible_s": round(t_imp - visible_from, 2),
                       "position_at": {str(s): ll(moto_at(t_imp - s)) for s in (1, 2, 3)}},
        "lead_car": {"track_ids": sorted(used), "speed_kmh": round(lead_speed * 3.6), "observations": len(lead),
                     "first_seen_t": float(lt[0])},
        "ego_check": ego.check, "counterfactual": out_cf,
        "measured": ["our car's GPS and speed (dashcam overlay)", "the car ahead (detector + tracker, every frame)",
                     "when the motorcycle becomes visible and the moment of impact (frame by frame)"],
        "estimated": ["motorcycle speed from 5 annotated frames (+-20%)", "positions from one camera (+-20%)"],
        "assumed": ["the motorcycle rides straight, perpendicular to our road", "a camera at the crossing sees the cross street up to the chosen range",
                    "track confirmed 0.4 s after first detection; driver reaction 1.0 s (1.5 s shown too); braking 7 m/s²",
                    "the motorcycle struck the middle of the car's left side"],
    }
