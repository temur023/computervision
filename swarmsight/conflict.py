"""CrossGuard: short-horizon trajectory prediction and conflict assessment in the ground plane.

Input: road users in metres (position, velocity, their uncertainty and a short timestamped history), from
Guardian's fusion, from one calibrated camera, or from ground truth in the benchmark.

1. Motion state of each road user: speed, heading, yaw rate and acceleration, fitted over the last ~1.2 s.
2. Weighted motion hypotheses: keep going (constant velocity with the observed acceleration), keep turning
   (observed yaw rate), brake to a stop.  A stopped road user stays.  For our own car, when it is stopped,
   we evaluate "start now" (2 m/s² from rest, straight on, over 6 s): the question a driver has at a crossing
   when the light or the traffic officer lets them go.
3. Each hypothesis is sampled (initial position, speed, heading, acceleration and yaw rate drawn from their
   uncertainty), so the spread of future positions grows along the direction of travel faster than across it.
4. For a pair of road users and every pair of hypotheses we test, step by step over the next 4 s, whether the
   two oriented footprints overlap.  Conflict probability = weighted share of sampled futures that overlap;
   time to conflict = median first-overlap time.
5. For the most probable pair of hypotheses: where the paths meet, when each road user gets there, the gap
   between the two arrival times, and the deceleration our car would need to stop before that point.
6. Risk level with hysteresis (ConflictMonitor): HIGH -> warn, CAUTION -> show, otherwise nothing.

This is decision support for a driver or an operator.  It never brakes, steers or changes a traffic signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HORIZON_S = 4.0  # prediction horizon while moving
START_HORIZON_S = 6.0  # "is it safe to start now?": comparable to the critical gap drivers accept at a crossing
DT = 0.1
TS = np.round(np.arange(1, int(round(START_HORIZON_S / DT)) + 1) * DT, 3)
N_SAMPLES = 48

# footprint half-length / half-width in metres
SHAPE = {"person": (0.35, 0.35), "bicycle": (0.9, 0.35), "cyclist": (0.9, 0.35), "motorcycle": (1.1, 0.45),
         "motorcyclist": (1.1, 0.45), "two_wheeler": (1.0, 0.4), "car": (2.3, 0.95), "vehicle": (2.3, 0.95),
         "truck": (4.0, 1.25), "bus": (6.0, 1.3)}
VRU = {"person", "cyclist", "motorcyclist", "bicycle", "motorcycle", "two_wheeler"}
MOVING_MS = 1.0  # below this (3.6 km/h) a road user counts as stopped: its heading from velocity is noise
START_ACCEL, START_VMAX = 2.0, 10.0  # "start now" for a stopped car
BRAKE_DECEL = 4.0
YAW_DECAY_S = 1.0
GAP_HIGH_S, GAP_CAUTION_S = 0.8, 1.5  # predicted gap between the two arrivals where the paths cross
HIGH_P, HIGH_TTC = 0.35, 3.0
CAUTION_P, CAUTION_TTC = 0.15, 4.0
START_HIGH_P, START_CAUTION_P = 0.35, 0.15  # start-now check: any conflict within START_HORIZON_S
NAMES = {"person": "pedestrian", "cyclist": "cyclist", "motorcyclist": "motorcyclist", "bicycle": "cyclist",
         "motorcycle": "motorcyclist", "two_wheeler": "two-wheeler", "car": "car", "vehicle": "vehicle", "truck": "truck",
         "bus": "bus"}


@dataclass
class Agent:
    id: str
    cls: str
    xy: np.ndarray
    v: np.ndarray
    pos_sigma: float = 0.5
    vel_sigma: float = 0.5
    hist: list = field(default_factory=list)  # [(t, x, y, vx, vy)], oldest first
    ego: bool = False
    heading: float | None = None  # known heading (our car's pose); otherwise from the velocity
    age_s: float = 1.0

    def __post_init__(self):
        self.xy = np.asarray(self.xy, float)[:2]
        self.v = np.asarray(self.v, float)[:2]


@dataclass
class Hypothesis:
    name: str
    weight: float
    speed: float
    heading: float
    accel: float
    yaw_rate: float
    stopped: bool = False


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    t = t - t.mean()
    den = float((t * t).sum())
    return float((t * (y - y.mean())).sum() / den) if den > 1e-9 else 0.0


def motion_state(a: Agent, window: float = 1.2) -> tuple[float, float | None, float, float]:
    """Speed (m/s), heading (rad), yaw rate (rad/s), acceleration (m/s²)."""
    speed = float(np.hypot(*a.v))
    heading = math.atan2(a.v[1], a.v[0]) if speed >= MOVING_MS else a.heading
    if a.ego and a.heading is not None:
        heading = a.heading  # our own car knows its heading from its pose
    yaw_rate = accel = 0.0
    if a.hist:
        h = [r for r in a.hist if r[0] >= a.hist[-1][0] - window]
        if len(h) >= 4 and h[-1][0] - h[0][0] >= 0.5:
            t = np.array([r[0] for r in h], float)
            vx, vy = np.array([r[3] for r in h], float), np.array([r[4] for r in h], float)
            accel = _slope(t, np.hypot(vx, vy))
            mv = np.hypot(vx, vy) >= 1.0
            if mv.sum() >= 4:
                yaw_rate = _slope(t[mv], np.unwrap(np.arctan2(vy[mv], vx[mv])))
    return speed, heading, float(np.clip(yaw_rate, -0.8, 0.8)), float(np.clip(accel, -6.0, 6.0))


def hypotheses(a: Agent) -> list[Hypothesis]:
    s, hd, w, acc = motion_state(a)
    if s < MOVING_MS or hd is None:
        if a.ego and hd is not None:
            return [Hypothesis("start", 1.0, 0.0, hd, START_ACCEL, 0.0)]
        return [Hypothesis("stay", 1.0, 0.0, hd or 0.0, 0.0, 0.0, stopped=True)]
    keep_acc = float(np.clip(acc, -2.0, 2.5))  # up to a normal start-up acceleration
    hs = [Hypothesis("straight", 0.6, s, hd, keep_acc, 0.0)]
    if abs(w) >= 0.15 and s >= 2.0:
        hs[0].weight = 0.35
        hs.append(Hypothesis("turn", 0.5, s, hd, keep_acc, w))
    hs.append(Hypothesis("brake", 0.35 if acc <= -1.0 else 0.12, s, hd, -BRAKE_DECEL, 0.0))
    tot = sum(x.weight for x in hs)
    for x in hs:
        x.weight /= tot
    return hs


def rollout(a: Agent, h: Hypothesis, n: int, rng: np.random.Generator | None):
    """Sampled futures of one hypothesis: positions (n, K, 2), headings (n, K), start positions (n, 2), start headings (n,)."""
    K = len(TS)
    if rng is None:  # nominal trajectory
        n = 1
        x0, h0 = a.xy[None, :], np.array([h.heading])
        s0, acc, w = np.array([h.speed]), np.array([h.accel]), np.array([h.yaw_rate])
    else:
        x0 = a.xy[None, :] + rng.normal(0, a.pos_sigma, (n, 2))
        h0 = h.heading + rng.normal(0, 0.03 if a.ego else min(0.35, a.vel_sigma / max(h.speed, 1.0)), n)
        if h.stopped:
            h0 = np.full(n, h.heading)
        s0 = np.maximum(0.0, h.speed + rng.normal(0, a.vel_sigma, n))
        acc = h.accel + rng.normal(0, 0.8, n)
        # vehicles keep their lane: little unexplained turning; pedestrians and riders wander more
        w = h.yaw_rate + rng.normal(0, (0.04 if a.cls in VRU else 0.015) + 0.15 * abs(h.yaw_rate), n)
    if h.stopped:
        return np.repeat(x0[:, None, :], K, 1), np.repeat(h0[:, None], K, 1), x0, h0
    vmax = START_VMAX if h.name == "start" else np.inf
    sp = np.clip(s0[:, None] + acc[:, None] * TS[None, :], 0.0, vmax)
    # an observed turn fades out (lane changes and merges straighten): yaw rate decays with a 1 s time constant
    hd = h0[:, None] + w[:, None] * (YAW_DECAY_S * (1.0 - np.exp(-TS[None, :] / YAW_DECAY_S)) if h.name == "turn" else TS[None, :])
    step = sp * DT
    pts = x0[:, None, :] + np.cumsum(np.stack([step * np.cos(hd), step * np.sin(hd)], -1), axis=1)
    return pts, hd, x0, h0


def _half_extent(cls: str, heading: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    """Half-extent of the oriented footprint projected on axis (ax, ay)."""
    L, W = SHAPE.get(cls, (1.5, 0.8))
    c, s = np.cos(heading), np.sin(heading)
    return L * np.abs(c * ax + s * ay) + W * np.abs(-s * ax + c * ay)


def _overlap(pa, ha, pb, hb, cls_a, cls_b) -> np.ndarray:
    """Oriented-rectangle overlap by the separating-axis test (4 axes), vectorised over samples and time."""
    dx, dy = pb[..., 0] - pa[..., 0], pb[..., 1] - pa[..., 1]
    hit = np.ones(dx.shape, bool)
    for th in (ha, ha + np.pi / 2, hb, hb + np.pi / 2):
        ax, ay = np.cos(th), np.sin(th)
        sep = np.abs(dx * ax + dy * ay) > _half_extent(cls_a, ha, ax, ay) + _half_extent(cls_b, hb, ax, ay)
        hit &= ~sep
    return hit


def _first_overlap(pa, ha, pb, hb, cls_a, cls_b, xa0, xb0, h0a, h0b):
    hit = _overlap(pa, ha, pb, hb, cls_a, cls_b)
    valid = ~_overlap(xa0[:, None, :], h0a[:, None], xb0[:, None, :], h0b[:, None], cls_a, cls_b)[:, 0]
    any_hit = hit.any(axis=1) & valid
    first = np.where(any_hit, hit.argmax(axis=1), -1)
    return any_hit, first, valid


def _arrival(path: np.ndarray, point: np.ndarray, radius: float) -> float | None:
    d = np.linalg.norm(path - point, axis=1)
    k = np.flatnonzero(d <= radius)
    return float(TS[k[0]]) if len(k) else None


def _bearing_word(ego_xy, ego_heading, xy) -> str:
    ang = math.atan2(xy[1] - ego_xy[1], xy[0] - ego_xy[0]) - ego_heading
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    if abs(ang) <= math.radians(30):
        return "ahead"
    if abs(ang) >= math.radians(150):
        return "behind"
    return "from the left" if ang > 0 else "from the right"


def assess(a: Agent, b: Agent, seed: int = 0) -> dict | None:
    """Conflict between two road users over the next HORIZON_S seconds, or None if they cannot meet."""
    sa, sb = float(np.hypot(*a.v)), float(np.hypot(*b.v))
    reach = (max(sa, START_VMAX if a.ego else 0) + max(sb, START_VMAX if b.ego else 0)) * HORIZON_S + 12.0
    if np.linalg.norm(a.xy - b.xy) > reach:
        return None
    Ha, Hb = hypotheses(a), hypotheses(b)
    if all(h.stopped for h in Ha) and all(h.stopped for h in Hb):
        return None
    # footprints that overlap already now are a duplicate or a measurement artefact, not a future conflict
    if _overlap(a.xy[None, None], np.array([[Ha[0].heading]]), b.xy[None, None], np.array([[Hb[0].heading]]), a.cls, b.cls)[0, 0]:
        return None
    rng = np.random.default_rng(seed)
    prob, best = 0.0, None
    for ha in Ha:
        pa, hda, xa0, ha0 = rollout(a, ha, N_SAMPLES, rng)
        for hb in Hb:
            pb, hdb, xb0, hb0 = rollout(b, hb, N_SAMPLES, rng)
            kh = int(round((START_HORIZON_S if "start" in (ha.name, hb.name) else HORIZON_S) / DT))
            hit, first, valid = _first_overlap(pa[:, :kh], hda[:, :kh], pb[:, :kh], hdb[:, :kh], a.cls, b.cls, xa0, xb0, ha0, hb0)
            share = float(hit.sum() / max(1, valid.sum()))
            w = ha.weight * hb.weight
            prob += w * share
            if share > 0 and (best is None or w * share > best[0]):
                k = first[hit]
                kk = int(np.median(k))
                idx = np.flatnonzero(hit)
                point = 0.5 * (pa[idx, kk].mean(0) + pb[idx, kk].mean(0))
                best = (w * share, ha, hb, float(TS[kk]), point)
    gap = _path_gap(a, b, max(Ha, key=lambda h: h.weight), max(Hb, key=lambda h: h.weight))
    if (best is None or prob < 0.05) and gap is None:
        return None
    if best is None or prob < 0.05:  # no sampled overlap, but the most likely paths cross with a small gap
        ha, hb = max(Ha, key=lambda h: h.weight), max(Hb, key=lambda h: h.weight)
        ttc, point = gap["t"], gap["point"]
    else:
        _, ha, hb, ttc, point = best
    na = rollout(a, ha, 1, None)[0]
    nb = rollout(b, hb, 1, None)[0]
    ta = _arrival(na[0], point, max(1.5, SHAPE.get(a.cls, (1.5, 0.8))[0]))
    tb = _arrival(nb[0], point, max(1.5, SHAPE.get(b.cls, (1.5, 0.8))[0]))
    dh = abs(((ha.heading - hb.heading) + math.pi) % (2 * math.pi) - math.pi)
    kind = "rear-end" if dh < math.radians(30) else "head-on" if dh > math.radians(150) else "crossing"
    if ha.stopped or hb.stopped:
        kind = "approach"
    res = {
        "pair": [a.id, b.id], "cls": [a.cls, b.cls], "p": round(prob, 2), "ttc_s": round(ttc, 1),
        "point": [round(float(point[0]), 2), round(float(point[1]), 2)], "type": kind,
        "arrive_s": [None if ta is None else round(ta, 1), None if tb is None else round(tb, 1)],
        "gap_s": None if ta is None or tb is None else round(abs(ta - tb), 1),
        "speed_kmh": [round(sa * 3.6), round(sb * 3.6)], "hyp": [ha.name, hb.name],
        "paths": [na[0][4::5].round(2).tolist(), nb[0][4::5].round(2).tolist()],
        "conf": round(min(_confidence(a), _confidence(b)), 2),
    }
    if a.ego and sa >= MOVING_MS:
        dist = float(np.linalg.norm(point - a.xy)) - SHAPE.get(a.cls, (2.3, 0.95))[0] - SHAPE.get(b.cls, (1.0, 0.4))[0]
        res["decel_needed"] = round(sa * sa / (2 * max(dist, 0.5)), 1)
    if gap is not None:
        res["path_gap_s"] = gap["gap"]
    res["level"] = _level(res)
    return res


def _path_gap(a: Agent, b: Agent, ha: Hypothesis, hb: Hypothesis) -> dict | None:
    """Where the two most likely paths cross (within their half-widths), and the gap between the arrivals."""
    if ha.stopped or hb.stopped:
        return None
    pa = rollout(a, ha, 1, None)[0][0]
    pb = rollout(b, hb, 1, None)[0][0]
    K = int(round((START_HORIZON_S if "start" in (ha.name, hb.name) else HORIZON_S) / DT))
    pa, pb = pa[:K], pb[:K]
    D = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    reach = SHAPE.get(a.cls, (1.5, 0.8))[1] + SHAPE.get(b.cls, (1.5, 0.8))[1] + 0.5
    i, j = np.unravel_index(int(D.argmin()), D.shape)
    if D[i, j] > reach:
        return None
    ta, tb = float(TS[i]), float(TS[j])
    return {"gap": round(abs(ta - tb), 2), "t": round(min(ta, tb), 1), "point": 0.5 * (pa[i] + pb[j])}


def _confidence(a: Agent) -> float:
    if a.ego:
        return 1.0
    return min(1.0, a.age_s / 0.8) * (1.0 if a.vel_sigma <= 1.0 else 0.6)


def _level(r: dict) -> str:
    g = r.get("path_gap_s")
    if g is not None and r["conf"] >= 0.6:  # the paths cross and the two arrive almost together
        if g <= GAP_HIGH_S and r["ttc_s"] <= (HIGH_TTC if "start" not in r["hyp"] else START_HORIZON_S):
            return "high"
    if "start" in r["hyp"]:
        return "high" if r["p"] >= START_HIGH_P and r["conf"] >= 0.6 else "caution" if r["p"] >= START_CAUTION_P else "low"
    if r["p"] >= HIGH_P and r["ttc_s"] <= HIGH_TTC and r["conf"] >= 0.6:
        return "high"
    if r["p"] >= CAUTION_P and r["ttc_s"] <= CAUTION_TTC:
        return "caution"
    return "low"


def explain(r: dict, a: Agent, b: Agent) -> tuple[str, str]:
    """One-sentence reason and the recommended action, from the point of view of `a` (our car if a.ego)."""
    other = NAMES.get(b.cls, b.cls)
    if a.ego:
        where = _bearing_word(a.xy, a.heading if a.heading is not None else math.atan2(a.v[1], a.v[0]), b.xy)
        who = f"{other.capitalize()} {where}" + (f" at {r['speed_kmh'][1]} km/h" if r["speed_kmh"][1] >= 5 else "")
        tb, ta = r["arrive_s"][1], r["arrive_s"][0]
        if r["hyp"][0] == "start":
            if tb is not None and ta is not None:
                reason = f"{who}: it reaches your path in {tb:.1f} s; if you start now you get there in {ta:.1f} s."
            else:
                reason = f"{who}: if you start now you meet it in {r['ttc_s']:.1f} s."
            action = "Do not start" if r["level"] == "high" else "Wait before starting"
        else:
            if tb is not None and ta is not None and r["type"] == "crossing":
                reason = f"{who}: paths cross in {r['ttc_s']:.1f} s; it gets there in {tb:.1f} s, you in {ta:.1f} s."
            else:
                reason = f"{who}: {r['type']} conflict in {r['ttc_s']:.1f} s."
            dn = r.get("decel_needed")
            action = ("Brake hard" if dn and dn > 6 else "Brake now" if dn and dn > 3 else "Slow down") if r["level"] == "high" else "Be ready to brake"
        return reason, action
    na = NAMES.get(a.cls, a.cls)
    gap = f", {r['gap_s']:.1f} s apart" if r.get("gap_s") is not None else ""
    reason = f"{na.capitalize()} ({r['speed_kmh'][0]} km/h) and {other} ({r['speed_kmh'][1]} km/h): {r['type']} conflict in {r['ttc_s']:.1f} s{gap}."
    return reason, ("Warn both road users" if r["level"] == "high" else "Watch")


def assess_for(ego: Agent, others: list[Agent], seed: int = 0) -> list[dict]:
    """Our car against every other road user, most urgent first."""
    out = []
    s_ego, hd_ego, _, _ = motion_state(ego)
    for i, b in enumerate(others):
        if hd_ego is not None and _bearing_word(ego.xy, hd_ego, b.xy) == "behind":
            continue  # a warning about the car behind is not something our driver can act on
        if s_ego < MOVING_MS and float(np.hypot(*b.v)) < MOVING_MS:
            continue  # both stopped (e.g. a queue at the light): nothing to predict
        r = assess(ego, b, seed + i)
        if r and r["hyp"][0] == "start" and r["type"] == "rear-end":
            continue  # starting behind a slow car ahead: the driver sees it, the question here is cross traffic
        if r and r["level"] != "low":
            r["reason"], r["action"] = explain(r, ego, b)
            out.append(r)
    out.sort(key=lambda r: ({"high": 0, "caution": 1}[r["level"]], r["ttc_s"]))
    return out


def assess_all(agents: list[Agent], seed: int = 0) -> list[dict]:
    """Every pair of road users (road-side camera, no ego), most urgent first."""
    out = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            r = assess(agents[i], agents[j], seed + i * 101 + j)
            if r and r["level"] != "low":
                r["reason"], r["action"] = explain(r, agents[i], agents[j])
                out.append(r)
    out.sort(key=lambda r: ({"high": 0, "caution": 1}[r["level"]], r["ttc_s"]))
    return out


class ConflictMonitor:
    """Turns per-frame assessments into alerts: raise after `raise_n` HIGH frames in a row (at once when the
    conflict is under 1.5 s away), clear after `clear_n` frames without it, one alert id per episode."""

    def __init__(self, raise_n: int = 2, clear_n: int = 3):
        self.raise_n, self.clear_n = raise_n, clear_n
        self.state: dict[tuple, dict] = {}
        self._next = 1

    def update(self, assessments: list[dict], t: float) -> list[dict]:
        seen = set()
        for r in assessments:
            key = tuple(r["pair"])
            seen.add(key)
            st = self.state.setdefault(key, {"hi": 0, "miss": 0, "alert": None})
            st["miss"] = 0
            st["last"] = r
            if r["level"] == "high":
                st["hi"] += 1
                if st["alert"] is None and (st["hi"] >= self.raise_n or (r["ttc_s"] <= 1.5 and r["p"] >= 0.5)):
                    st["alert"] = {"id": self._next, "since": t, "new": True}
                    self._next += 1
            else:
                st["hi"] = 0
        for key, st in list(self.state.items()):
            if key not in seen or st["last"]["level"] != "high":
                st["miss"] += 1
            if st["miss"] >= self.clear_n:
                del self.state[key]
        out = []
        for key, st in self.state.items():
            a = st["alert"]
            if a is None or key not in seen:
                continue
            out.append({**st["last"], "alert_id": a["id"], "since": a["since"], "new": a["new"]})
            a["new"] = False
        return out
