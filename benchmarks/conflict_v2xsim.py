"""CrossGuard benchmark on V2X-Sim 2.0 mini: trajectory prediction and conflict warnings against ground truth.

Everything runs end to end on what the system actually perceives: detections from the cameras (road-side unit +
the other vehicles' front cameras), Guardian tracking and fusion, then swarmsight.conflict.  Ground truth comes
from the simulator's 3D boxes.

1. Prediction error.  For every fused road user matched to a moving ground-truth object (>= 1 m/s), the
   predicted position 1, 2 and 3 s ahead is compared with where the object really was.  Our most likely
   hypothesis, the best of our hypotheses (minFDE), a constant-velocity baseline on the same fused velocity and
   a "stays where it is" baseline.

2. "Is it safe to start now?"  The ego cars 1 and 4 wait at the crossing.  At every frame, for every road user
   crossing or coming towards the ego path, the ground truth asks: had the ego started now (2 m/s² from rest,
   straight on), would its footprint have come within 0.5 m of that road user's *actual* footprint within 6 s?
   That is the label (cross traffic arriving when you would be there: positive; a gap: negative).  The
   prediction is the warning level CrossGuard gave for the matched fused object at that frame.

3. Moving ego (vehicle 2 driving through the crossing): warnings raised vs real near misses (actual footprints
   within 0.5 m in the next 4 s), reported as false warnings per minute of driving.

Usage: python -m benchmarks.conflict_v2xsim --root /path/to/v2x_sim_mini
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.coop_v2xsim import RSU, detect_cached, match
from swarmsight import conflict as cg
from swarmsight.datasets.v2xsim import V2XSim
from swarmsight.detector import load_default_detectors
from swarmsight.guardian import GROUP, GuardianScene

FPS = 5
CLEARANCE_M = 0.5
GT_SHAPE = {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle", "motorcycle": "motorcycle", "person": "person"}
MATCH_THR = {"person": 2.0, "two_wheeler": 2.0, "vehicle": 3.0}


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def gt_tracks(ds):
    tr = {}
    for f in range(len(ds)):
        for o in ds.ann[f]:
            d = tr.setdefault(o.instance, {"cls": o.cls, "pos": {}, "yaw": {}, "half": (o.size[1] / 2, o.size[0] / 2)})
            d["pos"][f] = o.center[:2].copy()
            d["yaw"][f] = yaw_of(o.R)
    return tr


def gt_speed(t, f):
    fs = [g for g in (f - 1, f, f + 1) if g in t["pos"]]
    if len(fs) < 2:
        return np.zeros(2)
    return (t["pos"][fs[-1]] - t["pos"][fs[0]]) * FPS / (fs[-1] - fs[0])


def overlap_box(pa, ya, half_a, pb, yb, half_b, margin=CLEARANCE_M) -> bool:
    """Separating-axis test for two oriented rectangles, each inflated by margin/2."""
    d = np.asarray(pb) - np.asarray(pa)
    La, Wa = half_a[0] + margin / 2, half_a[1] + margin / 2
    Lb, Wb = half_b[0] + margin / 2, half_b[1] + margin / 2
    for th in (ya, ya + math.pi / 2, yb, yb + math.pi / 2):
        ax = np.array([math.cos(th), math.sin(th)])
        ra = La * abs(math.cos(ya - th)) + Wa * abs(math.sin(ya - th))
        rb = Lb * abs(math.cos(yb - th)) + Wb * abs(math.sin(yb - th))
        if abs(float(d @ ax)) > ra + rb:
            return False
    return True


def run_ego(ds, det, cache, tr, ego, egos, det_ms, horizon_frames):
    A = f"CAM_FRONT_id_{ego}"
    sources = RSU + [f"CAM_FRONT_id_{e}" for e in egos if e != ego]
    scene = GuardianScene(ttl=1.0)
    for c in sources:
        scene.add_camera(c, None)
    scene.add_camera(A, None, mount="vehicle")
    N = len(ds)
    pred_rows, start_rows, moving = [], [], {"frames": 0, "alerts": 0, "alerts_true": 0, "gt_conflicts": 0, "gt_warned": 0}
    ego_half = (2.3, 0.95)
    for f in range(N):
        t = f / FPS
        for c in sources + [A]:
            scene.ingest(c, detect_cached(ds, det, cache, f, c, det_ms), t, ds.camera(f, c))
        view = scene.view_for(A, t)
        if f < 5:
            continue
        ego_xy = ds.ego_position(f, ego)[:2]
        camA = ds.camera(f, A)
        fwd = camA.R.T @ np.array([0.0, 0.0, 1.0])
        ego_yaw = math.atan2(fwd[1], fwd[0])
        ego_v = np.linalg.norm(ds.ego_position(min(f + 1, N - 1), ego)[:2] - ds.ego_position(f - 1, ego)[:2]) * FPS / 2
        objs = view["objects"]
        # ground-truth road users at f (not the ego itself)
        gts = [(k, t_) for k, t_ in tr.items() if f in t_["pos"] and np.linalg.norm(t_["pos"][f] - ego_xy) > 1.5]
        gt_xy = np.array([t_["pos"][f] for _, t_ in gts]).reshape(-1, 2)
        est_xy = np.array([o["xy"] for o in objs]).reshape(-1, 2)
        thr = np.array([MATCH_THR[GROUP.get(t_["cls"], "vehicle")] for _, t_ in gts])
        m = {i: j for i, j, _ in match(gt_xy, est_xy, thr)}

        # 1) prediction error of moving matched objects
        agents = {a.id: a for a in scene._agents(t, objs)}
        for i, (k, t_) in enumerate(gts):
            if i not in m or np.linalg.norm(gt_speed(t_, f)) < 1.0:
                continue
            a = agents.get(str(objs[m[i]]["gid"]))
            if a is None:
                continue
            hs = cg.hypotheses(a)
            best = max(hs, key=lambda h: h.weight)
            paths = {h.name: cg.rollout(a, h, 1, None)[0][0] for h in hs}
            cv = cg.rollout(a, cg.Hypothesis("cv", 1.0, float(np.hypot(*a.v)), math.atan2(a.v[1], a.v[0]), 0.0, 0.0), 1, None)[0][0]
            for sec in (1, 2, 3):
                g = f + sec * FPS
                if g not in t_["pos"]:
                    continue
                true = t_["pos"][g]
                kidx = sec * 10 - 1
                pred_rows.append({"h": sec, "ours": float(np.linalg.norm(paths[best.name][kidx] - true)),
                                  "best_of": float(min(np.linalg.norm(p[kidx] - true) for p in paths.values())),
                                  "cv": float(np.linalg.norm(cv[kidx] - true)),
                                  "static": float(np.linalg.norm(a.xy - true))})

        # 2) start-now check while the ego waits; 3) warnings while it drives
        ego_agent = scene.ego_agent(A, t)
        stopped = ego_v < cg.MOVING_MS
        horizon = cg.START_HORIZON_S if stopped else cg.HORIZON_S
        if f + int(horizon * FPS) >= N:
            continue
        if stopped:
            for i, (k, t_) in enumerate(gts):
                v = gt_speed(t_, f)
                if np.linalg.norm(v) < 1.0:
                    continue
                if cg._bearing_word(ego_xy, ego_yaw, t_["pos"][f]) == "behind":
                    continue
                dh = abs(((math.atan2(v[1], v[0]) - ego_yaw) + math.pi) % (2 * math.pi) - math.pi)
                if dh < math.radians(30):
                    continue  # same direction (a car pulling away ahead): out of scope of the start check
                label = False
                for step in range(1, int(horizon * FPS) + 1):
                    g = f + step
                    if g not in t_["pos"]:
                        break
                    tau = step / FPS
                    s = min(0.5 * cg.START_ACCEL * tau * tau, cg.START_VMAX * tau)
                    pe = ego_xy + s * np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
                    if overlap_box(pe, ego_yaw, ego_half, t_["pos"][g], t_["yaw"][g], t_["half"]):
                        label = True
                        break
                p, level, ttc = 0.0, "low", None
                if i in m:
                    a = agents.get(str(objs[m[i]]["gid"]))
                    r = cg.assess(ego_agent, a, seed=int(round(t * 10))) if a is not None else None
                    if r:
                        p, level, ttc = r["p"], r["level"], r["ttc_s"]
                near_est = float(np.min(np.linalg.norm(est_xy - t_["pos"][f], axis=1))) if len(est_xy) else None
                start_rows.append({"ego": ego, "f": f, "cls": t_["cls"], "label": label, "matched": i in m, "p": p, "level": level, "ttc": ttc,
                                   "nearest_est_m": near_est, "xy": t_["pos"][f].round(1).tolist()})
        else:
            moving["frames"] += 1
            near = set()
            for i, (k, t_) in enumerate(gts):
                for step in range(1, int(horizon * FPS) + 1):
                    g = f + step
                    if g not in t_["pos"]:
                        break
                    pe = ds.ego_position(g, ego)[:2]
                    if overlap_box(pe, ego_yaw, ego_half, t_["pos"][g], t_["yaw"][g], t_["half"]):
                        near.add(i)
                        break
            moving["gt_conflicts"] += len(near)  # frame-level: road users that will come within the clearance
            active = {a["gid"] for a in view["crossguard"]["alerts"]}
            new = {a["gid"] for a in view["crossguard"]["alerts"] if a["new"]}
            inv = {j: i for i, j in m.items()}
            moving["alerts"] += len(new)  # warning episodes
            moving["alerts_true"] += sum(1 for j, o in enumerate(objs) if o["gid"] in new and inv.get(j) in near)
            moving["gt_warned"] += sum(1 for i in near if i in m and objs[m[i]]["gid"] in active)
    return pred_rows, start_rows, moving


def oracle_start_rows(ds, tr, egos):
    """The same start-now check fed with ground-truth positions and velocities: the prediction logic alone."""
    rows, N = [], len(ds)
    for ego in egos:
        for f in range(5, N - int(cg.START_HORIZON_S * FPS)):
            ego_xy = ds.ego_position(f, ego)[:2]
            if np.linalg.norm(ds.ego_position(f + 1, ego)[:2] - ds.ego_position(f - 1, ego)[:2]) * FPS / 2 >= cg.MOVING_MS:
                continue
            camA = ds.camera(f, f"CAM_FRONT_id_{ego}")
            fwd = camA.R.T @ np.array([0.0, 0.0, 1.0])
            yaw = math.atan2(fwd[1], fwd[0])
            ea = cg.Agent("ego", "car", ego_xy, np.zeros(2), 0.2, 0.2, [], ego=True, heading=yaw)
            for k, t_ in tr.items():
                if f not in t_["pos"] or np.linalg.norm(t_["pos"][f] - ego_xy) < 1.5:
                    continue
                v = gt_speed(t_, f)
                if np.linalg.norm(v) < 1.0 or cg._bearing_word(ego_xy, yaw, t_["pos"][f]) == "behind":
                    continue
                if abs(((math.atan2(v[1], v[0]) - yaw) + math.pi) % (2 * math.pi) - math.pi) < math.radians(30):
                    continue
                label = False
                for step in range(1, int(cg.START_HORIZON_S * FPS) + 1):
                    g = f + step
                    if g not in t_["pos"]:
                        break
                    tau = step / FPS
                    pe = ego_xy + min(0.5 * cg.START_ACCEL * tau * tau, cg.START_VMAX * tau) * np.array([math.cos(yaw), math.sin(yaw)])
                    if overlap_box(pe, yaw, (2.3, 0.95), t_["pos"][g], t_["yaw"][g], t_["half"]):
                        label = True
                        break
                hist = [(g / FPS, *t_["pos"][g], *gt_speed(t_, g)) for g in range(f - 6, f + 1) if g in t_["pos"]]
                r = cg.assess(ea, cg.Agent(k, t_["cls"], t_["pos"][f], v, 0.3, 0.3, hist, age_s=1.2), seed=f)
                rows.append({"label": label, "matched": True, "p": r["p"] if r else 0.0, "level": r["level"] if r else "low"})
    return rows


def summarize_pred(rows):
    out = {}
    for sec in (1, 2, 3):
        r = [x for x in rows if x["h"] == sec]
        if r:
            out[f"{sec}s"] = {"n": len(r), **{k: round(float(np.median([x[k] for x in r])), 2) for k in ("ours", "best_of", "cv", "static")}}
    return out


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        idx = s == v
        ranks[idx] = ranks[idx].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize_start(rows):
    y = np.array([r["label"] for r in rows], bool)
    hi = np.array([r["level"] == "high" for r in rows], bool)
    ca = np.array([r["level"] in ("high", "caution") for r in rows], bool)
    pct = lambda a, b: round(100.0 * a / b, 1) if b else None
    return {
        "checks": len(rows), "positives": int(y.sum()),
        "high": {"precision_%": pct((hi & y).sum(), hi.sum()), "recall_%": pct((hi & y).sum(), y.sum()), "n": int(hi.sum())},
        "caution_or_high": {"precision_%": pct((ca & y).sum(), ca.sum()), "recall_%": pct((ca & y).sum(), y.sum()), "n": int(ca.sum())},
        "auroc": None if not len(rows) else (lambda v: None if v is None else round(v, 3))(auroc([r["p"] for r in rows], y)),
        "positives_not_perceived": int(sum(1 for r in rows if r["label"] and not r["matched"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--egos", default="1,2,4")
    ap.add_argument("--out", default="benchmarks/results/conflict_v2xsim.json")
    args = ap.parse_args()
    ds = V2XSim(args.root)
    det = load_default_detectors()["road_users"]
    cache = Path(args.root) / "_det_cache"
    cache.mkdir(exist_ok=True)
    egos = [int(e) for e in args.egos.split(",")]
    tr = gt_tracks(ds)
    det_ms: list = []
    pred, start, mov = [], [], {"frames": 0, "alerts": 0, "alerts_true": 0, "gt_conflicts": 0, "gt_warned": 0}
    for ego in egos:
        p, s, m = run_ego(ds, det, cache, tr, ego, egos, det_ms, None)
        pred += p
        start += s
        for k in mov:
            mov[k] += m[k]
        print("ego", ego, "pred rows", len(p), "start checks", len(s), "moving frames", m["frames"], flush=True)
    minutes = mov["frames"] / FPS / 60
    res = {
        "dataset": "V2X-Sim 2.0 mini, scene_5 (Town03 intersection), 100 frames @5 Hz",
        "pipeline": "detections (road-side unit + other vehicles' front cameras) -> Guardian fusion -> swarmsight.conflict",
        "prediction_error_m_median": summarize_pred(pred),
        "start_check": summarize_start(start),
        "start_check_prediction_only": {**summarize_start(oracle_start_rows(ds, tr, [e for e in egos if e in (1, 4)])),
                                        "note": "same check on ground-truth positions and velocities (perfect perception): the prediction logic alone"},
        "start_check_definition": ("label: had the waiting ego started now (2 m/s² from rest, straight on), its footprint would have come within "
                                   f"{CLEARANCE_M} m of the road user's actual footprint within {cg.START_HORIZON_S:.0f} s; road users moving "
                                   ">= 1 m/s, crossing or oncoming, not behind"),
        "moving": {"minutes": round(minutes, 2), "gt_near_misses": mov["gt_conflicts"], "warned_near_misses": mov["gt_warned"],
                   "warnings": mov["alerts"], "warnings_on_near_miss": mov["alerts_true"],
                   "false_warnings_per_min": round((mov["alerts"] - mov["alerts_true"]) / minutes, 2) if minutes else None},
        "thresholds": {"high_p": cg.HIGH_P, "high_ttc_s": cg.HIGH_TTC, "caution_p": cg.CAUTION_P, "start_high_p": cg.START_HIGH_P,
                       "start_caution_p": cg.START_CAUTION_P, "set": "a priori, not tuned on this scene"},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
