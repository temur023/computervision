"""Road Intelligence trust engine.

Turns individual, noisy camera observations into road events a partner can act on.

* Spatial-temporal merge: observations of the same damage family within a type-specific radius
  (GPS error + object size) and lifetime join an existing event instead of creating duplicates.
* Independent corroboration: repeated sightings by one device do not count as independent; the
  event confidence is a noisy-OR over *sources*, each weighted by that source's reliability.
* Source reliability: a Beta(alpha, beta) posterior per device, updated by moderator decisions.
  Devices whose reports get verified gain weight; devices whose reports get rejected lose it.
* Lifecycle: candidate -> corroborated -> verified | rejected, and expired when not re-observed
  within the type's time-to-live.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from .store import Store

# family, merge radius (m), time-to-live (s), base severity
EVENT_TYPES = {
    "pothole": ("surface", 12.0, 30 * 86400, 0.9),
    "alligator_crack": ("surface", 15.0, 90 * 86400, 0.6),
    "transverse_crack": ("surface", 15.0, 90 * 86400, 0.35),
    "longitudinal_crack": ("surface", 15.0, 90 * 86400, 0.3),
    "roadwork": ("roadwork", 40.0, 7 * 86400, 0.5),
    "obstruction": ("obstruction", 25.0, 2 * 3600, 0.8),
}
CORROBORATE_SOURCES = 2
CORROBORATE_CONF = 0.6


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class RoadObservation:
    type: str
    conf: float
    source_id: str
    lat: float | None = None
    lon: float | None = None
    gps_accuracy_m: float | None = None
    severity: float | None = None
    observed_at: float | None = None
    model_version: str = ""
    media: str | None = None
    bbox: list | None = None
    extra: dict | None = None


def damage_severity(cls: str, bbox, img_w: int, img_h: int, conf: float) -> float:
    """0..1 priority: damage type x apparent size x proximity (lower in frame = closer)."""
    base = EVENT_TYPES.get(cls, ("", 0, 0, 0.5))[3]
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1)) / float(img_w * img_h)
    size = min(1.0, math.sqrt(area) / 0.35)
    proximity = min(1.0, max(0.0, y2 / img_h))
    return round(min(1.0, base * (0.55 + 0.25 * size + 0.2 * proximity) * (0.7 + 0.3 * conf)), 3)


def severity_label(s: float) -> str:
    return "high" if s >= 0.6 else "medium" if s >= 0.35 else "low"


class TrustEngine:
    def __init__(self, store: Store):
        self.store = store

    def reliability(self, source_id: str) -> float:
        s = self.store.source(source_id)
        return s["alpha"] / (s["alpha"] + s["beta"])

    def _confidence(self, eid: int) -> tuple[float, int, int]:
        obs = self.store.observations_for(eid)
        best: dict[str, float] = {}
        for o in obs:
            best[o["source_id"]] = max(best.get(o["source_id"], 0.0), o["conf"])
        p_none = 1.0
        for sid, c in best.items():
            p_none *= 1.0 - c * self.reliability(sid)
        return 1.0 - p_none, len(obs), len(best)

    def _status(self, e: dict, conf: float, n_sources: int) -> str:
        if e["status"] in ("verified", "rejected"):
            return e["status"]
        if n_sources >= CORROBORATE_SOURCES and conf >= CORROBORATE_CONF:
            return "corroborated"
        return "candidate"

    def _find_match(self, ob: RoadObservation, now: float) -> dict | None:
        fam, radius, _, _ = EVENT_TYPES.get(ob.type, ("other", 15.0, 86400, 0.5))
        if ob.lat is None or ob.lon is None:
            return None
        gps = ob.gps_accuracy_m or 10.0
        best, best_d = None, 1e9
        for e in self.store.events("status NOT IN ('rejected','expired') AND lat IS NOT NULL AND type=?", (ob.type,)):
            if e["expires_at"] < now:
                continue
            d = haversine_m(ob.lat, ob.lon, e["lat"], e["lon"])
            if d <= max(radius, gps) and d < best_d:
                best, best_d = e, d
        return best

    def ingest(self, ob: RoadObservation) -> dict:
        now = time.time()
        t_obs = ob.observed_at or now
        _, radius, ttl, base = EVENT_TYPES.get(ob.type, ("other", 15.0, 86400, 0.5))
        sev = ob.severity if ob.severity is not None else base
        self.store.source(ob.source_id)
        e = self._find_match(ob, now)
        if e is None:
            hist = [{"t": now, "action": "created", "by": "trust-engine", "note": f"first observation ({ob.type})"}]
            cur = self.store.execute(
                "INSERT INTO road_events(type, lat, lon, radius_m, status, confidence, severity, first_seen, last_seen,"
                " expires_at, n_obs, n_sources, evidence, history) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ob.type, ob.lat, ob.lon, radius, "candidate", 0.0, sev, t_obs, t_obs, t_obs + ttl, 0, 0, ob.media, json.dumps(hist)),
            )
            eid = cur.lastrowid
            merged = False
        else:
            eid = e["id"]
            merged = True
        self.store.execute(
            "INSERT INTO observations(event_id, source_id, type, lat, lon, gps_accuracy_m, conf, severity, observed_at,"
            " received_at, model_version, media, bbox, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (eid, ob.source_id, ob.type, ob.lat, ob.lon, ob.gps_accuracy_m, ob.conf, sev, t_obs, now, ob.model_version,
             ob.media, json.dumps(ob.bbox), json.dumps(ob.extra or {})),
        )
        return self._refresh(eid, merged=merged, observed_at=t_obs, media=ob.media, severity=sev)

    def _refresh(self, eid: int, merged=False, observed_at=None, media=None, severity=None) -> dict:
        e = self.store.event(eid)
        conf, n_obs, n_src = self._confidence(eid)
        obs = self.store.observations_for(eid)
        # confidence-weighted position over all geotagged observations
        geo = [o for o in obs if o["lat"] is not None]
        if geo:
            w = [o["conf"] for o in geo]
            lat = sum(o["lat"] * wi for o, wi in zip(geo, w)) / sum(w)
            lon = sum(o["lon"] * wi for o, wi in zip(geo, w)) / sum(w)
        else:
            lat, lon = e["lat"], e["lon"]
        status = self._status(e, conf, n_src)
        hist = e["history"]
        if status != e["status"]:
            hist.append({"t": time.time(), "action": status, "by": "trust-engine",
                         "note": f"{n_src} independent sources, confidence {conf:.2f}"})
        last_seen = max(e["last_seen"], observed_at or e["last_seen"])
        ttl = EVENT_TYPES.get(e["type"], ("", 0, 86400, 0))[2]
        sev = max(e["severity"], severity or 0.0)
        evidence = e["evidence"]
        if media and (evidence is None or (severity or 0) >= e["severity"]):
            evidence = media
        self.store.execute(
            "UPDATE road_events SET lat=?, lon=?, status=?, confidence=?, n_obs=?, n_sources=?, last_seen=?, expires_at=?,"
            " severity=?, evidence=?, history=? WHERE id=?",
            (lat, lon, status, conf, n_obs, n_src, last_seen, last_seen + ttl, sev, evidence, json.dumps(hist), eid),
        )
        out = self.store.event(eid)
        out["merged"] = merged
        return out

    def review(self, eid: int, action: str, reviewer: str = "moderator", note: str = "") -> dict:
        """Moderator decision. Updates the reliability of every source that reported the event."""
        if action not in ("verify", "reject"):
            raise ValueError("action must be verify or reject")
        e = self.store.event(eid)
        if e is None:
            raise KeyError(eid)
        status = "verified" if action == "verify" else "rejected"
        hist = e["history"] + [{"t": time.time(), "action": status, "by": reviewer, "note": note}]
        self.store.execute("UPDATE road_events SET status=?, history=? WHERE id=?", (status, json.dumps(hist), eid))
        for sid in {o["source_id"] for o in self.store.observations_for(eid)}:
            if action == "verify":
                self.store.update_source(sid, d_alpha=1.0)
            else:
                self.store.update_source(sid, d_beta=1.0)
        return self.store.event(eid)

    def expire(self) -> int:
        now = time.time()
        rows = self.store.events("status IN ('candidate','corroborated','verified') AND expires_at < ?", (now,))
        for e in rows:
            hist = e["history"] + [{"t": now, "action": "expired", "by": "trust-engine", "note": "not re-observed within TTL"}]
            self.store.execute("UPDATE road_events SET status='expired', history=? WHERE id=?", (json.dumps(hist), e["id"]))
        return len(rows)

    def geojson(self, statuses=("verified",), bbox=None, since: float | None = None) -> dict:
        """Partner feed. Source identities and raw tracks are never exposed."""
        self.expire()
        q = f"status IN ({','.join('?' * len(statuses))}) AND lat IS NOT NULL"
        args = list(statuses)
        if since:
            q += " AND last_seen >= ?"
            args.append(since)
        feats = []
        for e in self.store.events(q, args):
            if bbox and not (bbox[0] <= e["lon"] <= bbox[2] and bbox[1] <= e["lat"] <= bbox[3]):
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(e["lon"], 6), round(e["lat"], 6)]},
                "properties": {
                    "id": e["id"], "event_type": e["type"], "status": e["status"],
                    "confidence": round(e["confidence"], 3), "severity": severity_label(e["severity"]),
                    "position_accuracy_m": e["radius_m"], "first_seen": e["first_seen"], "last_seen": e["last_seen"],
                    "expires_at": e["expires_at"], "independent_sources": e["n_sources"], "observations": e["n_obs"],
                    "evidence_policy": "anonymised snapshot on request",
                },
            })
        return {"type": "FeatureCollection", "features": feats}
