from swarmsight.store import Store
from swarmsight.trust import RoadObservation, TrustEngine

LAT, LON = 41.3111, 69.2797


def engine():
    return TrustEngine(Store(":memory:"))


def test_repeat_reports_from_one_device_do_not_corroborate():
    te = engine()
    for _ in range(4):
        e = te.ingest(RoadObservation("pothole", 0.9, "phone-1", LAT, LON, 5))
    assert e["status"] == "candidate" and e["n_sources"] == 1 and e["n_obs"] == 4


def test_independent_sources_corroborate_and_merge_nearby():
    te = engine()
    a = te.ingest(RoadObservation("pothole", 0.8, "phone-1", LAT, LON, 5))
    b = te.ingest(RoadObservation("pothole", 0.7, "phone-2", LAT + 0.00005, LON, 5))  # ~5.5 m away
    assert a["id"] == b["id"] and b["merged"]
    assert b["status"] == "corroborated" and b["n_sources"] == 2
    c = te.ingest(RoadObservation("pothole", 0.7, "phone-3", LAT + 0.001, LON, 5))  # ~110 m away: new event
    assert c["id"] != a["id"]


def test_moderation_updates_source_reliability_and_feed():
    te = engine()
    e = te.ingest(RoadObservation("pothole", 0.9, "phone-1", LAT, LON, 5))
    r0 = te.reliability("phone-1")
    te.review(e["id"], "reject")
    assert te.reliability("phone-1") < r0
    assert te.geojson(("verified",))["features"] == []
    e2 = te.ingest(RoadObservation("alligator_crack", 0.9, "phone-2", LAT, LON, 5))
    te.review(e2["id"], "verify")
    feats = te.geojson(("verified",))["features"]
    assert len(feats) == 1 and "phone-2" not in str(feats)
