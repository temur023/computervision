import pytest
from fastapi.testclient import TestClient

from swarmsight.alerts import decode_ble, encode_ble, normalise, to_denm, to_gpx
from swarmsight.server import app


def test_ble_frame_round_trip_and_crc():
    a = normalise({"kind": "collision_risk", "level": "high", "cls": "motorcyclist", "ttc_s": 2.14, "distance_m": 78.3,
                   "bearing_deg": 270, "speed_kmh": 132, "ttl_s": 3, "id": 7})
    f = encode_ble(a)
    assert len(f) == 16 and f[0] == 0x53
    d = decode_ble(f)
    assert d["kind"] == "collision_risk" and d["level"] == "high" and d["cls"] == "motorcyclist"
    assert d["ttc_s"] == 2.14 and d["distance_m"] == 78.3 and d["bearing_deg"] == 270 and d["speed_kmh"] == 132
    bad = bytearray(f)
    bad[6] ^= 1
    with pytest.raises(ValueError):
        decode_ble(bytes(bad))


def test_denm_and_gpx():
    a = normalise({"kind": "hidden_road_user", "level": "caution", "lat": 41.3288, "lon": 69.3546})
    d = to_denm(a)
    assert d["situation"]["eventType"]["causeCode"] == 12
    assert d["management"]["eventPosition"]["latitude"] == 413288000
    g = to_gpx([{"lat": 41.31, "lon": 69.28, "type": "pothole", "status": "verified", "confidence": 0.81}])
    assert '<wpt lat="41.310000" lon="69.280000">' in g and "Pothole (verified" in g


def test_publish_reaches_a_subscribed_display():
    c = TestClient(app)
    with c.websocket_connect("/ws/alerts/test-chan") as ws:
        assert ws.receive_json()["type"] == "hello"
        r = c.post("/api/v1/alerts/test-chan", json={"kind": "collision_risk", "level": "high", "cls": "car", "ttc_s": 1.5})
        assert r.status_code == 200 and r.json()["delivered_to"] == 1
        m = ws.receive_json()
        assert m["type"] == "alert" and m["ble_decoded"]["ttc_s"] == 1.5 and len(m["ble_hex"].split()) == 16
    assert c.post("/api/v1/alerts/Bad Channel!", json={}).status_code in (400, 404)
