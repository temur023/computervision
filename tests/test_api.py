from fastapi.testclient import TestClient

from swarmsight.server import app


def test_health_calibrate_and_road_events():
    with TestClient(app) as c:
        h = c.get("/api/v1/health").json()
        assert h["ok"] and "road_users" in h["models"]
        cal = c.post("/api/v1/guardian/calibrate", json={"width": 1280, "height": 720,
                     "image_points": [[400, 650], [880, 650], [760, 420], [520, 420]], "rect_w": 3, "rect_l": 8}).json()
        assert cal["height_m"] > 0 and len(cal["fov_polygon"]) > 3
        e = c.post("/api/v1/road-events", json={"type": "pothole", "lat": 41.31, "lon": 69.28, "conf": 0.9, "source_id": "t1"}).json()
        assert e["status"] == "candidate"
        r = c.patch(f"/api/v1/road-events/{e['id']}/review", json={"action": "verify"}).json()
        assert r["status"] == "verified"
        fc = c.get("/api/v1/road-events?status=verified&format=geojson").json()
        assert any(f["properties"]["id"] == e["id"] for f in fc["features"])


def test_scan_image_returns_detections():
    import cv2
    with TestClient(app) as c, open("demo/v2xsim/frames/CAM_FRONT_id_4_055.jpg", "rb") as f:
        r = c.post("/api/v1/scan/image", files={"file": ("a.jpg", f, "image/jpeg")}).json()
    assert any(u["cls"] in ("cyclist", "person", "car") for u in r["road_users"])
    assert r["events"] == []  # no location given -> no road event


def test_label_review_roundtrip():
    with TestClient(app) as c:
        d = c.get("/api/v1/labels/tashkent").json()
        item = d["items"][0]
        f = item["first_pass"]
        r = c.put(f"/api/v1/labels/tashkent/{item['id']}", json={**{k: f[k] for k in ("damage", "types", "uncertain", "not_road")}, "reviewer": "test"}).json()
        assert r["agrees"]
        assert c.get("/api/v1/labels/tashkent").json()["progress"]["reviewed"] >= 1
