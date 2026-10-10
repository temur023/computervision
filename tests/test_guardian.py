import numpy as np

from swarmsight.detector import Detections
from swarmsight.guardian import GuardianScene, compose_road_users
from tests.test_geometry import look_at_camera

NAMES = {0: "person", 1: "bicycle", 2: "car"}


def box_of(cam, xy, size, height):
    r = size / 2
    pts = np.array([[xy[0] + dx, xy[1] + dy, z] for dx in (-r, r) for dy in (-r, r) for z in (0.0, height)])
    px, _ = cam.project3d(pts)
    return [*px.min(0), *px.max(0)]


def dets(items):
    return Detections(np.array([b for b, _ in items], np.float32).reshape(-1, 4), np.full(len(items), 0.9, np.float32),
                      np.array([c for _, c in items], np.int32), NAMES)


def test_rider_merged_with_bike():
    d = Detections(np.array([[100, 50, 140, 160], [95, 110, 150, 175]], np.float32), np.array([0.9, 0.8], np.float32),
                   np.array([0, 1], np.int32), NAMES)
    users = compose_road_users(d)
    assert len(users) == 1 and users[0][1] == "cyclist"


def test_pedestrian_hidden_behind_car_is_reported_to_blocked_camera():
    cam_a = look_at_camera([0, -12, 1.5], [0, 20, 0])       # low car camera, looking north
    cam_b = look_at_camera([12, 8, 7.0], [0, 10, 0])        # roadside camera, looking west
    car_xy, ped_xy = [0, 0], [0.3, 6]
    scene = GuardianScene(ttl=1.0)
    scene.add_camera("A", cam_a)
    scene.add_camera("B", cam_b)
    for k in range(5):
        t = k * 0.2
        scene.ingest("B", dets([(box_of(cam_b, ped_xy, 0.5, 1.7), 0), (box_of(cam_b, car_xy, 4.5, 1.5), 2)]), t)
        # camera A sees only the car; the pedestrian behind it is occluded
        scene.ingest("A", dets([(box_of(cam_a, car_xy, 4.5, 1.5), 2)]), t)
    view = scene.view_for("A", 0.8)
    hidden = [a for a in view["alerts"] if a["status"] == "occluded"]
    assert len(hidden) == 1
    assert hidden[0]["cls"] == "person" and hidden[0]["occluded_by"]["cls"] == "car"
    assert hidden[0]["sources"] == ["B"]
    assert np.linalg.norm(np.array(hidden[0]["position_m"]) - ped_xy) < 1.0
    # the car is seen by both cameras and must be one fused object, not two
    cars = [o for o in view["objects"] if o["cls"] == "car"]
    assert len(cars) == 1 and set(cars[0]["sources"]) == {"A", "B"}
