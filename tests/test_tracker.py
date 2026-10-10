import numpy as np

from swarmsight.detector import Detections
from swarmsight.tracker import ByteTracker, time_to_contact


def det(boxes, conf=0.9, cls=0):
    b = np.array(boxes, np.float32).reshape(-1, 4)
    return Detections(b, np.full(len(b), conf, np.float32), np.full(len(b), cls, np.int32), {0: "car"})


def test_ids_stable_for_two_moving_objects():
    tr = ByteTracker(min_hits=1)
    ids = []
    for k in range(10):
        out = tr.update(det([[10 + 5 * k, 10, 60 + 5 * k, 60], [300 - 4 * k, 50, 360 - 4 * k, 110]]), t=k * 0.1)
        ids.append(sorted(t.track_id for t in out))
    assert all(i == ids[0] for i in ids) and len(ids[0]) == 2


def test_low_confidence_detection_keeps_track_alive():
    tr = ByteTracker(min_hits=1)
    first = tr.update(det([[100, 100, 150, 200]]), t=0.0)[0].track_id
    out = tr.update(det([[102, 101, 152, 201]], conf=0.2), t=0.1)
    assert out and out[0].track_id == first


def test_time_to_contact_from_box_expansion():
    tr = ByteTracker(min_hits=1)
    for k in range(10):
        s = 40 * (1 + 0.1 * k)  # box grows 10%/frame at 10 fps -> scale rate ~1/s
        out = tr.update(det([[320 - s / 2, 240 - s / 2, 320 + s / 2, 240 + s / 2]]), t=k * 0.1)
    ttc = time_to_contact(out[0])
    assert ttc is not None and 0.5 < ttc < 3.0
