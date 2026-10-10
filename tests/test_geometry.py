import numpy as np

from swarmsight.geometry import CameraGeometry, apply_h


def look_at_camera(pos, target, f=1000.0, w=1280, h=720):
    """World->camera pose (x right, y down, z forward) for a camera at `pos` looking at `target`."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ pos
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return CameraGeometry.from_KRt(K, R, t, w, h)


def test_ground_projection_roundtrip():
    cam = look_at_camera([0, -10, 6], [0, 15, 0])
    for xy in ([0, 5], [3, 12], [-4, 20]):
        px, front = cam.to_image(xy)
        assert front
        back = apply_h(cam.H_i2w, px[None])[0]
        assert np.allclose(back, xy, atol=1e-6)


def test_pose_recovered_from_four_clicked_points():
    true = look_at_camera([2, -8, 4.5], [0, 10, 0], f=1100)
    world = np.array([[0, 0], [3, 0], [3, 6], [0, 6]], float)
    img = np.array([true.to_image(p)[0] for p in world])
    cam = CameraGeometry.from_points(img, world, true.width, true.height)
    assert abs(cam.K[0, 0] - 1100) < 5
    assert abs(cam.height_m - 4.5) < 0.05
    assert np.allclose(cam.center_ground, [2, -8], atol=0.05)
    assert 5 < cam.pitch_deg < 30


def test_uncertainty_grows_with_distance():
    cam = look_at_camera([0, -10, 6], [0, 15, 0])
    near = cam.ground_point([600, 500, 680, 640], "person")
    far = cam.ground_point([630, 330, 650, 370], "person")
    assert near and far
    assert np.linalg.eigvalsh(far[1]).max() > np.linalg.eigvalsh(near[1]).max()
