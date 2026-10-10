"""Ground-plane geometry for collective perception.

Every camera is described by a homography between its image and a shared metric ground plane
(z = 0, metres, right-handed, z up).  When the full pinhole model (K, R, t) is known we use it;
for a phone calibrated by clicking four ground points we recover it from the homography alone
(focal length from the two rotation constraints, principal point at the image centre).

Positions are never returned as bare points: each projected observation carries a 2x2 covariance
obtained by propagating pixel uncertainty through the homography Jacobian plus calibration error.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Typical footprint size (m) and height (m) per class, used for vehicle-centre correction,
# ghost-box rendering and association gates.
CLASS_SHAPE = {
    "person": (0.5, 0.5, 1.7),
    "bicycle": (1.7, 0.6, 1.7),
    "motorcycle": (2.0, 0.8, 1.6),
    "car": (4.5, 1.9, 1.5),
    "bus": (11.0, 2.5, 3.2),
    "truck": (7.0, 2.4, 3.0),
}
DEFAULT_SHAPE = (1.0, 1.0, 1.5)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def h_jacobian(H: np.ndarray, p) -> np.ndarray:
    """d(H(p))/dp for the projective map p -> H p (2x2)."""
    u, v = float(p[0]), float(p[1])
    X, Y, W = H @ np.array([u, v, 1.0])
    return np.array([
        [(H[0, 0] * W - X * H[2, 0]) / W**2, (H[0, 1] * W - X * H[2, 1]) / W**2],
        [(H[1, 0] * W - Y * H[2, 0]) / W**2, (H[1, 1] * W - Y * H[2, 1]) / W**2],
    ])


def fit_homography(img_pts, world_pts) -> tuple[np.ndarray, float]:
    """Image->ground homography from >=4 correspondences. Returns (H, rmse_metres)."""
    img_pts = np.asarray(img_pts, np.float64)
    world_pts = np.asarray(world_pts, np.float64)
    if len(img_pts) < 4:
        raise ValueError("need at least 4 point correspondences")
    method = cv2.RANSAC if len(img_pts) >= 6 else 0
    H, _ = cv2.findHomography(img_pts, world_pts, method, 0.5)
    if H is None:
        raise ValueError("degenerate correspondences (are 3 points collinear?)")
    err = np.linalg.norm(apply_h(H, img_pts) - world_pts, axis=1)
    H = H / np.linalg.norm(H)
    return (H if H[2, 2] >= 0 else -H), float(np.sqrt(np.mean(err**2)))


def focal_from_homography(H_w2i: np.ndarray, cx: float, cy: float) -> float | None:
    """Least-squares focal length from the orthonormality of the first two rotation columns."""
    A = np.linalg.inv(np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])) @ H_w2i
    A = A / np.linalg.norm(A)
    a = A
    p1 = a[0, 0] * a[0, 1] + a[1, 0] * a[1, 1]
    q1 = a[2, 0] * a[2, 1]
    p2 = a[0, 0] ** 2 + a[1, 0] ** 2 - a[0, 1] ** 2 - a[1, 1] ** 2
    q2 = a[2, 0] ** 2 - a[2, 1] ** 2
    den = p1 * p1 + p2 * p2
    if den < 1e-18:
        return None
    u = -(p1 * q1 + p2 * q2) / den  # u = 1/f^2
    if u <= 0:
        return None
    return float(1.0 / np.sqrt(u))


def pose_from_homography(H_w2i: np.ndarray, K: np.ndarray, visible_px=None) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a plane-induced homography given intrinsics. Returns R, t (world->camera).

    The sign ambiguity (r1, r2, t) vs (-r1, -r2, -t) is resolved by requiring that a ground point
    seen in the image (default: bottom-centre pixel) lies in front of the camera.
    """
    M = np.linalg.inv(K) @ H_w2i
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if visible_px is None:
        visible_px = (K[0, 2], 1.8 * K[1, 2])
    g = apply_h(np.linalg.inv(H_w2i), np.array([visible_px]))[0]
    if (lam * M @ np.array([g[0], g[1], 1.0]))[2] < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


@dataclass
class CameraGeometry:
    """A calibrated camera looking at the shared ground plane."""

    H_i2w: np.ndarray  # image px -> ground metres
    width: int
    height: int
    K: np.ndarray | None = None
    R: np.ndarray | None = None  # world -> camera rotation
    t: np.ndarray | None = None
    calib_rmse_m: float = 0.1
    up: float = 1.0  # +1 if world z points up for this camera, -1 if the clicked frame is left-handed
    source: str = "manual"
    max_range_m: float = 80.0

    @classmethod
    def from_points(cls, img_pts, world_pts, width, height, focal_px: float | None = None):
        H, rmse = fit_homography(img_pts, world_pts)
        cam = cls(H_i2w=H, width=width, height=height, calib_rmse_m=max(rmse, 0.05), source="4-point")
        cam._recover_pose(focal_px)
        return cam

    @classmethod
    def from_KRt(cls, K, R, t, width, height, source="dataset"):
        K, R, t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float).reshape(3)
        H_w2i = K @ np.c_[R[:, 0], R[:, 1], t]
        H = np.linalg.inv(H_w2i)
        return cls(H_i2w=H / np.linalg.norm(H), width=width, height=height, K=K, R=R, t=t, calib_rmse_m=0.05, source=source)

    def _recover_pose(self, focal_px=None):
        H_w2i = np.linalg.inv(self.H_i2w)
        cx, cy = self.width / 2.0, self.height / 2.0
        f = focal_px or focal_from_homography(H_w2i, cx, cy)
        if f is None or not (0.3 * self.width < f < 6 * self.width):
            f = 0.8 * max(self.width, self.height)  # typical phone/dashcam field of view
            self.focal_estimated = False
        else:
            self.focal_estimated = True
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.R, self.t = pose_from_homography(H_w2i, self.K)
        c = self.center3d
        self.up = 1.0 if c[2] >= 0 else -1.0

    # --- basic properties -------------------------------------------------------------
    @property
    def center3d(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def center_ground(self) -> np.ndarray:
        return self.center3d[:2].copy()

    @property
    def height_m(self) -> float:
        return float(abs(self.center3d[2]))

    @property
    def pitch_deg(self) -> float:
        """Angle of the optical axis below the horizon."""
        z_axis_world = self.R.T @ np.array([0, 0, 1.0])
        return float(np.degrees(np.arcsin(np.clip(-self.up * z_axis_world[2], -1, 1))))

    def describe(self) -> dict:
        d = {"source": self.source, "calib_rmse_m": round(self.calib_rmse_m, 3), "image": [self.width, self.height]}
        if self.K is not None:
            d.update(focal_px=round(float(self.K[0, 0]), 1), height_m=round(self.height_m, 2),
                     pitch_deg=round(self.pitch_deg, 1), position=[round(float(v), 2) for v in self.center_ground])
        return d

    # --- projections ------------------------------------------------------------------
    def project3d(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) with z measured upwards -> pixels (N,2), depth (N,)."""
        X = np.asarray(X, float).reshape(-1, 3).copy()
        X[:, 2] *= self.up
        Xc = X @ self.R.T + self.t
        uvw = Xc @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]

    def to_image(self, xy) -> tuple[np.ndarray, bool]:
        px, depth = self.project3d(np.r_[np.asarray(xy, float)[:2], 0.0])
        return px[0], bool(depth[0] > 0.1)

    def in_view(self, xy, margin: float = 0.0) -> bool:
        px, front = self.to_image(xy)
        if not front:
            return False
        m = margin
        return (-m <= px[0] <= self.width + m) and (-m <= px[1] <= self.height + m)

    def ground_distance(self, xy) -> float:
        return float(np.linalg.norm(np.asarray(xy[:2]) - self.center_ground))

    def ghost_box(self, xy, cls: str) -> list[float] | None:
        """2D box an object of class `cls` standing at ground point `xy` would occupy in this image."""
        L, W, Hh = CLASS_SHAPE.get(cls, DEFAULT_SHAPE)
        r = max(L, W) / 2
        x, y = xy[0], xy[1]
        corners = np.array([[x + dx, y + dy, z] for dx in (-r, r) for dy in (-r, r) for z in (0.0, Hh)])
        px, depth = self.project3d(corners)
        if (depth <= 0.1).any():
            return None
        x1, y1 = px.min(0)
        x2, y2 = px.max(0)
        return [float(x1), float(y1), float(x2), float(y2)]

    def ground_point(self, bbox, cls: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Ground position (m) and 2x2 covariance of an object from its image bounding box.

        Uses the bottom-centre contact point; for vehicles the point is pushed half a footprint
        away from the camera along the viewing ray (the box bottom is the nearest face).
        Returns None for points above the horizon / beyond max range.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        foot = np.array([(x1 + x2) / 2, y2])
        su = 0.12 * w + 2.0
        sv = 0.04 * h + 2.0
        if y2 >= self.height - 2:  # truncated at the bottom edge: contact point unseen
            sv += 0.6 * h
        if x1 <= 2 or x2 >= self.width - 2:
            su += 0.25 * w
        Xh = self.H_i2w @ np.array([foot[0], foot[1], 1.0])
        if Xh[2] == 0:
            return None
        g = Xh[:2] / Xh[2]
        if self.K is not None:
            _, front = self.to_image(g)
            if not front:
                return None
        J = h_jacobian(self.H_i2w, foot)
        cov = J @ np.diag([su**2, sv**2]) @ J.T + np.eye(2) * self.calib_rmse_m**2
        L, W, _ = CLASS_SHAPE.get(cls, DEFAULT_SHAPE)
        if self.K is not None:
            c = self.center_ground
            ray = g - c
            dist = np.linalg.norm(ray)
            if dist > self.max_range_m or dist < 1e-6:
                return None
            ray /= dist
            depth = (L + W) / 4.0  # half of the mean footprint dimension
            g = g + ray * depth
            cov = cov + np.outer(ray, ray) * (0.5 * depth) ** 2
        return g, cov


def mahalanobis2(d: np.ndarray, cov: np.ndarray) -> float:
    return float(d @ np.linalg.solve(cov, d))


def uncertainty_radius(cov: np.ndarray, sigma: float = 2.0) -> float:
    """Radius of a circle enclosing the `sigma` ellipse (largest axis)."""
    return float(sigma * np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
