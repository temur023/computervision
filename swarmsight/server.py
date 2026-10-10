"""SwarmSight AI web service (FastAPI): web UI, REST + WebSocket APIs.

Run: uvicorn swarmsight.server:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import platform
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import ExifTags, Image, ImageOps
from pydantic import BaseModel, Field

from . import __version__
from .chargepulse import ChargePulse
from .detector import MODELS_DIR, Detections, load_default_detectors
from .geometry import CameraGeometry
from .guardian import GuardianScene, Observation, compose_road_users
from .pipeline import ScanPipeline, to_road_observation
from .store import Store
from .tracker import ByteTracker
from .trust import RoadObservation, TrustEngine, damage_severity

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DEMO_DIR = Path(os.environ.get("SWARMSIGHT_DEMO", ROOT / "demo"))
RUNTIME = Path(os.environ.get("SWARMSIGHT_RUNTIME", ROOT / "runtime"))
MEDIA_DIR = RUNTIME / "media"
UPLOAD_DIR = RUNTIME / "uploads"
BENCH_DIR = ROOT / "benchmarks" / "results"
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 30 * 1024 * 1024
for d in (MEDIA_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)



class State:
    def __init__(self):
        self.detectors = {}
        self.pipeline: ScanPipeline | None = None
        self.store = Store(RUNTIME / "swarmsight.db")
        self.trust = TrustEngine(self.store)
        self.jobs: dict[str, "Job"] = {}
        self.rooms: dict[str, "Room"] = {}
        self.job_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SWARMSIGHT_JOB_WORKERS", "2")))
        self.live_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SWARMSIGHT_LIVE_WORKERS", "2")))
        self.latency = deque(maxlen=500)  # (kind, ms)
        self.started = time.time()

    def load(self):
        if not self.detectors:
            self.detectors = load_default_detectors(threads=int(os.environ.get("SWARMSIGHT_THREADS", "0") or 0) or None)
            self.pipeline = ScanPipeline(self.detectors, MEDIA_DIR)


S = State()


@asynccontextmanager
async def lifespan(_app):
    S.load()
    yield


app = FastAPI(title="SwarmSight AI", version=__version__,
              lifespan=lifespan, description="Collective vision for connected mobility: Guardian (cooperative perception), "
                          "Road Intelligence (verified road events) and ChargePulse (EV bay occupancy).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- helpers
def _decode_image(data: bytes) -> tuple[np.ndarray, dict]:
    """Decode an upload, apply EXIF orientation, extract GPS + capture time if present."""
    meta: dict = {}
    try:
        im = Image.open(io.BytesIO(data))
        exif = im.getexif()
        if exif:
            gps = exif.get_ifd(0x8825)
            if gps and 2 in gps and 4 in gps:
                def deg(v):
                    return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
                lat, lon = deg(gps[2]), deg(gps[4])
                if gps.get(1) == "S":
                    lat = -lat
                if gps.get(3) == "W":
                    lon = -lon
                meta.update(lat=lat, lon=lon, gps_source="exif")
            dt = exif.get_ifd(0x8769).get(36867) or exif.get(306)
            if dt:
                try:
                    meta["observed_at"] = time.mktime(time.strptime(str(dt), "%Y:%m:%d %H:%M:%S"))
                except ValueError:
                    pass
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), meta
    except Exception:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise HTTPException(400, "could not decode image")
        return arr, meta


async def _read_upload(f: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := await f.read(1 << 20):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, f"file too large (limit {limit // (1 << 20)} MB)")
    return bytes(buf)


def _source_id(raw: str | None) -> str:
    raw = (raw or "anonymous-web")[:64]
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", raw)


def _ingest_observations(obs: list[dict], source: str, lat, lon, acc, observed_at) -> list[dict]:
    events = []
    if lat is None or lon is None:
        return events
    for o in obs:
        e = S.trust.ingest(to_road_observation(o, source, lat, lon, acc, observed_at, S.pipeline.model_version))
        events.append({"id": e["id"], "type": e["type"], "status": e["status"], "merged": e["merged"],
                       "confidence": round(e["confidence"], 3), "n_sources": e["n_sources"]})
    return events


# ----------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.progress = 0.0
        self.messages: list[dict] = []
        self.result = None
        self.error = None
        self.created = time.time()
        self.cancelled = False
        self.lock = threading.Lock()

    def emit(self, msg: dict):
        with self.lock:
            self.messages.append(msg)

    def info(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 3),
                "error": self.error, "n_messages": len(self.messages)}


def _submit(kind: str, fn, *args) -> Job:
    job = Job(kind)
    S.jobs[job.id] = job
    # keep the job table bounded
    if len(S.jobs) > 50:
        for jid in sorted(S.jobs, key=lambda k: S.jobs[k].created)[:-50]:
            S.jobs.pop(jid, None)

    def run():
        job.status = "running"
        try:
            job.result = fn(job, *args)
            job.status = "done"
            job.progress = 1.0
            job.emit({"type": "done", "result": job.result})
        except Exception as e:  # surfaced to the client
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.emit({"type": "error", "error": job.error})

    S.job_pool.submit(run)
    return job


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str, include_result: bool = True):
    job = S.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    d = job.info()
    if include_result and job.status == "done":
        d["result"] = job.result
    return d


@app.delete("/api/v1/jobs/{job_id}")
def job_cancel(job_id: str):
    job = S.jobs.get(job_id)
    if job:
        job.cancelled = True
    return {"ok": True}


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    job = S.jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "error": "unknown job"})
        await ws.close()
        return
    sent = 0
    try:
        while True:
            with job.lock:
                new = job.messages[sent:]
            for m in new:
                await ws.send_text(json.dumps(m, default=_json_default))
            sent += len(new)
            if job.status in ("done", "error") and sent >= len(job.messages):
                break
            if not new:
                await ws.send_json({"type": "progress", "progress": round(job.progress, 3), "status": job.status})
                await asyncio.sleep(0.15)
        await ws.close()
    except WebSocketDisconnect:
        pass


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- health / metrics
@app.get("/api/v1/health")
def health():
    S.load()
    lat = list(S.latency)

    def stats(kind):
        v = [ms for k, ms in lat if k == kind]
        return {"n": len(v), "p50_ms": round(float(np.median(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)} if v else None

    return {
        "ok": True, "version": __version__,
        "models": {k: {"file": Path(d.model_path).name, "imgsz": d.imgsz, "classes": len(d.names)} for k, d in S.detectors.items()},
        "hardware": {"cpu_count": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version()},
        "inference": "CPU · ONNX Runtime (server-side)",
        "uptime_s": round(time.time() - S.started),
        "live_latency": {k: stats(k) for k in ("scan_image", "live_scan", "live_guardian", "video_frame", "guardian_frame")},
    }


@app.get("/api/v1/data")
def data_registry():
    """Dataset cards (source, licence, splits, how labels were made), with live counts where available."""
    reg = json.loads((ROOT / "docs" / "data_registry.json").read_text())
    tk = ROOT / "docs" / "tashkent_testset.json"
    if tk.exists():
        info = json.loads(tk.read_text())
        for d in reg["datasets"]:
            if d["id"] == "tashkent":
                d["used"] = info.get("counts")
                d["status"] = info.get("status", d.get("status"))
    return reg


@app.get("/api/v1/metrics")
def metrics():
    out = {}
    if BENCH_DIR.exists():
        for f in sorted(BENCH_DIR.glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------------------------------------------------------- Scan
@app.post("/api/v1/scan/image")
async def scan_image(file: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None),
                     gps_accuracy_m: float | None = Form(None), source_id: str | None = Form(None),
                     submit: bool = Form(True)):
    S.load()
    data = await _read_upload(file, MAX_IMAGE_BYTES)
    img, meta = _decode_image(data)
    if lat is None and "lat" in meta:
        lat, lon, gps_accuracy_m = meta["lat"], meta["lon"], gps_accuracy_m or 15.0
    t0 = time.perf_counter()
    res = await asyncio.get_running_loop().run_in_executor(S.live_pool, S.pipeline.analyze_image, img)
    ms = (time.perf_counter() - t0) * 1e3
    S.latency.append(("scan_image", ms))
    res["server_ms"] = round(ms, 1)
    res["geotag"] = {"lat": lat, "lon": lon, "accuracy_m": gps_accuracy_m, "source": meta.get("gps_source", "client" if lat is not None else None)}
    res["events"] = _ingest_observations(res["observations"], _source_id(source_id), lat, lon, gps_accuracy_m,
                                         meta.get("observed_at")) if submit else []
    return res


@app.post("/api/v1/scan/video")
async def scan_video(file: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None),
                     gps_accuracy_m: float | None = Form(None), source_id: str | None = Form(None),
                     sample_fps: float = Form(5.0), max_seconds: float = Form(60.0)):
    S.load()
    data = await _read_upload(file, MAX_VIDEO_BYTES)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    src = _source_id(source_id)
    sample_fps = float(np.clip(sample_fps, 1.0, 10.0))
    max_seconds = float(np.clip(max_seconds, 1.0, 120.0))

    def work(job: Job):
        try:
            def on_frame(fr, idx, total):
                S.latency.append(("video_frame", fr["ms"]))
                job.progress = min(0.99, (idx + 1) / total) if total else 0.5
                job.emit({"type": "frame", "frame": fr})
            res = S.pipeline.analyze_video(path, sample_fps, max_seconds, on_frame, should_stop=lambda: job.cancelled)
            frames = res.pop("frames")
            res["n_frames"] = len(frames)
            res["events"] = _ingest_observations(res["observations"], src, lat, lon, gps_accuracy_m, None)
            res["geotag"] = {"lat": lat, "lon": lon, "accuracy_m": gps_accuracy_m}
            return res
        finally:
            path.unlink(missing_ok=True)

    job = _submit("scan_video", work)
    return {"job_id": job.id}


# live phone camera: text {"type":"meta", ...} then binary JPEG; one response per frame
@app.websocket("/ws/live/scan")
async def live_scan(ws: WebSocket):
    await ws.accept()
    S.load()
    loop = asyncio.get_running_loop()
    utrack = ByteTracker(high_thr=0.45, low_thr=0.15, new_track_thr=0.5, track_buffer=15, min_hits=2)
    dtrack = ByteTracker(high_thr=0.35, low_thr=0.15, new_track_thr=0.35, match_thr=0.95, track_buffer=5, min_hits=3)
    reported: set[int] = set()
    meta: dict = {}
    source = "live-" + uuid.uuid4().hex[:8]
    t_start = time.time()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                m = json.loads(msg["text"])
                if m.get("type") == "ping":
                    await ws.send_json({"type": "pong", "t": m.get("t"), "server": time.time() * 1e3})
                    continue
                meta = m
                if m.get("source_id"):
                    source = _source_id(m["source_id"])
                continue
            data = msg.get("bytes")
            if not data:
                continue
            t_recv = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            t_rel = time.time() - t_start

            def process():
                u, d, timing = S.pipeline.detect(img)
                h, w = img.shape[:2]
                users = compose_road_users(u)
                group = {"person": 0, "cyclist": 1, "motorcyclist": 1, "bicycle": 1, "motorcycle": 1}
                ud = Detections(np.array([x[0] for x in users], np.float32).reshape(-1, 4), np.array([x[2] for x in users], np.float32),
                                np.array([group.get(x[1], 2) for x in users], np.int32), {0: "person", 1: "two_wheeler", 2: "vehicle"})
                utr = utrack.update(ud, t_rel)
                from .tracker import time_to_contact
                out_users = []
                for tr in utr:
                    box = tr.history[-1][1]
                    cls = max(users, key=lambda x: _iou(x[0], box))[1] if users else "vehicle"
                    rec = {"id": tr.track_id, "cls": cls, "bbox": [round(float(v), 1) for v in box], "conf": round(tr.conf, 2)}
                    ttc = time_to_contact(tr)
                    cx = (box[0] + box[2]) / 2
                    if ttc is not None and ttc < 10 and 0.3 * w <= cx <= 0.7 * w:
                        rec["ttc"] = round(ttc, 2)
                    out_users.append(rec)
                dtr = dtrack.update(d, t_rel)
                out_dmg, new_obs = [], []
                for tr in dtr:
                    box = tr.history[-1][1]
                    cls = d.names.get(tr.cls, str(tr.cls))
                    sev = damage_severity(cls, box, w, h, tr.conf)
                    out_dmg.append({"id": tr.track_id, "cls": cls, "bbox": [round(float(v), 1) for v in box],
                                    "conf": round(tr.conf, 2), "severity": sev})
                    if tr.track_id not in reported and tr.conf >= 0.4 and meta.get("lat") is not None:
                        reported.add(tr.track_id)
                        media = S.pipeline._save_evidence(img, u, box, cls)
                        e = S.trust.ingest(RoadObservation(cls, tr.conf, source, meta.get("lat"), meta.get("lon"),
                                                           meta.get("acc"), sev, time.time(), S.pipeline.model_version, media,
                                                           [float(v) for v in box], {"live": True}))
                        new_obs.append({"event_id": e["id"], "type": cls, "status": e["status"], "merged": e["merged"]})
                return {"users": out_users, "damage": out_dmg, "events": new_obs, "timing": timing, "w": w, "h": h}

            res = await loop.run_in_executor(S.live_pool, process)
            ms = (time.perf_counter() - t_recv) * 1e3
            S.latency.append(("live_scan", ms))
            res.update(type="result", t=meta.get("t"), server_ms=round(ms, 1))
            await ws.send_text(json.dumps(res, default=_json_default))
    except WebSocketDisconnect:
        pass


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# ----------------------------------------------------------------------------- Guardian
class CalibIn(BaseModel):
    width: int
    height: int
    image_points: list[list[float]] = Field(..., min_length=4)
    world_points: list[list[float]] | None = None
    rect_w: float | None = None  # metres, if the 4 points are corners of a rectangle (clockwise from near-left)
    rect_l: float | None = None
    focal_px: float | None = None
    origin: list[float] | None = None  # world offset of the rectangle's first corner (shared scene frame)


def _calib_from(c: CalibIn) -> CameraGeometry:
    world = c.world_points
    if world is None:
        if not (c.rect_w and c.rect_l) or len(c.image_points) != 4:
            raise HTTPException(400, "give world_points, or exactly 4 image_points with rect_w and rect_l")
        world = [[0, 0], [c.rect_w, 0], [c.rect_w, c.rect_l], [0, c.rect_l]]
    if len(world) != len(c.image_points):
        raise HTTPException(400, "image_points and world_points differ in length")
    if c.origin:
        world = [[p[0] + c.origin[0], p[1] + c.origin[1]] for p in world]
    try:
        return CameraGeometry.from_points(c.image_points, world, c.width, c.height, c.focal_px)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/v1/guardian/calibrate")
def guardian_calibrate(c: CalibIn):
    cam = _calib_from(c)
    d = cam.describe()
    d["H_img_to_ground"] = np.round(cam.H_i2w, 8).tolist()
    d["fov_polygon"] = fov_polygon(cam)
    warn = []
    if not getattr(cam, "focal_estimated", True):
        warn.append("focal length could not be estimated from these points; assumed a typical phone field of view")
    if cam.height_m < 0.3 or cam.height_m > 30:
        warn.append(f"implausible camera height {cam.height_m:.1f} m: check point order and rectangle size")
    d["warnings"] = warn
    return d


def fov_polygon(cam: CameraGeometry, max_range: float = 45.0, n: int = 12) -> list[list[float]]:
    """Ground footprint of the camera's field of view (for bird's-eye rendering)."""
    pts = []
    c = cam.center_ground
    for u in np.linspace(0, cam.width, n):
        best = None
        for v in np.linspace(cam.height, 0, 40):
            p = cam.H_i2w @ np.array([u, v, 1.0])
            if p[2] == 0:
                continue
            g = p[:2] / p[2]
            if not cam.to_image(g)[1]:
                break
            if np.linalg.norm(g - c) > max_range:
                d = (g - c) / np.linalg.norm(g - c)
                best = c + d * max_range
                break
            best = g
        if best is not None:
            pts.append([round(float(best[0]), 2), round(float(best[1]), 2)])
    return [[round(float(c[0]), 2), round(float(c[1]), 2)]] + pts


def cam_pose2d(cam: CameraGeometry) -> dict:
    """Ground position, heading of the optical axis (rad, world frame) and horizontal field of view."""
    fwd = cam.R.T @ np.array([0.0, 0.0, 1.0])
    hfov = 2 * np.arctan(cam.width / (2 * cam.K[0, 0])) if cam.K is not None else 1.2
    return {"pos": [round(float(v), 2) for v in cam.center_ground], "yaw": round(float(np.arctan2(fwd[1], fwd[0])), 4),
            "hfov": round(float(hfov), 4), "height_m": round(cam.height_m, 2)}


def _demo_scene() -> dict:
    p = DEMO_DIR / "v2xsim" / "scene.json"
    if not p.exists():
        raise HTTPException(404, "demo scene not bundled")
    if not hasattr(_demo_scene, "cache"):
        _demo_scene.cache = json.loads(p.read_text())
    return _demo_scene.cache


@app.get("/api/v1/guardian/demo/scene")
def guardian_demo_scene():
    sc = _demo_scene()
    return {k: v for k, v in sc.items() if k != "frames"} | {
        "n_frames": len(sc["frames"]), "preview": {c: v["img"] for c, v in sc["frames"][0]["cams"].items()}}


class DemoIn(BaseModel):
    ego: str = "CAM_FRONT_id_1"
    sources: list[str] = ["CAM_id_0_0", "CAM_id_0_1", "CAM_id_0_2", "CAM_id_0_3"]
    ttl_s: float = 1.0


@app.post("/api/v1/guardian/demo/run")
def guardian_demo_run(req: DemoIn):
    S.load()
    sc = _demo_scene()
    cams = [req.ego] + [c for c in req.sources if c != req.ego]
    for c in cams:
        if c not in sc["cameras"]:
            raise HTTPException(400, f"unknown camera {c}")
    det = S.detectors["road_users"]

    def work(job: Job):
        scene = GuardianScene(ttl=req.ttl_s)
        for c in cams:
            scene.add_camera(c, None)
        n = len(sc["frames"])
        stats = {"occluded_alerts": 0, "frames": 0}
        for i, fr in enumerate(sc["frames"]):
            if job.cancelled:
                break
            t = fr["t"]
            t_frame = time.perf_counter()
            cam_out, det_ms = {}, {}
            # sources first: their observations reach the ego within the same cycle
            for c in cams[1:] + cams[:1]:
                cd = fr["cams"][c]
                img = cv2.imread(str(DEMO_DIR / "v2xsim" / "frames" / cd["img"]))
                geo = CameraGeometry.from_KRt(cd["K"], cd["R"], cd["t"], cd["w"], cd["h"], source="V2X-Sim calibration")
                t0 = time.perf_counter()
                d = det(img)
                det_ms[c] = round((time.perf_counter() - t0) * 1e3, 1)
                scene.ingest(c, d, t, geo)
                cam_out[c] = {"img": cd["img"], "tracks": scene.nodes[c].last_tracks, "fov": fov_polygon(geo), **cam_pose2d(geo)}
            t_fuse = time.perf_counter()
            view = scene.view_for(req.ego, t)
            fuse_ms = (time.perf_counter() - t_fuse) * 1e3 + scene.timing.get("fusion_ms", 0)
            total = (time.perf_counter() - t_frame) * 1e3
            S.latency.append(("guardian_frame", total))
            stats["frames"] += 1
            stats["occluded_alerts"] += sum(1 for a in view["alerts"] if a["status"] == "occluded")
            job.progress = (i + 1) / n
            job.emit({"type": "frame", "i": i, "t": t, "cams": cam_out, "ego": req.ego, "view": view, "gt": fr["gt"],
                      "timing": {"detect_ms": det_ms, "fusion_ms": round(fuse_ms, 1), "frame_total_ms": round(total, 1),
                                 "detect_parallel_ms": max(det_ms.values())}})
        return {"frames": stats["frames"], "occluded_alerts": stats["occluded_alerts"], "ego": req.ego, "sources": cams[1:]}

    job = _submit("guardian_demo", work)
    return {"job_id": job.id}


@app.post("/api/v1/guardian/videos")
async def guardian_videos(video_a: UploadFile = File(...), video_b: UploadFile = File(...), calib_a: str = Form(...),
                          calib_b: str = Form(...), offset_s: float = Form(0.0), sample_fps: float = Form(5.0),
                          max_seconds: float = Form(60.0), ttl_s: float = Form(1.0)):
    """Two static cameras looking at the same ground area. calib_* are CalibIn JSON objects."""
    S.load()
    geos = {}
    for k, raw in (("A", calib_a), ("B", calib_b)):
        try:
            geos[k] = _calib_from(CalibIn(**json.loads(raw)))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise HTTPException(400, f"bad calibration for camera {k}: {e}")
    paths = {}
    for k, f in (("A", video_a), ("B", video_b)):
        data = await _read_upload(f, MAX_VIDEO_BYTES)
        p = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(f.filename or 'v.mp4').suffix or '.mp4'}"
        p.write_bytes(data)
        paths[k] = p
    det = S.detectors["road_users"]
    sample_fps = float(np.clip(sample_fps, 1, 10))

    def work(job: Job):
        try:
            caps = {k: cv2.VideoCapture(str(p)) for k, p in paths.items()}
            fps = {k: (c.get(cv2.CAP_PROP_FPS) or 30.0) for k, c in caps.items()}
            dur = {k: (c.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps[k] for k, c in caps.items()}
            T = min(max_seconds, min(dur["A"], dur["B"] - offset_s) if all(dur.values()) else max_seconds)
            scene = GuardianScene(ttl=ttl_s)
            for k in ("A", "B"):
                scene.add_camera(k, geos[k])
                if geos[k].width != int(caps[k].get(cv2.CAP_PROP_FRAME_WIDTH)):
                    job.emit({"type": "warning", "message": f"camera {k}: calibration width {geos[k].width} differs from video width"})
            n = max(1, int(T * sample_fps))
            fovs = {k: fov_polygon(geos[k]) for k in geos}
            cursor = {"A": -1, "B": -1}
            for i in range(n):
                if job.cancelled:
                    break
                t = i / sample_fps
                out = {}
                t_frame = time.perf_counter()
                for k, tk in (("B", t + offset_s), ("A", t)):
                    img = _read_at(caps[k], fps[k], tk, cursor)
                    if img is None:
                        continue
                    d = det(img)
                    scene.ingest(k, d, t)
                    out[k] = {"tracks": scene.nodes[k].last_tracks}
                views = {k: scene.view_for(k, t) for k in ("A", "B")}
                total = (time.perf_counter() - t_frame) * 1e3
                S.latency.append(("guardian_frame", total))
                job.progress = (i + 1) / n
                job.emit({"type": "frame", "i": i, "t": round(t, 3), "cams": out, "views": views, "fov": fovs,
                          "pos": {k: [round(float(v), 2) for v in geos[k].center_ground] for k in geos},
                          "pose": {k: cam_pose2d(geos[k]) for k in geos},
                          "timing": {"frame_total_ms": round(total, 1)}})
            return {"frames": n, "geometry": {k: g.describe() for k, g in geos.items()}}
        finally:
            for p in paths.values():
                p.unlink(missing_ok=True)

    job = _submit("guardian_videos", work)
    return {"job_id": job.id}


def _read_at(cap, fps: float, t: float, cursor: dict):
    """Sequentially advance a capture to the frame nearest to time t (seeking is unreliable on phone codecs)."""
    key = id(cap)
    target = int(round(t * fps))
    if target < 0:
        return None
    cur = cursor.get(key, -1)
    img = None
    while cur < target:
        ok = cap.grab()
        if not ok:
            return None
        cur += 1
    cursor[key] = cur
    ok, img = cap.retrieve()
    return img if ok else None


class Room:
    """A live Guardian session: several phones as cameras, any number of viewers."""

    def __init__(self, name: str):
        self.name = name
        self.scene = GuardianScene(ttl=1.0)
        self.sockets: dict[str, WebSocket] = {}  # client id -> ws
        self.roles: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.created = time.time()
        self.last_capture: dict[str, float] = {}

    def cameras(self):
        return {cid: r for cid, r in self.roles.items() if r.get("role") == "camera"}


@app.get("/api/v1/guardian/rooms/{room}")
def room_info(room: str):
    r = S.rooms.get(room)
    if not r:
        return {"room": room, "cameras": {}, "viewers": 0}
    return {"room": room, "cameras": {k: {"cam": v.get("cam"), "calibrated": r.scene.nodes.get(v.get("cam")) is not None
                                          and r.scene.nodes[v["cam"]].geometry is not None} for k, v in r.cameras().items()},
            "viewers": sum(1 for v in r.roles.values() if v.get("role") == "viewer")}


@app.websocket("/ws/guardian/{room}")
async def guardian_room(ws: WebSocket, room: str):
    await ws.accept()
    S.load()
    room = re.sub(r"[^A-Za-z0-9_-]", "", room)[:32] or "default"
    r = S.rooms.setdefault(room, Room(room))
    cid = uuid.uuid4().hex[:8]
    r.sockets[cid] = ws
    r.roles[cid] = {"role": "viewer"}
    loop = asyncio.get_running_loop()
    meta: dict = {}
    det = S.detectors["road_users"]
    try:
        await ws.send_json({"type": "welcome", "client": cid, "room": room, "server": time.time() * 1e3})
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                m = json.loads(msg["text"])
                typ = m.get("type")
                if typ == "ping":
                    await ws.send_json({"type": "pong", "t": m.get("t"), "server": time.time() * 1e3})
                elif typ == "hello":
                    role = m.get("role", "viewer")
                    cam = re.sub(r"[^A-Za-z0-9_-]", "", str(m.get("cam", "")))[:16] or None
                    r.roles[cid] = {"role": role, "cam": cam}
                    if role == "camera" and cam and cam not in r.scene.nodes:
                        r.scene.add_camera(cam, None)
                    await _room_broadcast(r, {"type": "roster", "info": room_info(room)})
                elif typ == "calib":
                    cam = r.roles[cid].get("cam")
                    try:
                        geo = _calib_from(CalibIn(**m["calib"]))
                        if cam not in r.scene.nodes:
                            r.scene.add_camera(cam, geo)
                        r.scene.nodes[cam].geometry = geo
                        await ws.send_json({"type": "calib_ok", "geometry": geo.describe(), "fov": fov_polygon(geo)})
                        await _room_broadcast(r, {"type": "roster", "info": room_info(room)})
                    except HTTPException as e:
                        await ws.send_json({"type": "calib_error", "error": e.detail})
                elif typ == "frame":
                    meta = m
                continue
            data = msg.get("bytes")
            if not data or r.roles[cid].get("role") != "camera":
                continue
            cam = r.roles[cid]["cam"]
            t_recv = time.time() * 1e3
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # capture time in server clock: client timestamp corrected by its measured clock offset
            t_cap = float(meta.get("t_capture", t_recv)) + float(meta.get("offset_ms", 0.0))
            t0 = time.perf_counter()
            d = await loop.run_in_executor(S.live_pool, det, img)
            async with r.lock:
                r.scene.ingest(cam, d, t_cap / 1e3)
                r.last_capture[cam] = t_cap
                proc_ms = (time.perf_counter() - t0) * 1e3
                S.latency.append(("live_guardian", proc_ms))
                now = time.time() * 1e3
                # every camera client gets its own perspective; viewers get the whole picture
                for ocid, ows in list(r.sockets.items()):
                    role = r.roles.get(ocid, {})
                    try:
                        if role.get("role") == "camera" and role.get("cam") in r.scene.nodes:
                            v = r.scene.view_for(role["cam"], now / 1e3)
                            v.update(type="view", source_cam=cam, source_capture_ms=t_cap, server_send_ms=now,
                                     processing_ms=round(proc_ms, 1), upload_ms=round(t_recv - t_cap, 1))
                            if ocid != cid and not v["alerts"]:
                                continue  # only push to other cameras when the network has something for them
                            await ows.send_text(json.dumps(v, default=_json_default))
                        elif role.get("role") == "viewer":
                            objs = [{"gid": g.gid, "cls": g.cls, "xy": [round(float(g.x[0]), 2), round(float(g.x[1]), 2)],
                                     "sources": sorted(g.seen)} for g in r.scene.fusion.objects.values()]
                            cams = {c: ({**cam_pose2d(n.geometry), "fov": fov_polygon(n.geometry)} if n.geometry else {"pos": None, "fov": None})
                                        | {"tracks": n.last_tracks} for c, n in r.scene.nodes.items()}
                            await ows.send_text(json.dumps({"type": "world", "objects": objs, "cams": cams,
                                                            "alerts": {c: r.scene.view_for(c, now / 1e3)["alerts"]
                                                                       for c, n in r.scene.nodes.items() if n.geometry},
                                                            "processing_ms": round(proc_ms, 1)}, default=_json_default))
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        r.sockets.pop(cid, None)
        r.roles.pop(cid, None)
        if not r.sockets:
            S.rooms.pop(room, None)
        else:
            await _room_broadcast(r, {"type": "roster", "info": room_info(room)})


async def _room_broadcast(r: Room, msg: dict):
    for ws in list(r.sockets.values()):
        try:
            await ws.send_json(msg)
        except Exception:
            pass


class ObjectObservationIn(BaseModel):
    """Concept Appendix A1 format: an object observation from a partner device."""
    type: str = "object_observation"
    room: str = "default"
    source_id: str
    timestamp_ms: float | None = None
    object: dict
    position_local_m: dict
    position_uncertainty_m: float = 1.5
    calibration_id: str | None = None


@app.post("/api/v1/observations")
async def post_observation(o: ObjectObservationIn):
    r = S.rooms.setdefault(o.room, Room(o.room))
    cam = _source_id(o.source_id)[:16]
    if cam not in r.scene.nodes:
        r.scene.add_camera(cam, None)
    t = (o.timestamp_ms or time.time() * 1e3) / 1e3
    xy = np.array([float(o.position_local_m["x"]), float(o.position_local_m["y"])])
    obs = Observation(cam, int(o.object.get("track_id", 0)), str(o.object.get("class", "person")),
                      float(o.object.get("confidence", 0.5)), [0, 0, 0, 0], xy, np.eye(2) * o.position_uncertainty_m**2, t)
    async with r.lock:
        r.scene.fusion.step([obs], t)
    return {"ok": True, "room": o.room, "objects": len(r.scene.fusion.objects)}


@app.get("/api/v1/objects/nearby")
def objects_nearby(room: str = "default", cam: str = Query(..., description="subscriber camera id")):
    r = S.rooms.get(room)
    if not r or cam not in r.scene.nodes:
        raise HTTPException(404, "unknown room or camera")
    return r.scene.view_for(cam, time.time())


# ----------------------------------------------------------------------------- Road Intelligence
class RoadEventIn(BaseModel):
    type: str
    lat: float
    lon: float
    conf: float = 0.5
    severity: float | None = None
    gps_accuracy_m: float | None = 10.0
    source_id: str | None = None
    media: str | None = None
    bbox: list[float] | None = None
    observed_at: float | None = None
    count: int = 1


@app.post("/api/v1/road-events")
def create_road_event(e: RoadEventIn):
    if e.media and not (re.fullmatch(r"[0-9a-f]{16}\.jpg", e.media) and (MEDIA_DIR / e.media).exists()):
        raise HTTPException(400, "unknown media reference")
    ev = S.trust.ingest(RoadObservation(e.type, float(np.clip(e.conf, 0, 1)), _source_id(e.source_id), e.lat, e.lon,
                                        e.gps_accuracy_m, e.severity, e.observed_at, S.pipeline.model_version if S.pipeline else "",
                                        e.media, e.bbox, {"count": e.count}))
    return ev


class ReviewIn(BaseModel):
    action: str
    reviewer: str = "moderator"
    note: str = ""


@app.patch("/api/v1/road-events/{eid}/review")
def review_road_event(eid: int, r: ReviewIn):
    try:
        return S.trust.review(eid, r.action, r.reviewer[:64], r.note[:500])
    except KeyError:
        raise HTTPException(404, "unknown event")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/v1/road-events")
def list_road_events(status: str = "verified", format: str = "geojson", bbox: str | None = None, since: float | None = None):
    statuses = tuple(s for s in status.split(",") if s in ("candidate", "corroborated", "verified", "rejected", "expired")) or ("verified",)
    bb = [float(x) for x in bbox.split(",")] if bbox else None
    if format == "geojson":
        return S.trust.geojson(statuses, bb, since)
    S.trust.expire()
    rows = S.store.events(f"status IN ({','.join('?' * len(statuses))})", statuses)
    return {"events": rows}


@app.get("/api/v1/road-events/{eid}")
def get_road_event(eid: int):
    e = S.store.event(eid)
    if not e:
        raise HTTPException(404, "unknown event")
    obs = S.store.observations_for(eid)
    for o in obs:
        o["source_reliability"] = round(S.trust.reliability(o["source_id"]), 3)
        o["source_id"] = o["source_id"][:4] + "…"  # pseudonymised in the UI
    e["observations"] = obs
    return e


@app.get("/api/v1/sources")
def list_sources():
    rows = S.store.query("SELECT * FROM sources ORDER BY created_at DESC LIMIT 200")
    for r in rows:
        r["reliability"] = round(r["alpha"] / (r["alpha"] + r["beta"]), 3)
    return {"sources": rows}


# ----------------------------------------------------------------------------- label review (Tashkent test set)
TASHKENT_DIR = ROOT / "datasets" / "tashkent"
DAMAGE_TYPES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def _tashkent():
    labels = json.loads((TASHKENT_DIR / "labels_test.json").read_text())
    meta = {m["id"]: m for m in json.loads((TASHKENT_DIR / "manifest.json").read_text())["images"]}
    return labels, meta


def _final_label(first: dict, review: dict | None) -> dict:
    if not review:
        return {k: first[k] for k in ("damage", "types", "uncertain", "not_road")} | {"verified_by": None}
    return {"damage": bool(review["damage"]), "types": json.loads(review["types"]), "uncertain": bool(review["uncertain"]),
            "not_road": bool(review["not_road"]), "verified_by": review["reviewer"]}


@app.get("/api/v1/labels/tashkent")
def tashkent_labels():
    labels, meta = _tashkent()
    reviews = {r["image_id"]: r for r in S.store.query("SELECT * FROM label_reviews WHERE dataset='tashkent'")}
    items = []
    for l in labels["labels"]:
        m = meta.get(l["id"], {})
        r = reviews.get(l["id"])
        items.append({"id": l["id"], "image": f"/datasets/tashkent/images/{l['id']}.jpg", "lat": m.get("lat"), "lon": m.get("lon"),
                      "captured_at": m.get("captured_at"), "source": m.get("source"),
                      "first_pass": {k: l[k] for k in ("damage", "types", "uncertain", "not_road", "conditions")},
                      "review": None if not r else {"damage": bool(r["damage"]), "types": json.loads(r["types"]), "uncertain": bool(r["uncertain"]),
                                                    "not_road": bool(r["not_road"]), "reviewer": r["reviewer"], "note": r["note"],
                                                    "agrees": bool(r["agrees"]), "updated_at": r["updated_at"]}})
    reviewed = [i for i in items if i["review"]]
    return {"protocol": labels["protocol"], "types": DAMAGE_TYPES, "items": items,
            "progress": {"total": len(items), "reviewed": len(reviewed),
                         "agreement": round(sum(i["review"]["agrees"] for i in reviewed) / len(reviewed), 3) if reviewed else None}}


class ReviewLabelIn(BaseModel):
    damage: bool
    types: list[str] = []
    uncertain: bool = False
    not_road: bool = False
    reviewer: str
    note: str = ""


@app.put("/api/v1/labels/tashkent/{image_id}")
def review_tashkent_label(image_id: str, r: ReviewLabelIn):
    labels, _ = _tashkent()
    first = next((l for l in labels["labels"] if l["id"] == image_id), None)
    if first is None:
        raise HTTPException(404, "unknown image")
    types = sorted({t for t in r.types if t in DAMAGE_TYPES}) if r.damage else []
    reviewer = re.sub(r"[^\w .@-]", "", r.reviewer)[:40] or "anonymous"
    agrees = (r.damage == first["damage"] and r.not_road == first["not_road"] and r.uncertain == first["uncertain"]
              and (not r.damage or set(types) == set(first["types"])))
    S.store.execute("INSERT OR REPLACE INTO label_reviews(dataset, image_id, damage, types, uncertain, not_road, reviewer, note, agrees, updated_at) "
                    "VALUES ('tashkent',?,?,?,?,?,?,?,?,?)", (image_id, int(r.damage), json.dumps(types), int(r.uncertain), int(r.not_road),
                                                            reviewer, r.note[:300], int(agrees), time.time()))
    return {"ok": True, "agrees": agrees}


@app.get("/api/v1/labels/tashkent/export")
def export_tashkent_labels():
    labels, _ = _tashkent()
    reviews = {r["image_id"]: r for r in S.store.query("SELECT * FROM label_reviews WHERE dataset='tashkent'")}
    out = dict(labels)
    out["labels"] = []
    for l in labels["labels"]:
        r = reviews.get(l["id"])
        f = _final_label(l, r)
        out["labels"].append({**l, **{k: f[k] for k in ("damage", "types", "uncertain", "not_road")}, "verified_by": f["verified_by"],
                              "first_pass_agrees": None if not r else bool(r["agrees"])})
    n = sum(1 for l in out["labels"] if l["verified_by"])
    out["protocol"] = {**labels["protocol"], "status": f"{n}/{len(out['labels'])} images verified by a human reviewer"}
    return JSONResponse(out, headers={"Content-Disposition": "attachment; filename=labels_test.json"})


# ----------------------------------------------------------------------------- ChargePulse
@app.post("/api/v1/chargepulse/analyze")
async def chargepulse_analyze(file: UploadFile = File(...), bays: str = Form(...), site: str = Form("demo-site"),
                              sample_fps: float = Form(2.0), max_seconds: float = Form(120.0)):
    """bays: JSON {bay_id: [[x,y], ...]} in the image/video pixel coordinates."""
    S.load()
    try:
        bay_polys = {str(k): v for k, v in json.loads(bays).items()}
        assert bay_polys and all(len(v) >= 3 for v in bay_polys.values())
    except Exception:
        raise HTTPException(400, "bays must be a JSON object {bay_id: [[x,y],[x,y],[x,y],...]}")
    det = S.detectors["road_users"]
    site = _source_id(site)
    name = (file.filename or "").lower()
    data = await _read_upload(file, MAX_VIDEO_BYTES)
    is_video = (file.content_type or "").startswith("video") or name.endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))

    def store_events(cp: ChargePulse):
        for ev in cp.events:
            S.store.execute("INSERT INTO charger_observations(site, bay, state, conf, vehicle_cls, observed_at, source_id) "
                            "VALUES (?,?,?,?,?,?,?)", (site, ev["bay"], ev["state"], None, ev["vehicle_cls"],
                                                      time.time(), "chargepulse-camera"))

    if not is_video:
        img, _ = _decode_image(data)
        h, w = img.shape[:2]
        cp = ChargePulse(bay_polys, w, h, hold=1)
        d = det(img)
        states = cp.update(d, 0.0)
        store_events(cp)
        return {"kind": "image", "bays": states, "vehicles": d.to_list(),
                "note": "visual occupancy only - charging state requires operator telemetry (OCPP)"}
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(name).suffix or '.mp4'}"
    path.write_bytes(data)

    def work(job: Job):
        try:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, int(round(fps / np.clip(sample_fps, 0.5, 5))))
            cp = ChargePulse(bay_polys, w, h)
            idx = 0
            while idx < max_seconds * fps:
                if not cap.grab():
                    break
                if idx % step == 0:
                    ok, img = cap.retrieve()
                    if not ok:
                        break
                    d = det(img)
                    st = cp.update(d, idx / fps)
                    job.progress = min(0.99, idx / total) if total else 0.5
                    job.emit({"type": "frame", "t": round(idx / fps, 2), "bays": st, "vehicles": d.to_list()})
                idx += 1
            cap.release()
            store_events(cp)
            return {"kind": "video", "summary": cp.summary(), "events": cp.events,
                    "note": "visual occupancy only - charging state requires operator telemetry (OCPP)"}
        finally:
            path.unlink(missing_ok=True)

    job = _submit("chargepulse", work)
    return {"job_id": job.id}


@app.get("/api/v1/charger-observations")
def charger_observations(site: str | None = None, limit: int = 200):
    if site:
        rows = S.store.query("SELECT * FROM charger_observations WHERE site=? ORDER BY observed_at DESC LIMIT ?", (site, limit))
    else:
        rows = S.store.query("SELECT * FROM charger_observations ORDER BY observed_at DESC LIMIT ?", (limit,))
    return {"observations": rows, "semantics": "state = visual occupancy of the bay, not charging status"}


# ----------------------------------------------------------------------------- static
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
if (ROOT / "datasets" / "tashkent" / "images").exists():
    app.mount("/datasets/tashkent/images", StaticFiles(directory=str(ROOT / "datasets" / "tashkent" / "images")), name="tashkent")
@app.get("/demo/v2xsim/ortho.jpg", include_in_schema=False)
def demo_ortho():
    p = DEMO_DIR / "v2xsim" / "ortho.jpg"
    if not p.exists():
        raise HTTPException(404, "orthophoto not built")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


if (DEMO_DIR / "v2xsim" / "frames").exists():
    app.mount("/demo/v2xsim/frames", StaticFiles(directory=str(DEMO_DIR / "v2xsim" / "frames")), name="demo")
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
