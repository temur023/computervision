# SwarmSight

**Collective vision for connected mobility.** SwarmSight joins the cameras already on the road (taxis, fleets,
intersections) into one perception network. It tells a camera which road users are hidden from it and what is hiding them.
It turns road damage seen in ordinary phone and dashcam footage into verified, deduplicated road events that a city or a
map provider can act on.

Mode: **Startup** · Track: Computer Vision · Partners in scope: Yandex (map and fleet channel), Toyota / BYD Royal Motors and Ducati (fleets, riders, service intake)

| | |
|---|---|
| Live prototype | `https://huggingface.co/spaces/<org>/swarmsight` (deployed from this repository by GitHub Actions, see [Deploy](#deploy)) |
| Defence deck | Slides artifact, exported to PPTX/PDF for submission (`docs/` once exported) |
| Data sources and licences | [below](#data-sources-and-licences) and [`docs/data_registry.json`](docs/data_registry.json) |

---

## What it does

| Module | Input | Output | Stage |
|---|---|---|---|
| **01 Guardian**: cooperative perception | frames from 2+ cameras that see the same area | fused road users on a metric ground plane; for each camera: *visible / occluded by what / missed / out of view*, alerts with source, age, TTL and uncertainty | research core |
| **02 Road Intelligence**: verified road events | any road photo or video, or a live phone camera | potholes and cracks, tracked across frames, then merged by the trust engine into events: candidate → corroborated → verified, served to partners as GeoJSON | first revenue, pilot |
| **03 ChargePulse**: EV bay occupancy | a camera over a charging area, plus bay polygons | free / occupied per bay with timestamps and utilisation. Visual occupancy only, never "charging" | add-on |

Every input is processed live on our server (ONNX Runtime, CPU). Nothing is pre-computed or hard-coded.

## Measured results

All numbers come from scripts in this repository, on data the models did not see during training. Raw outputs are in
[`benchmarks/results/`](benchmarks/results) and are rendered on the site under **Data & evidence**.

**Guardian, V2X-Sim 2.0 (CARLA), ego vehicles 1/2/4, 285 ego-frames, range 40 m** · `python -m benchmarks.coop_v2xsim`

| Metric | Own camera | + road-side unit | + RSU + other vehicles |
|---|---:|---:|---:|
| Recall of road users in the field of view | 59.6 % | 82.3 % | 90.2 % |
| Recall of road users **occluded** for the ego camera | 0 % | 61.6 % | 83.2 % |
| … of which vulnerable (bicycles, motorcycles) | 0 % | 78.9 % | 94.4 % |
| Alert precision (alert matches a real object) | – | 76.4 % | 79.3 % |
| Median position error of fused objects | 0.85 m | 0.53 m | 0.56 m |

"Occluded" ground truth is computed by us per camera from the dataset's semantic segmentation and depth maps
(fewer than 400 depth-consistent object pixels visible). That label is not part of the dataset.

**Road damage**: the RDD2022 held-out test split (6 countries) and our Tashkent test set (image level). See
`benchmarks/results/road_damage_eval.json` and `benchmarks/results/tashkent_eval.json`. The site shows the current
values with per-class, per-country and robustness breakdowns.

**Speed and cost**: `benchmarks/results/speed_cost.json` (per stage p50/p95, $ per 1000 frames, per camera-month, per km).

## Architecture

```
camera frame ─► detect (YOLO11n road users · our YOLO11n road damage, ONNX Runtime)
             ─► compose riders (person + bicycle/motorcycle → cyclist/motorcyclist)
             ─► track (ByteTrack: Kalman + two-stage IoU, class groups)
     ┌───────────────────────────────┴────────────────────────────────┐
 Guardian (ms)                                               Road Intelligence (days)
 ground point + covariance (homography Jacobian)             severity = type × size × proximity × conf
 associate: sticky bindings → gated Hungarian (χ² 99%)       observation per damage type, anonymised evidence
 merge duplicates → Kalman filter in metres                  trust engine: merge radius, noisy-OR over sources,
 per-camera reasoning: visible / occluded-by / missed        Beta(α,β) source reliability, lifecycle + TTL
 alerts (TTL, age, ±m) over WebSocket                        moderation → partner GeoJSON API
```

| Path | What |
|---|---|
| `swarmsight/detector.py` | ONNX Runtime YOLO inference (letterbox, NMS), no PyTorch at runtime |
| `swarmsight/tracker.py` | ByteTrack-style tracker, monocular time-to-contact from box expansion |
| `swarmsight/geometry.py` | ground homographies, focal length and pose from 4 clicked points, uncertainty propagation, ghost boxes |
| `swarmsight/guardian.py` | cooperative fusion and occlusion reasoning |
| `swarmsight/trust.py`, `store.py` | road-event trust engine and SQLite persistence |
| `swarmsight/pipeline.py` | single-camera Scan pipeline (image, video, live) |
| `swarmsight/chargepulse.py` | EV bay occupancy with hysteresis |
| `swarmsight/privacy.py` | anonymisation of stored evidence |
| `swarmsight/server.py` | FastAPI: REST + WebSocket (jobs, live camera, Guardian rooms), static web UI |
| `web/` | the web platform (vanilla ES modules, no build step) |
| `benchmarks/` | V2X-Sim benchmark, demo scene export, speed and cost |
| `training/` | data collection, anonymisation, training and evaluation scripts |

**Where inference runs.** Prototype: our server, CPU only (ONNX Runtime), so any phone or laptop works through the
browser. Pilot: Road Intelligence stays server-side, uploading metadata and short evidence crops instead of video.
Guardian's first deployment is fixed cameras at blind crossings with an edge box per site. Alerts for moving vehicles
would need on-vehicle compute, GNSS/IMU pose and certification, and the prototype never claims that.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn swarmsight.server:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

Or with Docker: `docker build -t swarmsight . && docker run -p 7860:7860 swarmsight`.

Tests: `pip install pytest httpx && python -m pytest -q`.

The models are in `models/` (`yolo11n_coco.onnx`, `road_damage_yolo11n.onnx`). The demo scene is in `demo/v2xsim/`, and
the anonymised Tashkent test images are in `datasets/tashkent/`.

### Try it as a jury member

* **Any road photo or video**: *Road Intelligence → Scan*. Upload, or use *Live camera* on a phone. Add a location
  (photo GPS is read automatically, or use device GPS, or click the map) to create a road event, then see it under
  *Events & moderation*.
* **Two views of one scene**: *Guardian → Your two videos*. Tap the same 4 ground points in both, enter the rectangle
  size and fuse. *Guardian → Two phones, live* does the same with two phones in one room.
* **A parking or charging area**: *ChargePulse*. Draw the bays and analyse.

## Reproduce the data, models and numbers

```bash
pip install -r requirements-dev.txt

# Road damage: data → training → evaluation
python training/fetch_rdd2022.py --out data/rdd2022                # stratified 6-country subset (CC BY-SA 4.0)
python training/prepare_rdd.py --src data/rdd2022 --dst data/rdd_800
python training/train_road_damage.py --data data/rdd_800/rdd2022_subset.yaml --epochs 12 --imgsz 512
python training/eval_road_damage.py --data data/rdd_800 --model models/road_damage_yolo11n.onnx

# Tashkent local test set (needs a free Mapillary token)
MAPILLARY_TOKEN=... python training/fetch_mapillary.py --out data/tashkent
python training/tashkent_split.py --src data/tashkent --out datasets/tashkent --test-n 200 --train-n 400
python training/eval_tashkent.py --images data/tashkent/images

# Guardian benchmark and demo scene (V2X-Sim 2.0 mini, Apache-2.0, ~6 GB)
python -m benchmarks.coop_v2xsim --root data/v2x_sim_mini
python -m benchmarks.export_v2xsim_demo --root data/v2x_sim_mini

# Speed and cost on your hardware
python -m benchmarks.speed_cost --rdd data/rdd_800/images/test
```

## Deploy

`.github/workflows/deploy.yml` runs the tests, then uploads the service to a Hugging Face Space (Docker SDK). Each
upload's commit message names the source commit (`Deploy owner/repo@sha`), so the deployment history is recorded on both
sides. Setup:

1. Create a Hugging Face write token, then add it in GitHub → Settings → Secrets → Actions as `HF_TOKEN`.
2. Add a repository variable `HF_SPACE` = `your-user/swarmsight`.
3. Push to `main`, or run the workflow manually. The Space builds `Dockerfile` and serves on port 7860 over HTTPS, which
   phones need for camera access.

Any other container host works too: build the `Dockerfile`, expose port 7860 and put it behind HTTPS.

## API

Interactive reference at `/docs`. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/scan/image`, `/api/v1/scan/video` | analyse an upload (video as a job) |
| WS | `/ws/live/scan` | live phone camera: JPEG frames in, detections out |
| WS | `/ws/jobs/{id}` | progress and per-frame results of a job |
| POST | `/api/v1/guardian/calibrate` | 4-point calibration → focal, height, tilt, field-of-view polygon |
| POST | `/api/v1/guardian/demo/run`, `/api/v1/guardian/videos` | cooperative perception on the demo scene / two uploaded videos |
| WS | `/ws/guardian/{room}` | live multi-phone room |
| POST | `/api/v1/observations` | object observation from a partner device (concept appendix A1) |
| GET | `/api/v1/objects/nearby` | external objects for a subscriber camera |
| POST | `/api/v1/road-events` | device or manual road-event observation |
| PATCH | `/api/v1/road-events/{id}/review` | moderator verify / reject |
| GET | `/api/v1/road-events?status=verified&format=geojson` | partner feed |
| POST/GET | `/api/v1/chargepulse/analyze`, `/api/v1/charger-observations` | bay occupancy |
| GET/PUT | `/api/v1/labels/tashkent[/{id}]`, `/api/v1/labels/tashkent/export` | human review of our Tashkent labels |
| GET | `/api/v1/health`, `/api/v1/metrics`, `/api/v1/data` | live latency, benchmark results, dataset registry |

## Data sources and licences

| Source | Used for | Licence | Notes |
|---|---|---|---|
| [RDD2022](https://github.com/sekilab/RoadDamageDetector), Arya et al. 2022, via the [Hugging Face YOLO export](https://huggingface.co/datasets/dronefreak/RDD2022) | training (6,000), validation (720) and held-out test (1,500) of the road-damage detector | CC BY-SA 4.0 | Japan, India, Czech Republic, Norway, USA, China (motorbike); 1000/120/250 per country, seeded; aerial subset excluded |
| [Mapillary](https://www.mapillary.com) street imagery of Tashkent (Graph API v4) | our local test set (and local fine-tuning) | CC BY-SA 4.0, © Mapillary contributors | 1,449 images collected; 200 test images from 31 held-out map tiles are redistributed in `datasets/tashkent/images` **after we blurred every person and vehicle** (`training/anonymize_dir.py`). Our labels are CC BY-SA 4.0 |
| [V2X-Sim 2.0 mini](https://huggingface.co/datasets/ai4ce-drive/V2X-Sim-2.0-mini), Li et al., IEEE RA-L 2022 | Guardian benchmark and the bundled demo scene (60 frames × 6 cameras, downscaled) | Apache-2.0 | synthetic (CARLA), so it contains no real people |
| [Ultralytics YOLO11n](https://github.com/ultralytics/ultralytics) COCO weights | road-user detector; starting point for our road-damage model | AGPL-3.0 | hence this repository is AGPL-3.0; a commercial deployment needs an Ultralytics licence or an Apache-2.0 detector |
| [Leaflet](https://leafletjs.com) 1.9.4 | maps | BSD-2-Clause | vendored in `web/vendor/leaflet` |
| [Geist](https://vercel.com/font) fonts | typography | SIL OFL 1.1 | vendored in `web/vendor/geist` |
| OpenStreetMap tiles | map background | ODbL, © OpenStreetMap contributors | |
| Python libraries | FastAPI, Uvicorn, NumPy, SciPy, OpenCV, ONNX Runtime, Pillow, PyTorch (training only) | MIT / BSD / Apache-2.0 | see `requirements*.txt` |

**Labelling.** The Tashkent test labels are image level: whether road damage is visible on the drivable surface ahead,
which RDD2022 types, plus "not a road" and "uncertain" flags. The first pass on 120 images was done by Claude (an AI
assistant, reviewing images visually) **before any model prediction was seen**, so the test is not biased toward our
model. The team verifies these labels in the platform (*Road Intelligence → Label review*), and agreement with the first
pass is tracked and exported with the labels.

## Privacy and safety

* No customer records and no non-public data. Street imagery comes from public, licensed sources. Mapillary blurs faces
  and plates, but registration numbers painted on truck bodies and some faces survive, so we blur every person and
  vehicle in images we publish. Evidence snapshots created by the service are blurred (heads, plate band) before storage.
* The partner feed exposes events, never device identities or tracks.
* Guardian is advisory perception sharing for research and pilots. It is never connected to braking or steering.

## Known limitations

* Guardian is benchmarked on a synthetic scene. Real-world transfer is the first pilot milestone, with fixed cameras.
* Moving cameras need accurate pose (GNSS + IMU) before fusion. The live demo uses static phones.
* Crowdsourced input is noisy: about 9 % of the Tashkent images were not road scenes, and a quarter were too dark,
  blurred or wet to judge. A road-scene gate and night data are on the roadmap.
* Synthetic night and rain perturbations are a stress test, not a substitute for real night and rain footage.

## How this was built

Built during the final by the SwarmSight team, with Claude Code (Anthropic) as the coding assistant: code, the Tashkent
first-pass labels and documentation. The concept and the product decisions are the team's. Third-party models, data and
libraries are listed above with their licences.

## Licence

GNU Affero General Public License v3.0, see [LICENSE](LICENSE). Data keeps its own licence (see above).
