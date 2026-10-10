"""Collect street-level images of Tashkent from Mapillary (CC BY-SA 4.0) for a local dataset.

Mapillary blurs faces and licence plates on its side before serving images.  We sample a grid of
tiles over the city, keep perspective (non-panoramic) images, prefer recent captures and take only a
few frames per capture sequence, spread out, so the set covers many streets rather than one drive.

The access token is read from the MAPILLARY_TOKEN environment variable and never stored in the repo.

Usage: MAPILLARY_TOKEN=... python training/fetch_mapillary.py --out data/tashkent --per-tile 25
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://graph.mapillary.com/images"
FIELDS = "id,captured_at,is_pano,camera_type,thumb_1024_url,geometry,sequence,make,model,width,height,compass_angle"
# Tashkent city bounding box (lon_min, lat_min, lon_max, lat_max)
TASHKENT = (69.15, 41.23, 69.40, 41.40)


def tiles(bbox, step):
    x0, y0, x1, y1 = bbox
    x = x0
    while x < x1:
        y = y0
        while y < y1:
            yield (round(x, 4), round(y, 4), round(min(x + step, x1), 4), round(min(y + step, y1), 4))
            y += step
        x += step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tashkent")
    ap.add_argument("--step", type=float, default=0.02, help="tile size in degrees (~1.7 x 2.2 km)")
    ap.add_argument("--per-tile", type=int, default=25)
    ap.add_argument("--per-sequence", type=int, default=4)
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    token = os.environ["MAPILLARY_TOKEN"]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    min_ts = (args.min_year - 1970) * 365.25 * 86400 * 1000
    chosen = []
    for bb in tiles(TASHKENT, args.step):
        try:
            r = requests.get(API, params={"access_token": token, "fields": FIELDS, "bbox": ",".join(map(str, bb)), "limit": 500}, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print("tile failed", bb, e)
            continue
        imgs = [d for d in r.json().get("data", []) if not d.get("is_pano") and d.get("camera_type", "perspective") in ("perspective", "fisheye", None)
                and d.get("captured_at", 0) >= min_ts and d.get("thumb_1024_url")]
        by_seq = defaultdict(list)
        for d in imgs:
            by_seq[d.get("sequence")].append(d)
        pick = []
        for seq, lst in by_seq.items():
            lst.sort(key=lambda d: d["captured_at"])
            k = min(args.per_sequence, len(lst))
            idx = [round(i * (len(lst) - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
            pick += [lst[i] for i in sorted(set(idx))]
        rng.shuffle(pick)
        pick = pick[: args.per_tile]
        for d in pick:
            d["tile"] = "%.3f_%.3f" % (bb[0], bb[1])
        chosen += pick
        print(f"tile {bb}: {len(imgs)} images, {len(by_seq)} sequences -> {len(pick)}", flush=True)
        time.sleep(0.2)

    def get(d):
        p = out / "images" / f"{d['id']}.jpg"
        if p.exists():
            return True
        try:
            r = requests.get(d["thumb_1024_url"], timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
            return True
        except requests.RequestException:
            return False

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(get, chosen))
    meta = []
    for d, good in zip(chosen, ok):
        if not good:
            continue
        lon, lat = d["geometry"]["coordinates"]
        meta.append({"id": d["id"], "file": f"images/{d['id']}.jpg", "lat": lat, "lon": lon, "captured_at": d["captured_at"] / 1000,
                     "sequence": d.get("sequence"), "tile": d["tile"], "device": f"{d.get('make', '')} {d.get('model', '')}".strip(),
                     "compass": d.get("compass_angle"), "source": f"https://www.mapillary.com/app/?pKey={d['id']}"})
    (out / "meta.json").write_text(json.dumps({"license": "CC BY-SA 4.0", "attribution": "© Mapillary contributors",
                                               "collected": time.strftime("%Y-%m-%d"), "bbox": TASHKENT, "images": meta}, indent=1))
    print(len(meta), "images saved to", out)


if __name__ == "__main__":
    main()
