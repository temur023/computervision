"""Split the Tashkent Mapillary collection into train / test by map tile and write a manifest.

Splitting by tile (≈1.7 x 2.2 km) keeps the same street out of both splits, so the local test
measures generalisation to unseen streets, not memorised ones.  The test sample is drawn at random
from test tiles (not selected by any model), and is labelled without model pre-labels.

Usage: python training/tashkent_split.py --src data/tashkent --out datasets/tashkent --test-n 200 --train-n 400
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="output dir of fetch_mapillary.py")
    ap.add_argument("--out", default="datasets/tashkent")
    ap.add_argument("--test-tiles", type=float, default=0.4)
    ap.add_argument("--test-n", type=int, default=200)
    ap.add_argument("--train-n", type=int, default=400)
    ap.add_argument("--copy-test-images", action="store_true", help="copy (re-encoded) test images into the repo")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    meta = json.loads((src / "meta.json").read_text())
    imgs = meta["images"]
    by_tile = defaultdict(list)
    for m in imgs:
        by_tile[m["tile"]].append(m)
    tiles = sorted(by_tile)
    rng = random.Random(args.seed)
    rng.shuffle(tiles)
    n_test = round(len(tiles) * args.test_tiles)
    test_tiles = set(tiles[:n_test])
    test_pool = [m for t in tiles if t in test_tiles for m in by_tile[t]]
    train_pool = [m for t in tiles if t not in test_tiles for m in by_tile[t]]
    rng.shuffle(test_pool)
    rng.shuffle(train_pool)
    test = sorted(test_pool[: args.test_n], key=lambda m: m["id"])
    train = sorted(train_pool[: args.train_n], key=lambda m: m["id"])
    for m in test:
        m["split"] = "test"
    for m in train:
        m["split"] = "train"
    out.mkdir(parents=True, exist_ok=True)
    if args.copy_test_images:
        (out / "images").mkdir(exist_ok=True)
        for m in test:
            im = cv2.imread(str(src / m["file"]))
            cv2.imwrite(str(out / "images" / f"{m['id']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    manifest = {
        "license": meta["license"], "attribution": meta["attribution"], "collected": meta["collected"],
        "split_rule": f"by map tile (0.02 deg grid); {n_test}/{len(tiles)} tiles held out for test; random sample inside each split",
        "tiles": {"test": sorted(test_tiles), "train": sorted(set(tiles) - test_tiles)},
        "images": test + train,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"tiles: {len(tiles)} ({n_test} test) · test pool {len(test_pool)} · train pool {len(train_pool)}")
    print(f"selected: test {len(test)} · train {len(train)} -> {out}")


if __name__ == "__main__":
    main()
