"""Download a reproducible, country-stratified subset of RDD2022 (CC BY-SA 4.0).

Source: https://huggingface.co/datasets/dronefreak/RDD2022 - a 4-class (D00, D10, D20, D40) YOLO export
of Arya et al., "RDD2022: A multi-national image dataset for automatic Road Damage Detection" (2022).

The mirror ships a fixed train/valid/test split; we sample *within* each split, so no test image is
ever used for training.  Shards are sorted by country, so we sample per country (seeded) instead of
taking the first shards.  China_Drone (aerial imagery) is excluded: the product uses vehicle cameras.
"""
import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = "dronefreak/RDD2022"
EXCLUDE = {"China_Drone"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rdd2022")
    ap.add_argument("--train-per-country", type=int, default=1000)
    ap.add_argument("--valid-per-country", type=int, default=120)
    ap.add_argument("--test-per-country", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reuse", default=None, help="existing local mirror to hard-link files from")
    args = ap.parse_args()

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".jpg")]
    by = defaultdict(list)
    for f in files:
        m = re.match(r"data/images/(train|valid|test)/shard_\d+/(.+)_\d+\.jpg", f)
        if m and m.group(2) not in EXCLUDE:
            by[(m.group(1), m.group(2))].append(f)
    rng = random.Random(args.seed)
    quota = {"train": args.train_per_country, "valid": args.valid_per_country, "test": args.test_per_country}
    chosen = []
    for (split, country), lst in sorted(by.items()):
        lst = sorted(lst)
        rng.shuffle(lst)
        pick = lst[: quota[split]]
        chosen += pick
        print(f"{split:5s} {country:16s} {len(pick):5d} / {len(lst)}", flush=True)
    wanted = ["data/data.yaml"] + chosen + [c.replace("/images/", "/labels/").replace(".jpg", ".txt") for c in chosen]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "subset_files.txt").write_text("\n".join(sorted(chosen)))

    def get(f):
        dst = Path(args.out) / f
        if dst.exists():
            return 0
        if args.reuse and (Path(args.reuse) / f).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(Path(args.reuse) / f, dst)
            return 0
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.out)
            return 1
        except Exception as e:  # empty label files may be absent for background images
            print("skip", f, type(e).__name__, flush=True)
            return 0

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(get, wanted)):
            if i % 1000 == 0:
                print(f"{i}/{len(wanted)}", flush=True)
    for split in ("train", "valid", "test"):
        n = len(list(Path(args.out, "data/images", split).rglob("*.jpg")))
        print(split, n, "images on disk", flush=True)


if __name__ == "__main__":
    main()
