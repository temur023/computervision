"""Resize the RDD2022 subset to a training resolution and write an Ultralytics dataset YAML.

Labels are normalised YOLO boxes, so resizing images keeps them valid.  Pre-resizing matters on
CPU: Norway images are 3650x2044 and would dominate data-loading time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing data/images/{train,valid,test}")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-side", type=int, default=800)
    args = ap.parse_args()
    src, dst = Path(args.src) / "data", Path(args.dst)
    jobs = []
    for split, out_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        for img in sorted((src / "images" / split).rglob("*.jpg")):
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            jobs.append((img, lbl, dst / "images" / out_split / img.name, dst / "labels" / out_split / (img.stem + ".txt")))

    def work(j):
        img, lbl, oimg, olbl = j
        oimg.parent.mkdir(parents=True, exist_ok=True)
        olbl.parent.mkdir(parents=True, exist_ok=True)
        if not oimg.exists():
            im = cv2.imread(str(img))
            if im is None:
                return 0
            h, w = im.shape[:2]
            s = args.max_side / max(h, w)
            if s < 1:
                im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(oimg), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        olbl.write_text(lbl.read_text() if lbl.exists() else "")
        return 1

    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(work, jobs))
    yaml = dst / "rdd2022_subset.yaml"
    yaml.write_text(
        f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + "names:\n" + "".join(f"  {i}: {n_}\n" for i, n_ in enumerate(NAMES))
    )
    print(n, "images ->", dst, "yaml:", yaml)


if __name__ == "__main__":
    main()
