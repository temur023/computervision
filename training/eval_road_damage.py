"""Evaluate the exported road-damage detector (the ONNX file we deploy) on the RDD2022 held-out test split.

Reports overall and per-class AP, per-country mAP@0.5 and robustness under synthetic perturbations of the
same test images (darkening + sensor noise as "night", streaks + haze as "rain", motion blur, heavy JPEG).
Synthetic perturbations are a stress test, not a substitute for real night or rain footage.

Usage: python training/eval_road_damage.py --data data/rdd_800 --model models/road_damage_yolo11n.onnx
"""
import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def perturb(img, kind, rng):
    if kind == "night (gamma 2.2, -60% light, noise)":
        x = (img.astype(np.float32) / 255.0) ** 2.2 * 0.4
        x = x + rng.normal(0, 0.03, x.shape)
        x[..., 0] *= 1.15  # bluish sodium-free LED tint
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if kind == "rain (streaks + haze)":
        h, w = img.shape[:2]
        streak = np.zeros((h, w), np.float32)
        n = int(h * w / 600)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        streak[ys, xs] = 1.0
        k = np.zeros((15, 15), np.float32)
        np.fill_diagonal(np.fliplr(k), 1.0)
        streak = cv2.filter2D(streak, -1, k / 15.0)
        x = img.astype(np.float32) * 0.75 + 60 + streak[..., None] * 160
        return np.clip(cv2.GaussianBlur(x, (0, 0), 1.2), 0, 255).astype(np.uint8)
    if kind == "motion blur (9 px)":
        k = np.zeros((9, 9), np.float32)
        k[4, :] = 1.0 / 9
        return cv2.filter2D(img, -1, k)
    if kind == "low-quality JPEG (q=15)":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def write_yaml(path: Path, root: Path, test_list: Path):
    path.write_text(f"path: {root}\ntrain: {test_list}\nval: {test_list}\ntest: {test_list}\n"
                    + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))


def run_val(model_path, yaml_path, imgsz):
    m = YOLO(str(model_path), task="detect")
    r = m.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=1, device="cpu", plots=False, verbose=False, conf=0.001, iou=0.6)
    per = {}
    for i, n in enumerate(NAMES):
        if i in r.box.ap_class_index:
            j = list(r.box.ap_class_index).index(i)
            per[n] = {"ap50": float(r.box.ap50[j]), "ap": float(r.box.ap[j])}
    return {"map50": float(r.box.map50), "map": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr), "per_class": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output of prepare_rdd.py")
    ap.add_argument("--model", default="models/road_damage_yolo11n.onnx")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", default="benchmarks/results/road_damage_eval.json")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    root = Path(args.data).resolve()
    test_imgs = sorted((root / "images" / "test").glob("*.jpg"))
    work = Path(tempfile.mkdtemp(prefix="rdd_eval_"))
    lst = work / "test.txt"
    lst.write_text("\n".join(map(str, test_imgs)))
    yml = work / "test.yaml"
    write_yaml(yml, root, lst)
    res = {"model": Path(args.model).name, "imgsz": args.imgsz, "n_test_images": len(test_imgs)}
    res["test"] = run_val(args.model, yml, args.imgsz)
    print("test", json.dumps(res["test"]))

    by_country = defaultdict(list)
    for p in test_imgs:
        by_country[re.sub(r"_\d+$", "", p.stem)].append(p)
    res["countries"] = sorted(by_country)
    res["per_country"] = {}
    for c, imgs in sorted(by_country.items()):
        l2 = work / f"{c}.txt"
        l2.write_text("\n".join(map(str, imgs)))
        y2 = work / f"{c}.yaml"
        write_yaml(y2, root, l2)
        r = run_val(args.model, y2, args.imgsz)
        res["per_country"][c] = {"n": len(imgs), "map50": r["map50"]}
        print(c, len(imgs), round(r["map50"], 3))

    if not args.skip_robustness:
        res["robustness"] = {}
        rng = np.random.default_rng(0)
        for kind in ["night (gamma 2.2, -60% light, noise)", "rain (streaks + haze)", "motion blur (9 px)", "low-quality JPEG (q=15)"]:
            d = work / re.sub(r"\W+", "_", kind)
            (d / "images" / "test").mkdir(parents=True)
            (d / "labels" / "test").mkdir(parents=True)
            for p in test_imgs:
                cv2.imwrite(str(d / "images" / "test" / p.name), perturb(cv2.imread(str(p)), kind, rng))
                shutil.copy(root / "labels" / "test" / (p.stem + ".txt"), d / "labels" / "test" / (p.stem + ".txt"))
            l3 = d / "test.txt"
            l3.write_text("\n".join(str(x) for x in sorted((d / "images" / "test").glob("*.jpg"))))
            y3 = d / "test.yaml"
            write_yaml(y3, d, l3)
            r = run_val(args.model, y3, args.imgsz)
            res["robustness"][kind] = {"map50": r["map50"]}
            print(kind, round(r["map50"], 3))
    res["note"] = ("Held-out test split of the RDD2022 HF export (never used for training or model selection). "
                   "AP computed by Ultralytics val on the exported ONNX model, conf 0.001, IoU 0.6. "
                   "Robustness rows apply synthetic perturbations to the same images.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
