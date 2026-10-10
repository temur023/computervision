"""Image-level evaluation of the deployed road-damage detector on our Tashkent test set.

Each labelled image is scored by the highest damage confidence the detector reports.  We report the
image-level average precision (ranking quality), and precision / recall at the threshold the product
uses to create a road-event candidate.  Images labelled "not road" or "uncertain" are excluded from
the headline numbers; the false-alarm rate on "not road" images is reported separately, because a
crowdsourced feed contains them.

Usage: python training/eval_tashkent.py --images data/tashkent/images [--model models/road_damage_yolo11n.onnx]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swarmsight.detector import YoloOnnxDetector


def average_precision(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="original (non-anonymised) images, e.g. data/tashkent/images")
    ap.add_argument("--labels", default="datasets/tashkent/labels_test.json")
    ap.add_argument("--model", default="models/road_damage_yolo11n.onnx")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out", default="benchmarks/results/tashkent_eval.json")
    args = ap.parse_args()
    det = YoloOnnxDetector(args.model, conf=0.01)
    lab = json.loads(Path(args.labels).read_text())["labels"]
    rows = []
    for l in lab:
        img = cv2.imread(str(Path(args.images) / f"{l['id']}.jpg"))
        d = det(img)
        score = float(d.conf.max()) if len(d) else 0.0
        types = {d.label(i) for i in range(len(d)) if d.conf[i] >= args.threshold}
        rows.append({**l, "score": score, "pred_types": sorted(types)})
    ev = [r for r in rows if not r["not_road"] and not r["uncertain"]]
    y = [int(r["damage"]) for r in ev]
    s = [r["score"] for r in ev]
    pred = [sc >= args.threshold for sc in s]
    tp = sum(p and t for p, t in zip(pred, y))
    fp = sum(p and not t for p, t in zip(pred, y))
    fn = sum((not p) and t for p, t in zip(pred, y))
    pothole_imgs = [r for r in ev if "pothole" in r["types"]]
    not_road = [r for r in rows if r["not_road"]]
    uncertain = [r for r in rows if r["uncertain"] and not r["not_road"]]
    out = {
        "n_labelled": len(rows), "n_images": len(ev), "n_positive": int(sum(y)),
        "ap_image": average_precision(s, y),
        "threshold": args.threshold,
        "precision_at_thr": tp / max(1, tp + fp), "recall_at_thr": tp / max(1, tp + fn),
        "pothole_images": len(pothole_imgs),
        "pothole_image_recall": (sum(1 for r in pothole_imgs if r["score"] >= args.threshold) / len(pothole_imgs)) if pothole_imgs else None,
        "not_road_images": len(not_road),
        "not_road_false_alarm_rate": (sum(1 for r in not_road if r["score"] >= args.threshold) / len(not_road)) if not_road else None,
        "uncertain_images_flagged": sum(1 for r in uncertain if r["score"] >= args.threshold), "uncertain_images": len(uncertain),
        "model": Path(args.model).name,
        "labels": "image-level, first pass by Claude (LLM vision) without model pre-labels; human verification pending",
        "per_image": [{"id": r["id"], "damage": r["damage"], "types": r["types"], "score": round(r["score"], 3),
                       "pred_types": r["pred_types"], "excluded": r["not_road"] or r["uncertain"]} for r in rows],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "per_image"})


if __name__ == "__main__":
    main()
