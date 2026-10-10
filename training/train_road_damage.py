"""Fine-tune YOLO11n (COCO-pretrained) on the RDD2022 subset, on CPU, and export to ONNX.

Example:
  python training/train_road_damage.py --data data/rdd_800/rdd2022_subset.yaml --epochs 30 --imgsz 640
Outputs: runs/road_damage/<name>/weights/best.pt and models/road_damage_yolo11n.onnx
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="yolo11n_rdd")
    ap.add_argument("--project", default="runs/road_damage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="models/road_damage_yolo11n.onnx")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(str(Path(args.project) / args.name / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu",
            workers=args.workers, project=args.project, name=args.name, exist_ok=True, seed=0,
            fraction=args.fraction, patience=50, cos_lr=True, close_mosaic=3, plots=True,
            # road damage is orientation-dependent (longitudinal vs transverse): no vertical flips/rotations
            fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0, hsv_v=0.5, hsv_s=0.6,
        )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    m = YOLO(str(best))
    onnx = m.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17, dynamic=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, args.out)
    print("exported", args.out)


if __name__ == "__main__":
    main()
