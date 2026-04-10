"""
BONUS — Model Evaluator
========================
Run after training to see exactly how accurate your model is.
Prints mAP, precision, recall and shows example detections.

Usage:
    python evaluate_model.py --weights runs/egg_counter/weights/best.pt --data dataset/data.yaml

Requirements:
    pip install ultralytics
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print("\nRunning validation on test set...")
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )

    print("\n" + "="*50)
    print("  MODEL ACCURACY REPORT")
    print("="*50)
    print(f"  mAP@50         : {metrics.box.map50:.3f}   (target: >0.85)")
    print(f"  mAP@50-95      : {metrics.box.map:.3f}")
    print(f"  Precision      : {metrics.box.mp:.3f}   (how often detections are correct)")
    print(f"  Recall         : {metrics.box.mr:.3f}   (how many eggs are found)")
    print("="*50)

    map50 = metrics.box.map50
    if map50 >= 0.90:
        print("\n  Excellent — model is production ready!")
    elif map50 >= 0.80:
        print("\n  Good — consider labeling 50 more images to push higher.")
    elif map50 >= 0.65:
        print("\n  Moderate — label more images, especially of missed cases.")
    else:
        print("\n  Needs improvement — label more diverse images (varied lighting, angles).")

    print(f"\n  Plots saved in the model's run directory.")


if __name__ == "__main__":
    main()
