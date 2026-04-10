"""
STEP 3 — YOLOv8 Training
=========================
Trains YOLOv8 on your labeled egg dataset.
Automatically uses GPU if available, falls back to CPU.

Usage:
    python step3_train.py --data dataset/data.yaml
    python step3_train.py --data dataset/data.yaml --epochs 100 --model yolov8n.pt
    python step3_train.py --data dataset/data.yaml --resume  # resume interrupted training

Models (smallest → largest, accuracy vs speed tradeoff):
    yolov8n.pt  ← recommended for Raspberry Pi / low-end devices
    yolov8s.pt  ← good balance (recommended for PC)
    yolov8m.pt  ← higher accuracy, needs more GPU memory
    yolov8l.pt  ← best accuracy, slow on CPU

Requirements:
    pip install ultralytics torch torchvision
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected: {name} ({mem:.1f} GB)")
            return True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  Apple Silicon GPU (MPS) detected")
            return True
        else:
            print("  No GPU found — training on CPU (will be slower)")
            return False
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 egg detector")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
                        help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto-detect based on GPU memory)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--out", default="runs/egg_counter", help="Output run directory")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("   YOLOv8 Egg Counter — Training")
    print("="*55)

    # Verify data.yaml exists
    if not Path(args.data).exists():
        print(f"\n[ERROR] data.yaml not found: {args.data}")
        print("  Run step2_organize_dataset.py first.")
        sys.exit(1)

    print(f"\nConfig:")
    print(f"  data.yaml : {args.data}")
    print(f"  model     : {args.model}")
    print(f"  epochs    : {args.epochs}")
    print(f"  imgsz     : {args.imgsz}")
    print(f"\nChecking hardware...")
    has_gpu = check_gpu()
    device = "0" if has_gpu else "cpu"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    # Load model
    if args.resume:
        last_ckpt = Path(args.out) / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"\n[ERROR] No checkpoint to resume from: {last_ckpt}")
            sys.exit(1)
        print(f"\nResuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"\nLoading base model: {args.model} (downloads ~6MB if not cached)")
        model = YOLO(args.model)

    # Training hyperparameters — tuned for egg detection
    train_args = dict(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        resume=args.resume,
        patience=20,            # Early stopping: stop if no improvement for 20 epochs
        save_period=10,         # Save checkpoint every 10 epochs
        val=True,
        plots=True,             # Save training plots (loss curves, PR curves)
        verbose=True,

        # Augmentation — helps with varied lighting and camera angles
        hsv_h=0.02,             # Hue variation (keeps eggs looking egg-colored)
        hsv_s=0.4,              # Saturation variation
        hsv_v=0.4,              # Brightness variation (important for dim lighting)
        flipud=0.0,             # Don't flip vertically (eggs don't roll upward)
        fliplr=0.5,             # Flip horizontal (eggs can roll either direction)
        mosaic=0.8,             # Mosaic augmentation (mix 4 images)
        mixup=0.1,              # MixUp augmentation
        degrees=5.0,            # Small rotation (camera tilt variation)
        translate=0.1,          # Small translation
        scale=0.3,              # Scale variation (egg size can vary with camera distance)
        shear=2.0,              # Small shear

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
    )

    print(f"\nStarting training...")
    print(f"  Output dir: {args.out}")
    print(f"  Early stopping: {train_args['patience']} epochs")
    print(f"\n[Press Ctrl+C to stop — best weights are auto-saved]\n")

    start = datetime.now()
    results = model.train(**train_args)
    elapsed = datetime.now() - start

    best_weights = Path(args.out) / "weights" / "best.pt"
    print(f"\n{'='*55}")
    print(f"  Training complete in {elapsed}")
    print(f"  Best weights: {best_weights}")
    print(f"\n  Next step: run step4_test_and_count.py")
    print(f"  Use: --weights {best_weights}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
