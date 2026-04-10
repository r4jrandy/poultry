"""
STEP 2 — Dataset Organizer
===========================
Run this AFTER you have labeled your images with LabelImg.
It will:
  - Split your labeled images into train / val / test sets
  - Create the YOLOv8 data.yaml config file
  - Report class distribution

Folder structure expected BEFORE running:
    labeled/
        images/    ← your .jpg frames
        labels/    ← matching .txt YOLO annotation files from LabelImg

Usage:
    python step2_organize_dataset.py --labeled labeled/ --out dataset/
    python step2_organize_dataset.py --labeled labeled/ --out dataset/ --split 0.8 0.1 0.1

Requirements:
    pip install numpy
"""

import os
import shutil
import random
import argparse
import yaml
from pathlib import Path
from collections import Counter


# ── Class names for your poultry farm ──────────────────────────────────────
# Edit this list if you add more classes later (e.g. "broken_egg", "double_yolk")
CLASS_NAMES = ["egg"]
# ───────────────────────────────────────────────────────────────────────────


def collect_labeled_pairs(labeled_dir):
    """Find all image/label pairs where BOTH files exist."""
    img_dir = Path(labeled_dir)  # / "images"
    lbl_dir = Path(labeled_dir)  # / "labels"

    img_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    pairs = []
    missing_labels = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix not in img_extensions:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"\n[!] {len(missing_labels)} images have no label file — skipping:")
        for f in missing_labels[:10]:
            print(f"     {f}")
        if len(missing_labels) > 10:
            print(f"     ... and {len(missing_labels)-10} more")

    return pairs


def count_annotations(pairs):
    class_counts = Counter()
    total_boxes = 0
    for _, lbl in pairs:
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
                    total_boxes += 1
    return class_counts, total_boxes


def copy_split(pairs, out_dir, split_name):
    img_out = out_dir / split_name / "images"
    lbl_out = out_dir / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)


def main():
    parser = argparse.ArgumentParser(description="Organize labeled images into YOLOv8 dataset")
    parser.add_argument("--labeled", default="labeled", help="Folder with images/ and labels/ subfolders")
    parser.add_argument("--out", default="dataset", help="Output dataset folder")
    parser.add_argument("--split", nargs=3, type=float, default=[0.80, 0.10, 0.10],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (must sum to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_r, val_r, test_r = args.split
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting labeled pairs from: {args.labeled}")
    pairs = collect_labeled_pairs(args.labeled)
    print(f"Found {len(pairs)} labeled image/label pairs")

    if len(pairs) < 10:
        print("\n[!] Very few labeled images. Aim for at least 100 for good results.")

    # Count class distribution
    class_counts, total_boxes = count_annotations(pairs)
    print(f"\nAnnotation summary:")
    print(f"  Total bounding boxes: {total_boxes}")
    for cls_id, count in sorted(class_counts.items()):
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  Class {cls_id} ({name}): {count} boxes")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"\nSplit: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    # Copy files
    print("\nCopying files...")
    copy_split(train_pairs, out_dir, "train")
    copy_split(val_pairs, out_dir, "val")
    copy_split(test_pairs, out_dir, "test")

    # Write data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\ndata.yaml written to: {yaml_path}")
    print(f"\n{'='*50}")
    print(f"  DATASET READY at: {out_dir}")
    print(f"  Next step: run step3_train.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
