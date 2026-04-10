"""
STEP 1 — Frame Extractor
========================
Run this on your video files to extract frames for labeling.
It automatically skips near-duplicate frames (blur/similarity check)
so you only get useful, varied images.

Usage:
    python step1_extract_frames.py --video myvideo.mp4 --out data/raw_frames
    python step1_extract_frames.py --video myvideo.mp4 --out data/raw_frames --fps 2
    python step1_extract_frames.py --folder videos/ --out data/raw_frames

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def is_blurry(frame, threshold=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_too_similar(frame, prev_frame, threshold=0.97):
    if prev_frame is None:
        return False
    a = cv2.resize(frame, (64, 64)).astype(np.float32)
    b = cv2.resize(prev_frame, (64, 64)).astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return False
    return np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b) > threshold


def extract_frames(video_path, out_dir, target_fps=1.0, prefix=""):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(video_fps / target_fps))

    print(f"  Video: {video_path.name} | {video_fps:.1f}fps | {duration:.1f}s | {total_frames} frames")
    print(f"  Sampling every {frame_interval} frames ({target_fps} fps target)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_blur = 0
    skipped_similar = 0
    frame_idx = 0
    prev_saved = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if is_blurry(frame):
                skipped_blur += 1
            elif is_too_similar(frame, prev_saved):
                skipped_similar += 1
            else:
                stem = video_path.stem.replace(" ", "_")
                fname = f"{prefix}{stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                prev_saved = frame.copy()
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"  Saved: {saved} | Skipped blurry: {skipped_blur} | Skipped similar: {skipped_similar}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from poultry videos for labeling")
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--folder", help="Folder containing multiple videos")
    parser.add_argument("--out", default="data/raw_frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames to extract per second (default: 1)")
    args = parser.parse_args()

    total = 0
    if args.video:
        print(f"\nExtracting from: {args.video}")
        total = extract_frames(Path(args.video), args.out, args.fps)
    elif args.folder:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4"}
        videos = [p for p in Path(args.folder).rglob("*") if p.suffix in video_exts]
        print(f"\nFound {len(videos)} videos in {args.folder}")
        for i, v in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}]")
            total += extract_frames(v, args.out, args.fps, prefix=f"v{i:02d}_")
    else:
        parser.print_help()
        return

    print(f"\n{'='*50}")
    print(f"  DONE — {total} frames saved to: {args.out}")
    print(f"  Next step: label these with LabelImg (see guide)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
