"""
STEP 4 — YOLOv8 Egg Counter (Inference + Counting)
====================================================
Replaces the old Hough-circle detector with your trained YOLOv8 model.
Much more accurate — won't confuse hens with eggs.

Usage:
    python step4_test_and_count.py --weights runs/egg_counter/weights/best.pt --source 1
    python step4_test_and_count.py --weights best.pt --source video.mp4
    python step4_test_and_count.py --weights best.pt --source 0 --rows 4 --headless

Controls (when window is open):
    q — quit and save report
    r — reset count
    p — pause / resume

Requirements:
    pip install ultralytics opencv-python numpy pandas
"""

import cv2
import numpy as np
import argparse
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ─── Centroid Tracker ────────────────────────────────────────────────────────
class EggTracker:
    def __init__(self, max_disappeared=25, max_distance=60):
        self.next_id = 0
        self.objects = {}        # id → (x, y)
        self.disappeared = {}
        self.crossed = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, detections):
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if not self.objects:
            for c in detections:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()), dtype=float)
        det_centroids = np.array(detections, dtype=float)

        D = np.linalg.norm(obj_centroids[:, None] - det_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            self.objects[oid] = tuple(det_centroids[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(len(detections))) - used_cols:
            self.register(tuple(det_centroids[c]))

        return self.objects


# ─── Report ───────────────────────────────────────────────────────────────────
def save_report(total, per_cage, report_dir):
    Path(report_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = Path(report_dir) / f"egg_count_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", ts])
        w.writerow(["Cage Row", "Egg Count"])
        for row in sorted(per_cage):
            w.writerow([f"Row {row}", per_cage[row]])
        w.writerow(["TOTAL", total])
    print(f"[✓] Report saved: {csv_path}")

    json_path = Path(report_dir) / f"egg_count_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "total_eggs": total,
            "per_cage_row": {f"row_{k}": v for k, v in sorted(per_cage.items())},
        }, f, indent=2)
    print(f"[✓] Report saved: {json_path}")


# ─── Draw overlay ─────────────────────────────────────────────────────────────
def draw_overlay(frame, detections, tracker, total, per_cage, line_x_frac, num_rows):
    h, w = frame.shape[:2]
    lx = int(line_x_frac * w)

    # Cage row dividers
    for i in range(1, num_rows):
        y = int(i * h / num_rows)
        cv2.line(frame, (0, y), (w, y), (200, 200, 50), 1)
        cv2.putText(frame, f"Row {i}", (6, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)

    # Counting line
    cv2.line(frame, (lx, 0), (lx, h), (0, 255, 120), 2)

    # YOLOv8 detections
    for (x1, y1, x2, y2, conf) in detections:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 200, 255), 1)

    # Tracker dots
    for oid, (cx, cy) in tracker.objects.items():
        color = (50, 255, 50) if oid in tracker.crossed else (255, 180, 50)
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

    # HUD
    cv2.rectangle(frame, (0, 0), (200, 24 + num_rows * 22), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL: {total}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
    for i, row in enumerate(sorted(per_cage), 1):
        cv2.putText(frame, f"  Row {row}: {per_cage[row]}", (6, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1)

    return frame


# ─── Main ─────────────────────────────────────────────────────────────────────
def run(weights, source, num_rows, line_x, conf_thresh, headless, report_dir):
    from ultralytics import YOLO

    print(f"\n[INFO] Loading model: {weights}")
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_px = int(line_x * w)
    tol = 8

    tracker = EggTracker()
    total_count = 0
    per_cage_count = defaultdict(int)

    print(f"[INFO] Running | {w}x{h} | rows={num_rows} | conf≥{conf_thresh}")
    print("[INFO] Press q=quit, r=reset, p=pause\n")

    paused = False
    fps_time = time.time()
    fps_frames = 0
    fps_display = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # YOLOv8 inference
            results = model(frame, conf=conf_thresh, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, conf))

            # Update tracker with centroids
            centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2, _) in detections]
            tracker.update(centroids)

            # Count eggs crossing line
            for oid, (cx, cy) in tracker.objects.items():
                if oid not in tracker.crossed and abs(int(cx) - line_px) <= tol:
                    tracker.crossed.add(oid)
                    total_count += 1
                    row = min(num_rows, max(1, int(cy / h * num_rows) + 1))
                    per_cage_count[row] += 1
                    print(f"[EGG] #{total_count} | Row {row} | conf={detections[centroids.index((int(cx), int(cy)))][4]:.2f}" if (int(cx), int(cy)) in centroids else f"[EGG] #{total_count} | Row {row}")

            # FPS
            fps_frames += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_frames
                fps_frames = 0
                fps_time = time.time()

        if not headless:
            vis = draw_overlay(frame.copy(), detections if not paused else [],
                               tracker, total_count, per_cage_count, line_x, num_rows)
            cv2.putText(vis, f"{fps_display} FPS", (w - 70, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            if paused:
                cv2.putText(vis, "PAUSED", (w // 2 - 40, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            cv2.imshow("Egg Counter (YOLOv8)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                total_count = 0
                per_cage_count = defaultdict(int)
                tracker.crossed.clear()
                print("[INFO] Count reset.")
            elif key == ord("p"):
                paused = not paused

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"\n{'='*45}")
    print(f"  Total eggs: {total_count}")
    for row in sorted(per_cage_count):
        print(f"    Row {row}: {per_cage_count[row]}")
    print(f"{'='*45}")
    save_report(total_count, dict(per_cage_count), report_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Egg Counter")
    parser.add_argument("--weights", required=True, help="Path to best.pt from training")
    parser.add_argument("--source", default=0,
                        help="Camera index (0, 1) or video file path")
    parser.add_argument("--rows", type=int, default=4, help="Number of cage rows")
    parser.add_argument("--line", type=float, default=0.5, help="Counting line position (0–1)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument("--report-dir", default="egg_reports", help="Where to save reports")
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    run(
        weights=args.weights,
        source=source,
        num_rows=args.rows,
        line_x=args.line,
        conf_thresh=args.conf,
        headless=args.headless,
        report_dir=args.report_dir,
    )
