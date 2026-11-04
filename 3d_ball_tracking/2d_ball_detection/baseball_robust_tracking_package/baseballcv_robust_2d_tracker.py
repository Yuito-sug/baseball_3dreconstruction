
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust 2D Baseball Tracking Prototype
========================================
Usage:
python baseballcv_robust_2d_tracker.py input.mp4 \
    --output tracked.mp4 \
    --csv tracked.csv \
    --backend auto

Backends:
- auto: Try BaseballCV pretrained detector; fallback to safe mode
- baseballcv-yolo: Use BaseballCV pretrained detector
- opencv-classical: Use classical OpenCV motion detection

Important:
This is a prototype. It cannot guarantee accurate tracking on every video.
Occlusion positions are estimates, not detections.
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np


# ============================================================
# Video utilities
# ============================================================

def load_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame is None or frame.size == 0:
            continue

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No valid frames found in video.")

    return frames, fps, width, height


# ============================================================
# BaseballCV detector
# ============================================================

class BaseballCVDetector:

    def __init__(self):
        self.model = None
        self.model_name = None

        try:
            from baseballcv.functions.load_tools import LoadTools

            loader = LoadTools()

            # Try current ball-tracking model first.
            model_names = [
                "ball_trackingv4",
                "ball_tracking",
            ]

            for name in model_names:

                try:
                    self.model = loader.load_model(name)
                    self.model_name = name

                    print(
                        f"[INFO] BaseballCV model loaded: {name}"
                    )

                    break

                except Exception as e:
                    print(
                        f"[WARNING] Could not load {name}: {e}"
                    )

        except Exception as e:

            print(
                "[WARNING] BaseballCV unavailable:",
                e
            )

    def available(self):
        return self.model is not None

    def detect(self, frame):

        if self.model is None:
            return []

        try:

            results = self.model.predict(
                source=frame,
                verbose=False
            )

            candidates = []

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = xyxy

                    confidence = float(
                        box.conf[0].cpu().numpy()
                    )

                    x = (x1 + x2) / 2.0
                    y = (y1 + y2) / 2.0

                    candidates.append({
                        "x": float(x),
                        "y": float(y),
                        "confidence": confidence,
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                        "source": "baseballcv"
                    })

            return candidates

        except Exception as e:

            print(
                f"[WARNING] BaseballCV inference error: {e}"
            )

            return []


# ============================================================
# OpenCV fallback detector
# ============================================================

class OpenCVMotionDetector:

    def __init__(self):

        self.previous_gray = None

    def detect(self, frame):

        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if self.previous_gray is None:

            self.previous_gray = gray

            return []

        diff = cv2.absdiff(
            gray,
            self.previous_gray
        )

        self.previous_gray = gray

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        brightness = hsv[:, :, 2]

        # Motion + brightness mask.
        mask = (
            (diff > 15) &
            (brightness > 150)
        ).astype(np.uint8) * 255

        # Remove tiny noise.
        kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if not 3 <= area <= 500:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w > width * 0.15:
                continue

            if h > height * 0.15:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            candidates.append({
                "x": float(cx),
                "y": float(cy),
                "confidence": 0.25,
                "w": float(w),
                "h": float(h),
                "source": "opencv"
            })

        return candidates


# ============================================================
# Tracking
# ============================================================

def distance(a, b):

    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2
    )


def select_candidate(
    candidates,
    previous,
    velocity,
    frame_width,
    frame_height
):

    if not candidates:
        return None

    # First detection: avoid extreme borders.
    if previous is None:

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        valid = []

        for c in candidates:

            if not (
                0 <= c["x"] < frame_width and
                0 <= c["y"] < frame_height
            ):
                continue

            score = math.sqrt(
                (c["x"] - center_x) ** 2 +
                (c["y"] - center_y) ** 2
            )

            # Slight preference for small compact objects.
            score += 0.5 * abs(
                math.log(
                    max(c["w"], 1) /
                    max(c["h"], 1)
                )
            )

            score -= 20.0 * c.get(
                "confidence", 0.0
            )

            valid.append((score, c))

        if not valid:
            return None

        return min(
            valid,
            key=lambda item: item[0]
        )[1]

    # Predict next position.
    pred_x = previous["x"] + velocity[0]
    pred_y = previous["y"] + velocity[1]

    valid = []

    for c in candidates:

        if not (
            0 <= c["x"] < frame_width and
            0 <= c["y"] < frame_height
        ):
            continue

        d = math.sqrt(
            (c["x"] - pred_x) ** 2 +
            (c["y"] - pred_y) ** 2
        )

        # Adaptive gate based on frame size.
        gate = max(
            45.0,
            0.12 * math.sqrt(
                frame_width ** 2 +
                frame_height ** 2
            )
        )

        if d > gate:
            continue

        shape_penalty = 0.5 * abs(
            math.log(
                max(c["w"], 1) /
                max(c["h"], 1)
            )
        )

        confidence_bonus = 25.0 * c.get(
            "confidence", 0.0
        )

        score = (
            d +
            shape_penalty -
            confidence_bonus
        )

        valid.append((score, c))

    if not valid:
        return None

    return min(
        valid,
        key=lambda item: item[0]
    )[1]


def track_frames(
    frames,
    backend="auto"
):

    height, width = frames[0].shape[:2]

    baseballcv_detector = BaseballCVDetector()

    opencv_detector = OpenCVMotionDetector()

    if backend == "baseballcv-yolo":

        if not baseballcv_detector.available():

            raise RuntimeError(
                "BaseballCV model could not be loaded. "
                "Please check installation and model download."
            )

        detector = baseballcv_detector
        backend_used = "baseballcv"

    elif backend == "opencv-classical":

        detector = opencv_detector
        backend_used = "opencv-classical"

    else:

        if baseballcv_detector.available():

            detector = baseballcv_detector
            backend_used = "baseballcv"

        else:

            detector = opencv_detector
            backend_used = "opencv-classical"

            print(
                "[WARNING] BaseballCV model unavailable. "
                "Using OpenCV fallback."
            )

    points = []

    previous = None

    velocity = (0.0, 0.0)

    missed = 0

    max_missed = 8

    for frame_index, frame in enumerate(frames):

        candidates = detector.detect(frame)

        chosen = select_candidate(
            candidates,
            previous,
            velocity,
            width,
            height
        )

        if chosen is not None:

            if previous is not None:

                dx = chosen["x"] - previous["x"]
                dy = chosen["y"] - previous["y"]

                velocity = (
                    0.7 * velocity[0] + 0.3 * dx,
                    0.7 * velocity[1] + 0.3 * dy
                )

            previous = chosen.copy()

            missed = 0

            points.append({
                "frame": frame_index,
                "x": chosen["x"],
                "y": chosen["y"],
                "status": "detected",
                "confidence": chosen.get(
                    "confidence", 0.0
                )
            })

        else:

            missed += 1

            # Do not create fake detections.
            points.append({
                "frame": frame_index,
                "x": None,
                "y": None,
                "status": "not_detected",
                "confidence": None
            })

            if missed > max_missed:

                previous = None

                velocity = (0.0, 0.0)

    return points, backend_used


# ============================================================
# Occlusion estimation
# ============================================================

def estimate_occlusion(points):

    estimated = {}

    detected_indices = [
        i for i, p in enumerate(points)
        if p["status"] == "detected"
        and p["x"] is not None
    ]

    if len(detected_indices) < 2:
        return estimated

    for i in range(
        len(detected_indices) - 1
    ):

        left = detected_indices[i]
        right = detected_indices[i + 1]

        gap = right - left

        # Only interpolate short gaps.
        if gap <= 1 or gap > 8:
            continue

        p1 = points[left]
        p2 = points[right]

        for j in range(left + 1, right):

            t = (
                (j - left) /
                (right - left)
            )

            x = (
                p1["x"] +
                t * (p2["x"] - p1["x"])
            )

            y = (
                p1["y"] +
                t * (p2["y"] - p1["y"])
            )

            estimated[j] = {
                "x": x,
                "y": y,
                "status": "occlusion_estimate"
            }

    return estimated


# ============================================================
# Rendering
# ============================================================

def draw_dashed_line(
    frame,
    p1,
    p2,
    color=(0, 165, 255),
    thickness=2
):

    p1 = np.array(
        p1,
        dtype=float
    )

    p2 = np.array(
        p2,
        dtype=float
    )

    vector = p2 - p1

    length = np.linalg.norm(vector)

    if length < 1:
        return

    unit = vector / length

    dash_length = 12
    gap_length = 8

    current = 0.0

    while current < length:

        end = min(
            current + dash_length,
            length
        )

        start_point = tuple(
            np.round(
                p1 + unit * current
            ).astype(int)
        )

        end_point = tuple(
            np.round(
                p1 + unit * end
            ).astype(int)
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            color,
            thickness,
            cv2.LINE_AA
        )

        current = end + gap_length


def render_video(
    input_path,
    output_path,
    csv_path,
    backend
):

    frames, fps, width, height = load_video(
        input_path
    )

    print(
        f"[INFO] Loaded video: "
        f"{width}x{height}, "
        f"{fps:.2f} FPS, "
        f"{len(frames)} frames"
    )

    points, backend_used = track_frames(
        frames,
        backend=backend
    )

    occlusion = estimate_occlusion(
        points
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Cannot create output video: "
            f"{output_path}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, frame in enumerate(frames):

        vis = frame.copy()

        # Draw observed trajectory.
        observed = [
            p for p in points[:i + 1]
            if p["status"] == "detected"
            and p["x"] is not None
        ]

        if len(observed) >= 2:

            poly = np.array(
                [
                    [
                        int(p["x"]),
                        int(p["y"])
                    ]
                    for p in observed
                ],
                dtype=np.int32
            )

            cv2.polylines(
                vis,
                [poly],
                False,
                (0, 220, 0),
                2,
                cv2.LINE_AA
            )

        # Draw detected ball candidate.
        p = points[i]

        if (
            p["status"] == "detected"
            and p["x"] is not None
        ):

            x = int(p["x"])
            y = int(p["y"])

            cv2.circle(
                vis,
                (x, y),
                6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                "DETECTED",
                (x + 10, y - 10),
                font,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        # Draw occlusion estimate.
        if i in occlusion:

            q = occlusion[i]

            x = int(q["x"])
            y = int(q["y"])

            # Connect nearest previous detection.
            previous_detected = [
                p for p in points[:i]
                if p["status"] == "detected"
                and p["x"] is not None
            ]

            if previous_detected:

                prev = previous_detected[-1]

                draw_dashed_line(
                    vis,
                    (prev["x"], prev["y"]),
                    (x, y)
                )

            cv2.circle(
                vis,
                (x, y),
                8,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                "ESTIMATED",
                (x + 10, y - 10),
                font,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA
            )

        # Legend.
        box_width = min(
            width - 20,
            680
        )

        cv2.rectangle(
            vis,
            (10, 10),
            (box_width, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            vis,
            "2D Baseball Tracking Prototype",
            (20, 35),
            font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            "Red: detected candidate | Green: observed trajectory",
            (20, 60),
            font,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            "Orange: short-gap occlusion interpolation",
            (20, 82),
            font,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            f"Frame {i}/{len(frames)-1}",
            (20, 100),
            font,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        writer.write(vis)

    writer.release()

    # Save CSV.
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer_csv = csv.writer(f)

        writer_csv.writerow([
            "frame",
            "time_sec",
            "x",
            "y",
            "status",
            "confidence"
        ])

        for p in points:

            frame_index = p["frame"]

            writer_csv.writerow([
                frame_index,
                round(
                    frame_index / fps,
                    4
                ),
                p["x"],
                p["y"],
                p["status"],
                p["confidence"]
            ])

    print("[INFO] Processing complete.")
    print("[INFO] Backend:", backend_used)
    print("[INFO] Output video:", output_path)
    print("[INFO] Output CSV:", csv_path)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Robust 2D baseball ball tracking prototype"
        )
    )

    parser.add_argument(
        "input",
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        default="tracked.mp4",
        help="Output video path"
    )

    parser.add_argument(
        "--csv",
        default="tracked.csv",
        help="Output tracking CSV path"
    )

    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "baseballcv-yolo",
            "opencv-classical"
        ],
        default="auto"
    )

    args = parser.parse_args()

    render_video(
        args.input,
        args.output,
        args.csv,
        args.backend
    )


if __name__ == "__main__":
    main()
