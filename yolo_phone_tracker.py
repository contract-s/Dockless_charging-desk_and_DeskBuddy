"""
yolo_phone_tracker.py

Detects a phone in a webcam feed using YOLOv8's pretrained COCO model
(class 67 = "cell phone"), converts the detected pixel position into
real-world desk coordinates via a one-time homography calibration, and
prints/sends the target position. Meant to run on your laptop first
(webcam index 0), then move unchanged onto the Raspberry Pi later.

Install:
    pip install ultralytics opencv-python numpy pyserial

Run:
    python yolo_phone_tracker.py
    python yolo_phone_tracker.py --calibrate   (run this first, once)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
# Logitech C270. Use a 4:3 mode: the 18x24 in pane is also 4:3, so the full
# frame maps onto the pane with nothing wasted. 720p (16:9) crops the top and
# bottom of the sensor, which would force the camera much higher to see the
# 18 in side. 640x480 is ~1 mm per pixel across the pane, and YOLO resizes
# to 640 internally anyway.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_NAME = "yolov8n.pt"          # nano model: smallest, fastest, good first choice
CELL_PHONE_CLASS_ID = 67           # COCO class index for "cell phone"
CONFIDENCE_THRESHOLD = 0.4
CALIBRATION_FILE = "calibration.json"

# Real-world size in cm of the area you click during calibration.
# Set to the full 24 x 18 in plexiglass pane (24 in along the camera's
# width, 18 in along its height). Once the gantry is built, shrink these to
# the area the coil can actually reach, and click that area's corners instead.
DESK_WIDTH_CM = 60.96   # 24 in
DESK_DEPTH_CM = 45.72   # 18 in

# Smoothing: how many recent positions to average
SMOOTH_WINDOW = 5

# Only send a new target if the phone moved more than this many cm
MOVE_TOLERANCE_CM = 1.5

# If a serial connection to the motion controller is available, set this to
# the right port (e.g. "/dev/ttyUSB0" or "COM5") and set SEND_SERIAL = True.
SEND_SERIAL = False
SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200


# ---------------------------------------------------------------------------
# Calibration: map pixel coordinates -> real desk coordinates (cm)
# ---------------------------------------------------------------------------

def run_calibration(cap):
    """
    Click the 4 corners of the desk's reachable area in the live feed, in this
    order: top-left, top-right, bottom-right, bottom-left (as seen on screen).
    These get paired with the real-world rectangle (0,0) .. (DESK_WIDTH_CM, DESK_DEPTH_CM)
    to compute a homography.
    """
    clicked_points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"  point {len(clicked_points)}: ({x}, {y})")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("Click the 4 corners of the desk's charging area, in order:")
    print("  1) top-left  2) top-right  3) bottom-right  4) bottom-left")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, pt, 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) == 4:
            break

    cv2.destroyWindow("Calibration")

    if len(clicked_points) != 4:
        raise RuntimeError("Calibration needs exactly 4 points — try again.")

    pixel_pts = np.array(clicked_points, dtype=np.float32)
    real_pts = np.array([
        [0, 0],
        [DESK_WIDTH_CM, 0],
        [DESK_WIDTH_CM, DESK_DEPTH_CM],
        [0, DESK_DEPTH_CM],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(pixel_pts, real_pts)

    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"homography": H.tolist()}, f)

    print(f"Saved calibration to {CALIBRATION_FILE}")
    return H


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(
            f"No {CALIBRATION_FILE} found — run with --calibrate first."
        )
    with open(CALIBRATION_FILE) as f:
        data = json.load(f)
    return np.array(data["homography"])


def pixel_to_desk(H, px, py):
    p = np.array([px, py, 1.0])
    dp = H @ p
    dp /= dp[2]
    return float(dp[0]), float(dp[1])

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
 
class PositionSmoother:
    def __init__(self, window=SMOOTH_WINDOW):
        self.window = window
        self.history = []
 
    def update(self, x, y):
        self.history.append((x, y))
        if len(self.history) > self.window:
            self.history.pop(0)
        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        return sum(xs) / len(xs), sum(ys) / len(ys)
 
# ---------------------------------------------------------------------------
# Serial (optional — only used if SEND_SERIAL = True)
# ---------------------------------------------------------------------------
 
def open_serial():
    import serial
    return serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
 
 
def send_target(ser, x_cm, y_cm):
    if ser is None:
        return
    ser.write(f"{x_cm:.1f},{y_cm:.1f}\n".encode())
 
