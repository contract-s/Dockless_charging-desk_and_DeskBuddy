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

CAMERA_INDEX = 1
# Logitech C270. Use a 4:3 mode: the 18x24 in pane is also 4:3, so the full
# frame maps onto the pane with nothing wasted. 720p (16:9) crops the top and
# bottom of the sensor, which would force the camera much higher to see the
# 18 in side. 640x480 is ~1 mm per pixel across the pane, and YOLO resizes
# to 640 internally anyway.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_NAME = "yolov8n.pt"          # nano model: smallest, fastest, good first choice
CELL_PHONE_CLASS_ID = 67           # COCO class index for "cell phone"
CONFIDENCE_THRESHOLD = 0.5

# False-positive filters (tune using the debug label on the preview)
MIN_BOX_FRAC = 0.01     # box area / frame area must be at least this
MAX_BOX_FRAC = 0.20     # ...and at most this
MIN_ASPECT = 1.2        # long side / short side of the box
MAX_ASPECT = 3.2
PANE_MARGIN_CM = 2.0    # ignore detections centered outside the pane (+margin)
CONFIRM_FRAMES = 3      # must be seen this many frames in a row, in one spot
CONFIRM_RADIUS_CM = 4.0
MISS_TOLERANCE = 2      # missed frames allowed before the streak resets
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
 
class TargetConfirmer:
    """Only trust a detection after it shows up in the same spot several frames running."""

    def __init__(self):
        self.pos = None
        self.streak = 0
        self.misses = 0

    def update(self, x, y):
        self.misses = 0
        if (self.pos is not None
                and abs(x - self.pos[0]) <= CONFIRM_RADIUS_CM
                and abs(y - self.pos[1]) <= CONFIRM_RADIUS_CM):
            self.streak += 1
        else:
            self.streak = 1
        self.pos = (x, y)
        return (x, y) if self.streak >= CONFIRM_FRAMES else None

    def miss(self):
        self.misses += 1
        if self.misses > MISS_TOLERANCE:
            self.streak = 0
            self.pos = None


def pick_phone(boxes, H, frame_shape):
    """Return the best detection that passes the size/shape/location filters, else None."""
    fh, fw = frame_shape[:2]
    best, best_conf = None, -1.0
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        frac = (w * h) / (fw * fh)
        aspect = max(w, h) / min(w, h)
        if not (MIN_BOX_FRAC <= frac <= MAX_BOX_FRAC):
            continue
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = pixel_to_desk(H, cx, cy)
        if not (-PANE_MARGIN_CM <= dx <= DESK_WIDTH_CM + PANE_MARGIN_CM
                and -PANE_MARGIN_CM <= dy <= DESK_DEPTH_CM + PANE_MARGIN_CM):
            continue
        conf = float(b.conf[0])
        if conf > best_conf:
            best_conf = conf
            best = ((x1, y1, x2, y2), cx, cy, dx, dy)
    return best


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
 
# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                         help="Run the 4-point calibration step and exit.")
    parser.add_argument("--no-display", action="store_true",
                         help="Run headless (no preview window) — use on the Pi.")
    args = parser.parse_args()
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam — check CAMERA_INDEX.")
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
 
    if args.calibrate:
        run_calibration(cap)
        cap.release()
        return
 
    H = load_calibration()
    model = YOLO(MODEL_NAME)
    smoother = PositionSmoother()
    confirmer = TargetConfirmer()
 
    ser = open_serial() if SEND_SERIAL else None
    last_sent = None
 
    fps_counter, fps_timer, fps_value = 0, time.time(), 0.0
 
    print("Tracking started. Press 'q' in the preview window to quit (or Ctrl+C if headless).")
 
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
 
            results = model(frame, classes=[CELL_PHONE_CLASS_ID],
                             conf=CONFIDENCE_THRESHOLD, verbose=False)
 
            boxes = results[0].boxes
            target_desk = None
 
            best = pick_phone(boxes, H, frame.shape) if len(boxes) > 0 else None
            if best is None:
                confirmer.miss()
            else:
                (x1, y1, x2, y2), cx, cy, desk_x, desk_y = best
                confirmed = confirmer.update(desk_x, desk_y)
                if confirmed is not None:
                    desk_x, desk_y = smoother.update(*confirmed)
                    target_desk = (desk_x, desk_y)
 
                if not args.no_display:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"desk: ({desk_x:.1f}, {desk_y:.1f}) cm",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
            if target_desk is not None:
                if (last_sent is None
                        or abs(target_desk[0] - last_sent[0]) > MOVE_TOLERANCE_CM
                        or abs(target_desk[1] - last_sent[1]) > MOVE_TOLERANCE_CM):
                    last_sent = target_desk
                    send_target(ser, *target_desk)
                    print(f"target -> x={target_desk[0]:.1f}cm  y={target_desk[1]:.1f}cm")
 
            # simple FPS counter, printed once a second
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_value = fps_counter / (time.time() - fps_timer)
                fps_counter, fps_timer = 0, time.time()
                if args.no_display:
                    print(f"FPS: {fps_value:.1f}")
 
            if not args.no_display:
                cv2.putText(frame, f"FPS: {fps_value:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Phone Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
 
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
 
 
if __name__ == "__main__":
    main()
