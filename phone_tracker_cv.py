"""
phone_tracker_cv.py

Alternative phone tracker that does NOT use YOLO or any training set.
It looks for dark, phone-sized rectangles (a powered-off phone seen from above),
measuring candidate sizes in real centimetres through the same calibration
homography as yolo_phone_tracker.py. Reuses calibration.json (run
`python yolo_phone_tracker.py --calibrate` first if you don't have one).

Best on a plain LIGHT background under the pane. Dark phone on a dark
background will not work with this method.

Run:
    python phone_tracker_cv.py
Keys: q = quit, m = show/hide the dark-pixel mask window.
Use the 'dark' slider to tune: raise it until the phone is solid white in
the mask, lower it if the background starts showing up.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1                 # same as your current script
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CALIBRATION_FILE = "calibration.json"
DESK_WIDTH_CM = 60.96            # 24 in
DESK_DEPTH_CM = 45.72            # 18 in

DARK_THRESHOLD = 70              # gray level below this counts as "dark" (0-255); slider tunes it
# Adaptive darkness: also count a pixel as dark if it is LOCAL_DELTA darker than the local
# background (handles glare / uneven light across the pane). Set ADAPTIVE = False to disable.
ADAPTIVE = False
LOCAL_DELTA = 50                 # how much darker than the surroundings (0-255)
LOCAL_MAX = 150                  # ...but never count pixels brighter than this
MIN_CONTOUR_PX = 200             # ignore tiny blobs

# Phone size window in centimetres (covers small phones up to big phones + case)
PHONE_LONG_CM = (11.0, 19.0)
PHONE_SHORT_CM = (5.0, 9.5)
MIN_FILL = 0.75                  # contour area / bounding rectangle area (rectangularity)
PANE_MARGIN_CM = 2.0             # ignore blobs whose centre is outside the pane (+margin)

CONFIRM_FRAMES = 3               # must be seen this many frames in a row, in one spot
CONFIRM_RADIUS_CM = 4.0
MISS_TOLERANCE = 3
SMOOTH_WINDOW = 5
MOVE_TOLERANCE_CM = 1.5

SEND_SERIAL = False              # True = actually move the charger (home first!)
SERIAL_PORT = "auto"           # "auto" finds the ESP32; or e.g. /dev/cu.usbserial-0001
SERIAL_BAUD = 115200


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(
            f"No {CALIBRATION_FILE} found. Run `python yolo_phone_tracker.py --calibrate` first.")
    with open(CALIBRATION_FILE) as f:
        return np.array(json.load(f)["homography"], dtype=np.float64)


def pts_to_desk(H, pts):
    """Map an Nx2 array of pixel points to desk centimetres."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def find_phone(frame, H, dark_thresh):
    """
    Return (best, candidates, mask).
    best: dict with 'box' (4x2 px), 'center_px', 'center_cm', 'fill', 'long', 'short' or None.
    candidates: list of (box_px, label, accepted) for every blob considered, for drawing.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = gray < dark_thresh
    if ADAPTIVE:
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))   # ~124 px: bigger than a phone's short side
        bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, k)
        bg = cv2.GaussianBlur(bg, (0, 0), 5)
        bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        dark = dark | ((gray.astype(np.int16) < bg.astype(np.int16) - LOCAL_DELTA) & (gray < LOCAL_MAX))
    mask = dark.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_PX:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_cm = pts_to_desk(H, box)
        s1 = float(np.linalg.norm(box_cm[0] - box_cm[1]))
        s2 = float(np.linalg.norm(box_cm[1] - box_cm[2]))
        long_, short_ = max(s1, s2), min(s1, s2)
        rect_area = rect[1][0] * rect[1][1]
        fill = area / rect_area if rect_area > 0 else 0.0
        center_cm = pts_to_desk(H, np.array([rect[0]]))[0]
        label = f"{long_:.1f}x{short_:.1f}cm f{fill:.2f}"

        ok = (PHONE_LONG_CM[0] <= long_ <= PHONE_LONG_CM[1]
              and PHONE_SHORT_CM[0] <= short_ <= PHONE_SHORT_CM[1]
              and fill >= MIN_FILL
              and -PANE_MARGIN_CM <= center_cm[0] <= DESK_WIDTH_CM + PANE_MARGIN_CM
              and -PANE_MARGIN_CM <= center_cm[1] <= DESK_DEPTH_CM + PANE_MARGIN_CM)
        candidates.append((box.astype(int), label, ok))

        if ok and fill > best_score:
            best_score = fill
            best = dict(box=box.astype(int), center_px=rect[0],
                        center_cm=(float(center_cm[0]), float(center_cm[1])),
                        fill=fill, long=long_, short=short_)
    return best, candidates, mask


# ---------------------------------------------------------------------------
# Confirmation + smoothing (same idea as yolo_phone_tracker.py)
# ---------------------------------------------------------------------------
class TargetConfirmer:
    def __init__(self):
        self.pos, self.streak, self.misses = None, 0, 0

    def update(self, x, y):
        self.misses = 0
        if (self.pos is not None and abs(x - self.pos[0]) <= CONFIRM_RADIUS_CM
                and abs(y - self.pos[1]) <= CONFIRM_RADIUS_CM):
            self.streak += 1
        else:
            self.streak = 1
        self.pos = (x, y)
        return (x, y) if self.streak >= CONFIRM_FRAMES else None

    def miss(self):
        self.misses += 1
        if self.misses > MISS_TOLERANCE:
            self.streak, self.pos = 0, None


class PositionSmoother:
    def __init__(self, window=SMOOTH_WINDOW):
        self.window, self.history = window, []

    def update(self, x, y):
        self.history.append((x, y))
        self.history = self.history[-self.window:]
        return (sum(p[0] for p in self.history) / len(self.history),
                sum(p[1] for p in self.history) / len(self.history))


def open_serial():
    """Connect to the ESP32 (charger_mover.ino) through motion.py. Returns a motion.Mover."""
    import motion
    return motion.start(port="auto", use_limit_switches=False)


def send_target(ser, x_cm, y_cm):
    if ser is not None:
        ser.goto_desk(x_cm, y_cm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    argparse.ArgumentParser().parse_args()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    H = load_calibration()
    confirmer, smoother = TargetConfirmer(), PositionSmoother()
    ser = open_serial() if SEND_SERIAL else None
    last_sent = None
    show_mask = False

    cv2.namedWindow("Phone Tracking (CV)")
    cv2.createTrackbar("dark", "Phone Tracking (CV)", DARK_THRESHOLD, 255, lambda v: None)
    print("Tracking started. q = quit, m = toggle mask window.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            dark = max(1, cv2.getTrackbarPos("dark", "Phone Tracking (CV)"))
            best, cands, mask = find_phone(frame, H, dark)

            target = None
            if best is None:
                confirmer.miss()
            else:
                cx, cy = best["center_cm"]
                confirmed = confirmer.update(cx, cy)
                if confirmed is not None:
                    target = smoother.update(*confirmed)

            for box, label, accepted in cands:
                color = (0, 255, 0) if accepted else (0, 0, 255)
                cv2.polylines(frame, [box], True, color, 2)
                cv2.putText(frame, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if target is not None:
                cv2.putText(frame, f"target ({target[0]:.1f}, {target[1]:.1f}) cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if (last_sent is None
                        or abs(target[0] - last_sent[0]) > MOVE_TOLERANCE_CM
                        or abs(target[1] - last_sent[1]) > MOVE_TOLERANCE_CM):
                    last_sent = target
                    send_target(ser, *target)
                    print(f"target -> x={target[0]:.1f}cm  y={target[1]:.1f}cm")

            cv2.imshow("Phone Tracking (CV)", frame)
            if show_mask:
                cv2.imshow("Dark mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"Final dark threshold: {dark}  (set DARK_THRESHOLD = {dark})")
                break
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Dark mask")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
