"""
phone_tracker_screen_off.py  —  finds a phone with the screen OFF. OpenCV only, no YOLO.

How it works:
  1. At start it photographs the EMPTY pane (keep the pane clear for the first second).
  2. Every frame it compares the camera image to that empty-pane photo. Anything new that
     appeared on the pane and is phone-shaped (rectangle, phone-sized in real cm) is the phone.
     Reflections that are always on the plexiglass are in the empty-pane photo too, so they cancel.
  3. Only the pane area is looked at (from calibration.json), so things around it are ignored.
  4. Brightness is evened out every frame, and the empty-pane photo slowly follows light changes
     (only where nothing is lying), so normal light drift doesn't break it.

Run (from the folder with calibration.json):   python phone_tracker_screen_off.py
Keys:   q quit  |  b retake the empty pane (clear the pane first)  |  m show/hide what changed
Slider: "sensitivity" = how different from the empty pane counts as "something is there".
        Raise it if shadows or glare get boxed; lower it if the phone is missed.
Boxes:  BLUE = phone   thin RED = something changed but isn't phone-shaped (label says why)

To drive the motors: set SEND_SERIAL = True (needs motion.py and the ESP32).
"""

import json
import os
import time

import cv2
import numpy as np

# ============================== SETTINGS ==============================
CAMERA_INDEX = 1
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CALIBRATION_FILE = "calibration.json"
DESK_WIDTH_CM, DESK_DEPTH_CM = 60.96, 45.72      # 24 x 18 in pane

DIFF_THRESHOLD = 30          # 0-255, slider "sensitivity" tunes it
PHONE_LONG_CM = (11.0, 19.0)
PHONE_SHORT_CM = (5.0, 9.5)
MIN_FILL = 0.70              # how rectangular (blob area / rotated box area)
JOIN_GAP_PX = 25             # 2nd try: join pieces this close (a reflection splitting the phone)
PANE_INSET_CM = 0.5          # ignore a thin strip at the pane's edge (frame/mount shadows)

BG_FRAMES = 15               # frames averaged for the empty-pane photo
BG_LEARN_RATE = 0.02         # how fast the empty-pane photo follows slow light changes

CONFIRM_FRAMES = 5           # phone must stay in one spot this many frames
CONFIRM_RADIUS_CM = 4.0
MISS_TOLERANCE = 5           # frames it can drop out before the streak restarts
SMOOTH_WINDOW = 5
MOVE_TOLERANCE_CM = 1.5      # only send a new target if it moved more than this

SEND_SERIAL = False          # True -> move the charger (motion.py + ESP32)
# ======================================================================


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError("No calibration.json here. Run: python yolo_phone_tracker.py --calibrate")
    with open(CALIBRATION_FILE) as f:
        return np.array(json.load(f)["homography"], dtype=np.float64)


def to_desk(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def pane_mask(H, shape):
    """White where the pane is in the camera image."""
    i = PANE_INSET_CM
    corners = np.array([[i, i], [DESK_WIDTH_CM - i, i], [DESK_WIDTH_CM - i, DESK_DEPTH_CM - i],
                        [i, DESK_DEPTH_CM - i]], dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(corners, np.linalg.inv(H)).reshape(-1, 2).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [px], 255)
    if cv2.countNonZero(m) < 1000:          # odd calibration: use the whole image
        m[:] = 255
    return m


def prep(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


class EmptyPane:
    def __init__(self, pane):
        self.pane = pane
        self.bg = None
        self.frames = []

    def ready(self):
        return self.bg is not None

    def reset(self):
        self.bg, self.frames = None, []

    def add_startup_frame(self, gray):
        self.frames.append(gray)
        if len(self.frames) >= BG_FRAMES:
            self.bg = np.median(np.stack(self.frames), axis=0).astype(np.float32)
            self.frames = []

    def difference(self, gray):
        """|frame - empty pane| after evening out overall brightness, only inside the pane."""
        inside = self.pane > 0
        gain = np.median(self.bg[inside]) / max(np.median(gray[inside]), 1.0)
        g = gray * gain
        diff = np.abs(g - self.bg)
        diff[~inside] = 0
        return diff, g

    def learn(self, g, foreground):
        """Slowly follow light changes, but never where something is lying on the pane."""
        keep_out = cv2.dilate(foreground, np.ones((25, 25), np.uint8))
        update = ((keep_out == 0) & (self.pane > 0)).astype(np.uint8)
        cv2.accumulateWeighted(g, self.bg, BG_LEARN_RATE, mask=update)


def find_phone(diff, H, threshold):
    """-> (best, candidates, mask). best = dict(box, center_cm) or None."""
    mask = (diff > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    best, cands = _shapes(mask, H)
    if best is None and len(cands) > 1:
        # 2nd try: a reflection may have split the phone into pieces -> join nearby pieces
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (JOIN_GAP_PX, JOIN_GAP_PX)))
        best2, cands2 = _shapes(joined, H)
        if best2 is not None:
            return best2, cands2, joined
    return best, cands, mask


def _shapes(mask, H):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_fill, cands = None, -1.0, []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        cm = to_desk(H, box)
        s1 = float(np.linalg.norm(cm[0] - cm[1]))
        s2 = float(np.linalg.norm(cm[1] - cm[2]))
        long_, short_ = max(s1, s2), min(s1, s2)
        fill = area / max(rect[1][0] * rect[1][1], 1.0)
        center = to_desk(H, [rect[0]])[0]

        why = None
        if not (PHONE_LONG_CM[0] <= long_ <= PHONE_LONG_CM[1]
                and PHONE_SHORT_CM[0] <= short_ <= PHONE_SHORT_CM[1]):
            why = "size"
        elif fill < MIN_FILL:
            why = "shape"
        label = f"{long_:.1f}x{short_:.1f}cm fill {fill:.2f}" + (f" ({why})" if why else "")
        cands.append((box.astype(int), label, why is None))
        if why is None and fill > best_fill:
            best_fill = fill
            best = dict(box=box.astype(int), center_cm=(float(center[0]), float(center[1])))
    return best, cands


class Confirmer:
    def __init__(self):
        self.pos, self.streak, self.misses = None, 0, 0

    def update(self, x, y):
        self.misses = 0
        if self.pos and abs(x - self.pos[0]) <= CONFIRM_RADIUS_CM and abs(y - self.pos[1]) <= CONFIRM_RADIUS_CM:
            self.streak += 1
        else:
            self.streak = 1
        self.pos = (x, y)
        return self.streak >= CONFIRM_FRAMES

    def miss(self):
        self.misses += 1
        if self.misses > MISS_TOLERANCE:
            self.pos, self.streak = None, 0


class Smoother:
    def __init__(self):
        self.hist = []

    def update(self, x, y):
        self.hist = (self.hist + [(x, y)])[-SMOOTH_WINDOW:]
        return (sum(p[0] for p in self.hist) / len(self.hist),
                sum(p[1] for p in self.hist) / len(self.hist))


def open_serial():
    import motion
    return motion.start(port="auto", use_limit_switches=False)


def send_target(ser, x_cm, y_cm):
    if ser is not None:
        ser.goto_desk(x_cm, y_cm)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam (check CAMERA_INDEX).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    H = load_calibration()
    ok, frame = cap.read()
    while not ok:
        ok, frame = cap.read()
    empty = EmptyPane(pane_mask(H, frame.shape))
    confirmer, smoother = Confirmer(), Smoother()
    ser = open_serial() if SEND_SERIAL else None
    last_sent, show_mask = None, False

    win = "Phone tracking (screen off)"
    cv2.namedWindow(win)
    cv2.createTrackbar("sensitivity", win, DIFF_THRESHOLD, 120, lambda v: None)
    print("Keep the pane EMPTY for a second while it photographs it...")
    print("Keys: q quit | b retake empty pane | m show what changed")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = prep(frame)

            if not empty.ready():
                empty.add_startup_frame(gray)
                cv2.putText(frame, "Photographing the EMPTY pane - keep it clear...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if empty.ready():
                    print("Empty pane saved. Put the phone down.")
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            thr = max(5, cv2.getTrackbarPos("sensitivity", win))
            diff, g = empty.difference(gray)
            best, cands, mask = find_phone(diff, H, thr)
            empty.learn(g, mask)

            target = None
            if best is None:
                confirmer.miss()
            elif confirmer.update(*best["center_cm"]):
                target = smoother.update(*best["center_cm"])

            for box, label, ok_ in cands:
                if not ok_:
                    cv2.polylines(frame, [box], True, (0, 0, 255), 1)
                    cv2.putText(frame, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if best is not None:
                cx, cy = best["center_cm"]
                cv2.polylines(frame, [best["box"]], True, (255, 128, 0), 2)
                cv2.putText(frame, f"phone ({cx:.1f}, {cy:.1f}) cm [{confirmer.streak}/{CONFIRM_FRAMES}]",
                            tuple(best["box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)

            if target is not None:
                cv2.putText(frame, f"TARGET ({target[0]:.1f}, {target[1]:.1f}) cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                if (last_sent is None or abs(target[0] - last_sent[0]) > MOVE_TOLERANCE_CM
                        or abs(target[1] - last_sent[1]) > MOVE_TOLERANCE_CM):
                    last_sent = target
                    send_target(ser, *target)
                    print(f"target -> x={target[0]:.1f}cm  y={target[1]:.1f}cm")
            else:
                cv2.putText(frame, "searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            if show_mask:
                cv2.imshow("What changed", mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print(f"sensitivity was {thr} (set DIFF_THRESHOLD = {thr} to keep it)")
                break
            elif k == ord("b"):
                empty.reset()
                confirmer, smoother, last_sent = Confirmer(), Smoother(), None
                print("Retaking the empty pane - clear the pane...")
            elif k == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("What changed")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
