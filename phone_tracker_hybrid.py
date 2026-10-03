"""
phone_tracker_hybrid.py

Combines YOLO (good when the phone is ON) with the dark-rectangle detector from
phone_tracker_cv.py (good when the phone is OFF), and uses agreement between
them to avoid false positives on random dark rectangles.

Decision rules, per frame:
  1. YOLO detects a phone with confidence >= STRONG_CONF       -> use it       (3 frames to confirm)
  2. Else a dark-rectangle candidate that a WEAK YOLO box
     (conf >= WEAK_CONF) also covers                             -> use it       (3 frames to confirm)
  3. Else a dark-rectangle candidate on its own                  -> use it ONLY if you have taught it
     your phone's size (press 'l'), and then it must stay put
     for CONFIRM_CV_ONLY frames.

Teach it your phone: put the phone (off, or on) on the pane where it is detected as a
green box, press 'l'. It stores the phone's real size in phone_profile.json and from then
on the CV-only path accepts only blobs within +/-12% of that size. Press 'c' to clear.

Files needed in the same folder: phone_tracker_cv.py, calibration.json
Run:  python phone_tracker_hybrid.py
Keys: q quit | l learn phone size | c clear learned size | m show/hide dark mask
"""

import json
import os

import cv2
import numpy as np

import phone_tracker_cv as ptc

MODEL_NAME = "yolov8n.pt"
CELL_PHONE_CLASS_ID = 67
YOLO_IMGSZ = 640            # try 960 if YOLO misses a lit phone
STRONG_CONF = 0.35
WEAK_CONF = 0.08
# YOLO boxes are axis-aligned, so a rotated phone's box is larger than the phone
YOLO_LONG_CM = (6.0, 30.0)
YOLO_SHORT_CM = (3.0, 22.0)

# Same filters as yolo_phone_tracker.py (the ones that work well for lit phones)
MIN_BOX_FRAC = 0.01
MAX_BOX_FRAC = 0.20
MIN_ASPECT = 1.2
MAX_ASPECT = 3.2
CONFIRM_YOLO = 2
CONFIRM_CV_ONLY = 12
CONFIRM_RADIUS_CM = 5.0
MISS_TOLERANCE = 8          # frames a detection may drop out before the streak resets
HOLD_FRAMES = 15            # keep reporting the last confirmed target this long after losing it

PROFILE_FILE = "phone_profile.json"
PROFILE_TOL = 0.12
ALLOW_CV_ONLY_WITHOUT_PROFILE = False   # leave False to avoid false positives until you teach it

DEFAULT_LONG = ptc.PHONE_LONG_CM
DEFAULT_SHORT = ptc.PHONE_SHORT_CM
LAST_REJECTED = []   # YOLO boxes rejected by the size/pane filter (drawn in magenta)


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------
def yolo_detections(model, frame, H):
    """Phone detections (conf >= WEAK_CONF) that pass loose size and pane checks."""
    res = model(frame, classes=[CELL_PHONE_CLASS_ID], conf=WEAK_CONF,
                imgsz=YOLO_IMGSZ, verbose=False)[0]
    dets = []
    LAST_REJECTED.clear()
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        cm = ptc.pts_to_desk(H, corners)
        s1 = float(np.linalg.norm(cm[0] - cm[1]))
        s2 = float(np.linalg.norm(cm[1] - cm[2]))
        long_, short_ = max(s1, s2), min(s1, s2)
        center_cm = ptc.pts_to_desk(H, np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
        in_pane = (-ptc.PANE_MARGIN_CM <= center_cm[0] <= ptc.DESK_WIDTH_CM + ptc.PANE_MARGIN_CM
                   and -ptc.PANE_MARGIN_CM <= center_cm[1] <= ptc.DESK_DEPTH_CM + ptc.PANE_MARGIN_CM)
        fh, fw = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        conf = float(b.conf[0])
        frac = (bw * bh) / (fw * fh) if bw > 0 and bh > 0 else 0
        aspect = max(bw, bh) / max(1e-6, min(bw, bh))
        # lit-phone rule: identical to yolo_phone_tracker.pick_phone
        lit_ok = (conf >= STRONG_CONF and in_pane
                  and MIN_BOX_FRAC <= frac <= MAX_BOX_FRAC
                  and MIN_ASPECT <= aspect <= MAX_ASPECT)
        cm_ok = (YOLO_LONG_CM[0] <= long_ <= YOLO_LONG_CM[1]
                 and YOLO_SHORT_CM[0] <= short_ <= YOLO_SHORT_CM[1] and in_pane)
        if lit_ok or cm_ok:
            dets.append(dict(conf=conf, box_px=(x1, y1, x2, y2), strong_ok=lit_ok,
                             center_cm=(float(center_cm[0]), float(center_cm[1]))))
        else:
            why = "outside pane" if not in_pane else "size/shape"
            LAST_REJECTED.append(((x1, y1, x2, y2),
                                  f"yolo REJECT({why}) {conf:.2f} {long_:.1f}x{short_:.1f}cm"))
    return dets


def decide(yolo, cv_best, profile_learned, allow_unlearned=ALLOW_CV_ONLY_WITHOUT_PROFILE):
    """Return (center_cm, source, frames_needed) or None."""
    strong = [d for d in yolo if d.get("strong_ok")]
    if strong:
        d = max(strong, key=lambda d: d["conf"])
        return d["center_cm"], "yolo", CONFIRM_YOLO
    if cv_best is not None:
        px, py = cv_best["center_px"]
        for d in yolo:                       # weak YOLO box covering the CV blob
            x1, y1, x2, y2 = d["box_px"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return cv_best["center_cm"], "cv+yolo", CONFIRM_YOLO
        if profile_learned or allow_unlearned:
            return cv_best["center_cm"], "cv-only", CONFIRM_CV_ONLY
    return None


# ---------------------------------------------------------------------------
# Confirmation / profile
# ---------------------------------------------------------------------------
class Confirmer:
    def __init__(self):
        self.pos, self.streak, self.misses = None, 0, 0

    def update(self, x, y, need):
        self.misses = 0
        if (self.pos is not None and abs(x - self.pos[0]) <= CONFIRM_RADIUS_CM
                and abs(y - self.pos[1]) <= CONFIRM_RADIUS_CM):
            self.streak += 1
        else:
            self.streak = 1
        self.pos = (x, y)
        return (x, y) if self.streak >= need else None

    def miss(self):
        self.misses += 1
        if self.misses > MISS_TOLERANCE:
            self.streak, self.pos = 0, None


def apply_profile(long_cm, short_cm):
    ptc.PHONE_LONG_CM = (long_cm * (1 - PROFILE_TOL), long_cm * (1 + PROFILE_TOL))
    ptc.PHONE_SHORT_CM = (short_cm * (1 - PROFILE_TOL), short_cm * (1 + PROFILE_TOL))


def clear_profile():
    ptc.PHONE_LONG_CM, ptc.PHONE_SHORT_CM = DEFAULT_LONG, DEFAULT_SHORT
    if os.path.exists(PROFILE_FILE):
        os.remove(PROFILE_FILE)


def load_profile():
    if os.path.exists(PROFILE_FILE):
        p = json.load(open(PROFILE_FILE))
        apply_profile(p["long_cm"], p["short_cm"])
        return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from ultralytics import YOLO

    cap = cv2.VideoCapture(ptc.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX in phone_tracker_cv.py.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ptc.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ptc.FRAME_HEIGHT)

    H = ptc.load_calibration()
    model = YOLO(MODEL_NAME)
    profile = load_profile()
    confirmer, smoother = Confirmer(), ptc.PositionSmoother()
    ser = ptc.open_serial() if ptc.SEND_SERIAL else None
    last_sent, show_mask = None, False
    held, hold_left = None, 0
    win = "Phone Tracking (hybrid)"

    cv2.namedWindow(win)
    cv2.createTrackbar("dark", win, ptc.DARK_THRESHOLD, 255, lambda v: None)
    print("Learned phone size:", profile if profile else "none (CV-only path disabled)")
    print("Keys: q quit | l learn phone size | c clear | m mask | s save debug snapshot")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            raw_frame = frame.copy()
            dark = max(1, cv2.getTrackbarPos("dark", win))
            yolo = yolo_detections(model, frame, H)
            cv_best, cands, mask = ptc.find_phone(frame, H, dark)

            choice = decide(yolo, cv_best, profile is not None)
            target, source = None, "-"
            if choice is None:
                confirmer.miss()
            else:
                (cx, cy), source, need = choice
                confirmed = confirmer.update(cx, cy, need)
                if confirmed is not None:
                    target = smoother.update(*confirmed)
                    held, hold_left = target, HOLD_FRAMES
            if target is None and held is not None and hold_left > 0:
                hold_left -= 1
                target, source = held, "hold"

            for d in yolo:
                x1, y1, x2, y2 = map(int, d["box_px"])
                col = (0, 200, 0) if d.get("strong_ok") else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
                cv2.putText(frame, f"yolo {d['conf']:.2f}", (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            for (rx1, ry1, rx2, ry2), rl in LAST_REJECTED:
                cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 0, 255), 1)
                cv2.putText(frame, rl, (int(rx1), max(12, int(ry1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            for box, label, accepted in cands:
                col = (0, 255, 0) if accepted else (0, 0, 255)
                cv2.polylines(frame, [box], True, col, 2)
                cv2.putText(frame, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            if target is None:
                info = (f"searching  yolo:{len(yolo)} cv:{1 if cv_best else 0} "
                        f"streak:{confirmer.streak} misses:{confirmer.misses}")
                if choice is not None:
                    info += f" via {choice[1]} need {choice[2]}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if target is not None:
                cv2.putText(frame, f"[{source}] target ({target[0]:.1f}, {target[1]:.1f}) cm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if (last_sent is None
                        or abs(target[0] - last_sent[0]) > ptc.MOVE_TOLERANCE_CM
                        or abs(target[1] - last_sent[1]) > ptc.MOVE_TOLERANCE_CM):
                    last_sent = target
                    ptc.send_target(ser, *target)
                    print(f"[{source}] target -> x={target[0]:.1f}cm  y={target[1]:.1f}cm")

            overlay = frame.copy()
            cv2.imshow(win, frame)
            if show_mask:
                cv2.imshow("Dark mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"Final dark threshold: {dark} (set DARK_THRESHOLD = {dark} in phone_tracker_cv.py)")
                break
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Dark mask")
            elif key == ord("l"):
                if cv_best is not None:
                    profile = dict(long_cm=round(cv_best["long"], 2), short_cm=round(cv_best["short"], 2))
                    json.dump(profile, open(PROFILE_FILE, "w"))
                    apply_profile(profile["long_cm"], profile["short_cm"])
                    print("Learned phone size:", profile)
                else:
                    print("Nothing to learn from: no green CV box right now.")
            elif key == ord("s"):
                import time as _t
                tag = _t.strftime("%H%M%S")
                cv2.imwrite(f"snap_{tag}_overlay.png", overlay)
                cv2.imwrite(f"snap_{tag}_mask.png", mask)
                cv2.imwrite(f"snap_{tag}_raw.png", raw_frame)
                print(f"saved snap_{tag}_raw.png / _overlay.png / _mask.png (dark={dark}, adaptive={ptc.ADAPTIVE})")
            elif key == ord("c"):
                profile = None
                clear_profile()
                print("Cleared learned phone size.")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
