"""
phone_tracker_cascade.py

Simple two-stage tracker:
  1. Look for a phone with the screen ON using YOLO (same rules as yolo_phone_tracker.py).
  2. Only if YOLO finds nothing, look for a black rectangle with OpenCV (phone OFF).

No learning step needed. Optional: press 'l' while the dark box is on your phone to restrict
the OpenCV stage to your phone's size (fewer false positives). 'c' clears it.

Files needed in this folder: phone_tracker_cv.py, phone_tracker_hybrid.py, calibration.json
Run:  python phone_tracker_cascade.py
Keys: q quit | l learn size | c clear | m mask | s save debug snapshot
"""

import time

import cv2

import phone_tracker_cv as ptc
import phone_tracker_hybrid as hyb

CONFIRM_YOLO = 2         # frames in a row to accept a lit phone
CONFIRM_CV = 6           # frames in a row to accept a black rectangle
HOLD_FRAMES = 15         # keep the last target this long after losing it


def main():
    from ultralytics import YOLO

    cap = cv2.VideoCapture(ptc.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX in phone_tracker_cv.py.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ptc.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ptc.FRAME_HEIGHT)

    H = ptc.load_calibration()
    model = YOLO(hyb.MODEL_NAME)
    profile = hyb.load_profile()
    confirmer, smoother = hyb.Confirmer(), ptc.PositionSmoother()
    ser = ptc.open_serial() if ptc.SEND_SERIAL else None
    last_sent, show_mask, held, hold_left = None, False, None, 0
    win = "Phone Tracking (cascade)"
    cv2.namedWindow(win)
    cv2.createTrackbar("dark", win, ptc.DARK_THRESHOLD, 255, lambda v: None)
    print("Stage 1: YOLO (screen on). Stage 2: dark rectangle (screen off).")
    print("Learned size:", profile if profile else "none (using default phone size range)")
    print("Keys: q quit | l learn size | c clear | m mask | s save snapshot")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            raw = frame.copy()
            dark = max(1, cv2.getTrackbarPos("dark", win))

            yolo = [d for d in hyb.yolo_detections(model, frame, H) if d.get("strong_ok")]
            cv_best, cands, mask = ptc.find_phone(frame, H, dark)

            target, source, need, pos = None, "-", 0, None
            if yolo:                                          # stage 1
                d = max(yolo, key=lambda d: d["conf"])
                pos, source, need = d["center_cm"], "yolo", CONFIRM_YOLO
            elif cv_best is not None:                         # stage 2
                pos, source, need = cv_best["center_cm"], "cv", CONFIRM_CV

            if pos is None:
                confirmer.miss()
            else:
                confirmed = confirmer.update(pos[0], pos[1], need)
                if confirmed is not None:
                    target = smoother.update(*confirmed)
                    held, hold_left = target, HOLD_FRAMES
            if target is None and held is not None and hold_left > 0:
                hold_left -= 1
                target, source = held, "hold"

            for d in yolo:
                x1, y1, x2, y2 = map(int, d["box_px"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                cv2.putText(frame, f"yolo {d['conf']:.2f}", (x1, y2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
            for box, label, accepted in cands:
                col = (0, 255, 0) if accepted else (0, 0, 255)
                cv2.polylines(frame, [box], True, col, 2)
                cv2.putText(frame, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            if target is not None:
                cv2.putText(frame, f"[{source}] target ({target[0]:.1f}, {target[1]:.1f}) cm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if (last_sent is None
                        or abs(target[0] - last_sent[0]) > ptc.MOVE_TOLERANCE_CM
                        or abs(target[1] - last_sent[1]) > ptc.MOVE_TOLERANCE_CM):
                    last_sent = target
                    ptc.send_target(ser, *target)
                    print(f"[{source}] target -> x={target[0]:.1f}cm  y={target[1]:.1f}cm")
            else:
                cv2.putText(frame, f"searching  yolo:{len(yolo)} cv:{1 if cv_best else 0} "
                                   f"streak:{confirmer.streak}/{need or '-'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            if show_mask:
                cv2.imshow("Dark mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"Final dark threshold: {dark}")
                break
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Dark mask")
            elif key == ord("l"):
                if cv_best is not None:
                    import json
                    profile = dict(long_cm=round(cv_best["long"], 2), short_cm=round(cv_best["short"], 2))
                    json.dump(profile, open(hyb.PROFILE_FILE, "w"))
                    hyb.apply_profile(profile["long_cm"], profile["short_cm"])
                    print("Learned phone size:", profile)
                else:
                    print("Nothing to learn from: no green dark box right now.")
            elif key == ord("c"):
                profile = None
                hyb.clear_profile()
                print("Cleared learned size.")
            elif key == ord("s"):
                tag = time.strftime("%H%M%S")
                cv2.imwrite(f"snap_{tag}_raw.png", raw)
                cv2.imwrite(f"snap_{tag}_mask.png", mask)
                print(f"saved snap_{tag}_raw.png / _mask.png (dark={dark}, adaptive={ptc.ADAPTIVE})")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()
