"""
test_one_motor.py  —  bench test: camera -> phone_tracker_screen_off -> motion.py -> ESP32 -> ONE motor.

Move the phone LEFT/RIGHT across the pane and the carriage on your short test track follows:
phone at the pane's left edge -> carriage at the motor end, right edge -> far end of the track.
(The whole pane width is squeezed onto the short track so you can see it work.)

Nothing in phone_tracker_screen_off.py or motion.py is changed: this script only swaps the
"desk cm -> steps" maths for one that fits the test track, then runs the normal tracker.

Wiring: the test motor on the ESP32's X driver (STEP gpio26, DIR gpio27, ENABLE gpio25),
same as the real top-track motor. Motor power from its own supply, common GND with the ESP32.

Steps:
  1. Set TRACK_MM below to how far the carriage can travel on your test track (measure it,
     then subtract ~10 mm so it never hits the end).
  2. Push the carriage to the MOTOR end of the track by hand.
  3. python test_one_motor.py --sweep     # motor only: goes to the far end and back
       - goes the wrong way?        -> set FLIP = True
       - goes the wrong distance?   -> check MICROSTEPS / PULLEY_TEETH in motion.py
  4. python test_one_motor.py             # camera + motor: start with the pane empty, then slide the phone

  python test_one_motor.py --dry-run      # no ESP32: prints the steps it would send
On quit (q in the video window) the carriage drives back to the motor end.
"""

import argparse
import types

import motion

# ============================== SETTINGS ==============================
TRACK_MM = 150.0         # usable carriage travel on the test track, mm
FOLLOW = "x"             # "x" = phone left/right moves the carriage, "y" = phone up/down
FLIP = False             # True if the carriage goes the opposite way to the phone
DESK_WIDTH_CM, DESK_DEPTH_CM = 60.96, 45.72
# ======================================================================


def test_desk_to_steps(x_cm, y_cm):
    """Pane position -> steps on the short test track (X motor only; Y stays at 0)."""
    v, size = (x_cm, DESK_WIDTH_CM) if FOLLOW == "x" else (y_cm, DESK_DEPTH_CM)
    frac = v / size
    clamped = not (0.0 <= frac <= 1.0)
    frac = min(1.0, max(0.0, frac))
    if FLIP:
        frac = 1.0 - frac
    return int(round(frac * TRACK_MM * motion.STEPS_PER_MM)), 0, clamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="motor only: far end and back, no camera")
    ap.add_argument("--dry-run", action="store_true", help="no ESP32, just print")
    ap.add_argument("--port", default="auto")
    a = ap.parse_args()

    motion.desk_to_steps = test_desk_to_steps            # the only change: fit the short track
    motion.TRAVEL_MM = (TRACK_MM, 0.0)
    far = int(TRACK_MM * motion.STEPS_PER_MM)
    print(f"[test] {motion.MOTOR_TYPE}, {motion.STEPS_PER_MM:g} steps/mm, track {TRACK_MM:.0f} mm "
          f"= {far} steps. Carriage must be at the MOTOR end now.")

    m = motion.Mover(port=a.port, dry_run=a.dry_run).connect()
    m.zero()
    try:
        if a.sweep:
            input(f"[test] Enter = move {TRACK_MM:.0f} mm to the far end... ")
            m.goto_steps(far, 0)
            m.wait_idle()
            input("[test] Did it stop at the far end (not short, not crashing)? Enter = go back... ")
            m.goto_steps(0, 0)
            m.wait_idle()
            print("[test] back at the motor end. Sweep done.")
        else:
            import phone_tracker_screen_off as tracker
            print("[test] Keep the pane empty for a second, then slide the phone "
                  + ("left/right." if FOLLOW == "x" else "up/down."))
            tracker.main(state=types.SimpleNamespace(mover=m, tracking_enabled=True))
    except KeyboardInterrupt:
        m.stop()
    finally:
        m.close()       # drives back to the motor end


if __name__ == "__main__":
    main()
