"""
grbl_controller.py

Host-side driver for the moving charger. Talks G-code over USB serial to an ESP32
running FluidNC (or any GRBL-compatible controller) that drives the two independent track motors
(X motor moves rod 1, Y motor moves rod 2). Plain Cartesian: from here it is just X/Y in mm.

Desk coordinates (what the tracker gives you): centimetres from the pane corner you clicked
first during calibration. This module converts them to machine millimetres, clamps them to
the area the charger can physically reach, and sends the move.

Hardware bring-up (no camera needed):
    python grbl_controller.py --list-ports
    python grbl_controller.py --port auto --home
    python grbl_controller.py --port auto --home --goto 30 22      # desk cm
    python grbl_controller.py --port auto --home --square          # drive the reach-zone corners
    python grbl_controller.py --dry-run --goto 30 22               # print G-code only

!! First test with the belts/motors UNLOADED or with the puck lifted off the pane.
!! Keep a hand on the power supply. Jog a few mm first, and check each axis direction.
"""

import argparse
import re
import sys
import time

# ---------------------------------------------------------------------------
# Config. Everything marked TUNE depends on how you wire and assemble it.
# ---------------------------------------------------------------------------
BAUD = 115200
FEED_MM_MIN = 3000          # TUNE: start slow, raise once it is reliable
MIN_INTERVAL_S = 0.25       # never send moves faster than this (protects the queue)
HOME_TIMEOUT_S = 120
MOVE_TIMEOUT_S = 30

# Reachable area for the puck centre, in desk cm (from the crossed-rod CAD: 434 x 242 mm).
# Origin is the pane corner used as (0,0) in calibration.json.
# TUNE if you changed the build or your calibration corner order.
REACH_X_CM = (10.0, 53.3)
REACH_Y_CM = (9.0, 33.1)

# Where the puck sits (desk mm) when the machine is homed (machine position 0,0).
# By default: the minimum corner of the reach zone.
HOME_DESK_MM = (REACH_X_CM[0] * 10, REACH_Y_CM[0] * 10)

# +1 if the machine axis increases in the same direction as the desk axis, -1 if it is opposite.
# TUNE: check with --goto and a ruler, flip a sign if the puck goes the wrong way.
AXIS_SIGN = (1, 1)

# Extra offset between where the code thinks the puck is and where it is (mm). TUNE after calibration.
PUCK_OFFSET_MM = (0.0, 0.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def desk_to_machine(x_cm, y_cm):
    """Desk cm -> (machine_x_mm, machine_y_mm, was_clamped)."""
    cx = clamp(x_cm, *REACH_X_CM)
    cy = clamp(y_cm, *REACH_Y_CM)
    mx = AXIS_SIGN[0] * (cx * 10 - HOME_DESK_MM[0]) + PUCK_OFFSET_MM[0]
    my = AXIS_SIGN[1] * (cy * 10 - HOME_DESK_MM[1]) + PUCK_OFFSET_MM[1]
    return mx, my, (cx != x_cm or cy != y_cm)


def list_ports():
    from serial.tools import list_ports as lp
    return [(p.device, p.description) for p in lp.comports()]


def find_port():
    """Best guess at the ESP32's USB serial port on macOS/Linux/Windows."""
    cands = []
    for dev, desc in list_ports():
        d = (dev + " " + desc).lower()
        if any(k in d for k in ("usbserial", "slab_usbtouart", "wchusbserial", "usbmodem",
                                "ttyusb", "ttyacm", "cp210", "ch340", "silicon labs")):
            cands.append(dev)
    if not cands:
        raise RuntimeError("No ESP32 serial port found. Plug it in, or pass --port explicitly "
                           "(see --list-ports).")
    return cands[0]


class GantryError(RuntimeError):
    pass


class Gantry:
    def __init__(self, port="auto", baud=BAUD, dry_run=False, serial_obj=None, auto_home=False):
        self.port, self.baud, self.dry_run = port, baud, dry_run
        self.ser = serial_obj
        self.auto_home = auto_home
        self._last_send = 0.0
        self.homed = False

    # ---- connection ------------------------------------------------------
    def connect(self):
        if self.dry_run:
            print("[gantry] dry-run: no serial port opened")
            return self
        if self.ser is None:
            import serial
            port = find_port() if self.port == "auto" else self.port
            print(f"[gantry] opening {port} @ {self.baud}")
            self.ser = serial.Serial(port, self.baud, timeout=0.2)
        time.sleep(2.0)                      # ESP32 resets when the port opens
        self.ser.write(b"\r\n\r\n")
        time.sleep(0.5)
        self._drain()
        self.send("G21 G90")                 # millimetres, absolute
        if self.auto_home:
            self.home()
        return self

    def close(self):
        if self.ser is not None and not self.dry_run:
            try:
                self.ser.close()
            except Exception:
                pass

    # ---- low level -------------------------------------------------------
    def _drain(self):
        out = []
        while True:
            line = self.ser.readline()
            if not line:
                break
            out.append(line.decode(errors="replace").strip())
        return out

    def send(self, cmd, timeout=10.0):
        """Send one line, wait for ok/error/ALARM. Returns the lines received."""
        if self.dry_run:
            print(f"[gantry] > {cmd}")
            return ["ok"]
        self.ser.write((cmd + "\n").encode())
        lines, t0 = [], time.time()
        while time.time() - t0 < timeout:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            low = line.lower()
            if low == "ok":
                return lines
            if low.startswith("error") or low.startswith("alarm"):
                raise GantryError(f"'{cmd}' -> {line}")
        raise GantryError(f"timeout waiting for reply to '{cmd}' (got {lines})")

    def status(self):
        """Return (state, mx, my) from a '?' status report, e.g. ('Idle', 12.0, 3.0)."""
        if self.dry_run:
            return "Idle", 0.0, 0.0
        self.ser.write(b"?")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            raw = self.ser.readline().decode(errors="replace").strip()
            if raw.startswith("<"):
                m = re.match(r"<([^|>]+)", raw)
                p = re.search(r"MPos:(-?[\d.]+),(-?[\d.]+)", raw)
                return (m.group(1) if m else "?",
                        float(p.group(1)) if p else None,
                        float(p.group(2)) if p else None)
        raise GantryError("no status reply to '?'")

    def wait_idle(self, timeout=MOVE_TIMEOUT_S):
        if self.dry_run:
            return
        t0 = time.time()
        while time.time() - t0 < timeout:
            state, _, _ = self.status()
            if state.startswith("Idle"):
                return
            if state.startswith("Alarm"):
                raise GantryError("controller is in ALARM state; unlock with $X or re-home")
            time.sleep(0.1)
        raise GantryError("timed out waiting for the move to finish")

    # ---- high level ------------------------------------------------------
    def unlock(self):
        self.send("$X")

    def home(self):
        print("[gantry] homing ($H). Keep clear of the mechanism.")
        self.send("$H", timeout=HOME_TIMEOUT_S)
        self.wait_idle(HOME_TIMEOUT_S)
        self.homed = True
        print("[gantry] homed")

    def goto_machine(self, x_mm, y_mm, feed=FEED_MM_MIN):
        self.send(f"G1 X{x_mm:.2f} Y{y_mm:.2f} F{feed:.0f}")

    def goto_desk(self, x_cm, y_cm, feed=FEED_MM_MIN, force=False):
        """Move the puck under desk position (x_cm, y_cm). Returns True if a move was sent."""
        now = time.time()
        if not force and not self.dry_run and now - self._last_send < MIN_INTERVAL_S:
            return False
        mx, my, clamped = desk_to_machine(x_cm, y_cm)
        if clamped:
            print(f"[gantry] target ({x_cm:.1f},{y_cm:.1f}) cm is outside the reach zone, clamped")
        self.goto_machine(mx, my, feed)
        self._last_send = now
        return True

    def stop(self):
        """Feed hold then soft reset: stops motion immediately."""
        if self.dry_run:
            print("[gantry] STOP")
            return
        self.ser.write(b"!")
        time.sleep(0.1)
        self.ser.write(b"\x18")


# ---------------------------------------------------------------------------
# Hooks used by phone_tracker_cv.py / phone_tracker_hybrid.py
# ---------------------------------------------------------------------------
def open_gantry(port="auto", auto_home=True):
    return Gantry(port=port, auto_home=auto_home).connect()


# ---------------------------------------------------------------------------
# CLI for bring-up
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="auto")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--goto", nargs=2, type=float, metavar=("X_CM", "Y_CM"))
    ap.add_argument("--square", action="store_true", help="visit the four corners of the reach zone")
    ap.add_argument("--feed", type=float, default=FEED_MM_MIN)
    a = ap.parse_args()

    if a.list_ports:
        for dev, desc in list_ports():
            print(dev, "-", desc)
        return

    g = Gantry(port=a.port, dry_run=a.dry_run).connect()
    try:
        if a.home:
            g.home()
        if a.goto:
            g.goto_desk(*a.goto, feed=a.feed, force=True)
            g.wait_idle()
        if a.square:
            corners = [(REACH_X_CM[0], REACH_Y_CM[0]), (REACH_X_CM[1], REACH_Y_CM[0]),
                       (REACH_X_CM[1], REACH_Y_CM[1]), (REACH_X_CM[0], REACH_Y_CM[1]),
                       (REACH_X_CM[0], REACH_Y_CM[0])]
            for cx, cy in corners:
                print(f"[gantry] -> corner ({cx:.1f}, {cy:.1f}) cm")
                g.goto_desk(cx, cy, feed=a.feed, force=True)
                g.wait_idle()
    except KeyboardInterrupt:
        g.stop()
    finally:
        g.close()


if __name__ == "__main__":
    main()
