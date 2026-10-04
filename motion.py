"""
motion.py  —  phone position (desk cm)  ->  stepper motor steps  ->  ESP32 (charger_mover.ino)

Mechanism:
    X motor = TOP track  -> moves the VERTICAL rod left/right  -> charger X
    Y motor = LEFT track -> moves the HORIZONTAL rod up/down   -> charger Y
    The MagSafe is fixed where the rods cross, so moving the two rods puts it at (X, Y).

The maths, per axis:
    distance_mm = (phone_cm - home_cm) * 10 * direction
    steps       = distance_mm * STEPS_PER_MM
    STEPS_PER_MM = motor steps per revolution / mm of belt per revolution
                 = (200 * microsteps)  / (pulley teeth * 2 mm)   for NEMA17 + GT2 belt
                 = 2048                / (pulley teeth * 2 mm)   for 28BYJ-48 + GT2 belt

Bring-up (no camera), from this folder:
    python motion.py --list-ports
    python motion.py --zero                     # charger is at HOME by hand -> call it 0,0
    python motion.py --zero --jog x 20          # top-track motor: charger should go 20 mm along +X
    python motion.py --zero --jog y 20          # left-track motor: charger should go 20 mm along +Y
    python motion.py --zero --goto 30 22        # move under desk point (30 cm, 22 cm)
    python motion.py --dry-run --goto 30 22     # just print the steps it would send

HOME = the corner of the charger's travel nearest the two motors (where the rods are closest to
the top and left tracks' motors). Before each run, push the charger there by hand (motors off),
or wire limit switches and use --home. On quit, the code drives it back home so the next run
starts from the right place.
"""

import argparse
import time

# ============================== SETTINGS ==============================
MOTOR_TYPE = "nema17"        # "nema17" (A4988/DRV8825/TMC2209)  or  "28byj48" (ULN2003). Match the .ino
MICROSTEPS = 16              # nema17 only: driver microstep setting (A4988 with MS1-3 high = 16)
PULLEY_TEETH = 20            # GT2 pulley on each motor (2 mm per tooth)

if MOTOR_TYPE == "nema17":
    STEPS_PER_REV = 200 * MICROSTEPS
else:
    STEPS_PER_REV = 2048
STEPS_PER_MM = STEPS_PER_REV / (PULLEY_TEETH * 2.0)      # 80 for nema17 1/16, 51.2 for 28BYJ-48

# Where the charger's centre is, in the camera's desk coordinates (cm), when it is at HOME.
# Measure once: push the charger home, lay the phone over it, run the tracker, read the cm it prints.
HOME_DESK_CM = (10.0, 9.0)

# How far each rod can travel from HOME, mm (from the CAD: 434 x 242). Moves are clamped to this.
TRAVEL_MM = (434.0, 242.0)

# +1 if a positive motor step moves the charger toward larger desk cm on that axis, else -1.
# Check with --jog: if the charger moves the wrong way, flip the sign here.
DIRECTION = (+1, +1)

# True if the camera's desk X runs along the LEFT track (i.e. the camera's axes are swapped
# relative to the mechanism). Check: move the phone left/right in front of the camera;
# the TOP-track motor should be the one that responds.
SWAP_XY = False

PORT = "auto"
BAUD = 115200
MIN_INTERVAL_S = 0.2         # don't send new targets faster than this
PARK_ON_CLOSE = True         # drive back home on quit, so the next run starts at home
# ======================================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def desk_to_steps(x_cm, y_cm):
    """Phone position in desk cm -> (x_steps, y_steps, clamped?)."""
    if SWAP_XY:
        x_cm, y_cm = y_cm, x_cm
    out, clamped = [], False
    for i, v in enumerate((x_cm, y_cm)):
        mm = (v - HOME_DESK_CM[i]) * 10.0 * DIRECTION[i]
        c = clamp(mm, 0.0, TRAVEL_MM[i])
        clamped |= abs(c - mm) > 1e-6
        out.append(int(round(c * STEPS_PER_MM)))
    return out[0], out[1], clamped


def find_port():
    from serial.tools import list_ports
    keys = ("usbserial", "slab_usbtouart", "wchusbserial", "usbmodem", "ttyusb", "ttyacm",
            "cp210", "ch340", "ch910", "silicon labs")
    for p in list_ports.comports():
        if any(k in (p.device + " " + (p.description or "")).lower() for k in keys):
            return p.device
    raise RuntimeError("No ESP32 found. Plug it in, or pass --port (see --list-ports).")


class MoverError(RuntimeError):
    pass


class Mover:
    def __init__(self, port=PORT, dry_run=False, serial_obj=None):
        self.port, self.dry_run, self.ser = port, dry_run, serial_obj
        self.last_send = 0.0
        self.target = (0, 0)

    # ---- connection ----
    def connect(self):
        if self.dry_run:
            print("[motion] dry run: nothing is sent")
            return self
        if self.ser is None:
            import serial
            port = find_port() if self.port == "auto" else self.port
            print(f"[motion] opening {port}")
            self.ser = serial.Serial()
            self.ser.port, self.ser.baudrate, self.ser.timeout = port, BAUD, 0.2
            self.ser.dtr = False          # try not to reset the ESP32 on open
            self.ser.rts = False
            self.ser.open()
        t0 = time.time()                   # wait for "ready" if it did reset
        while time.time() - t0 < 2.5:
            ln = self.ser.readline().decode(errors="replace").strip()
            if ln.startswith("ready"):
                break
        self.ser.reset_input_buffer()
        self.status()                      # proves the board answers
        return self

    def close(self, park=PARK_ON_CLOSE):
        if self.dry_run or self.ser is None:
            return
        try:
            if park:
                print("[motion] parking at home")
                self.send("G 0 0")
                self.wait_idle()
        except Exception as e:
            print("[motion] could not park:", e)
        self.ser.close()

    # ---- protocol ----
    def send(self, cmd, timeout=5.0):
        if self.dry_run:
            print(f"[motion] > {cmd}")
            return "ok"
        self.ser.write((cmd + "\n").encode())
        t0 = time.time()
        while time.time() - t0 < timeout:
            ln = self.ser.readline().decode(errors="replace").strip()
            if not ln:
                continue
            if ln == "ok" or ln.startswith("pos "):
                return ln
            if ln.startswith("error"):
                raise MoverError(f"{cmd!r} -> {ln}")
        raise MoverError(f"no reply to {cmd!r} (wrong port, or firmware not uploaded?)")

    def status(self):
        """-> (x_steps, y_steps, moving)"""
        if self.dry_run:
            return self.target[0], self.target[1], False
        r = self.send("?").split()
        return int(r[1]), int(r[2]), r[4] == "1"

    def wait_idle(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.status()[2]:
                return
            time.sleep(0.1)
        raise MoverError("move did not finish in time")

    # ---- actions ----
    def zero(self):
        """The charger is at HOME right now: call this position 0,0."""
        self.send("Z")
        self.target = (0, 0)

    def home(self):
        """Home with limit switches (only if the firmware has USE_LIMIT_SWITCHES 1)."""
        print("[motion] homing...")
        self.send("H", timeout=120)
        self.target = (0, 0)

    def stop(self):
        self.send("S")

    def motors(self, on):
        self.send(f"E {1 if on else 0}")

    def goto_steps(self, xs, ys):
        self.send(f"G {xs} {ys}")
        self.target = (xs, ys)

    def goto_desk(self, x_cm, y_cm, force=False):
        """Move the charger under desk point (x_cm, y_cm). Returns the step target, or None if skipped."""
        now = time.time()
        if not force and now - self.last_send < MIN_INTERVAL_S:
            return None
        xs, ys, clamped = desk_to_steps(x_cm, y_cm)
        if (xs, ys) == self.target:
            return xs, ys
        if clamped:
            print(f"[motion] ({x_cm:.1f}, {y_cm:.1f}) cm is outside the charger's reach; going to the nearest edge")
        self.goto_steps(xs, ys)
        self.last_send = now
        print(f"[motion] phone ({x_cm:.1f}, {y_cm:.1f}) cm -> X {xs} steps ({xs / STEPS_PER_MM:.0f} mm), "
              f"Y {ys} steps ({ys / STEPS_PER_MM:.0f} mm)")
        return xs, ys

    def jog(self, axis, mm):
        x, y = self.target
        d = int(round(mm * STEPS_PER_MM))
        if axis == "x":
            x = int(clamp(x + d, 0, TRAVEL_MM[0] * STEPS_PER_MM))
        else:
            y = int(clamp(y + d, 0, TRAVEL_MM[1] * STEPS_PER_MM))
        self.goto_steps(x, y)


def start(port=PORT, use_limit_switches=False):
    """Used by the tracker: connect and set the starting position."""
    m = Mover(port=port).connect()
    if use_limit_switches:
        m.home()
    else:
        print("[motion] assuming the charger is at HOME now (push it there by hand if not)")
        m.zero()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--zero", action="store_true", help="charger is at home now: call it 0,0")
    ap.add_argument("--home", action="store_true", help="home with limit switches")
    ap.add_argument("--jog", nargs=2, metavar=("AXIS", "MM"))
    ap.add_argument("--goto", nargs=2, type=float, metavar=("X_CM", "Y_CM"))
    ap.add_argument("--square", action="store_true", help="drive to the 4 corners of the travel")
    ap.add_argument("--off", action="store_true", help="turn the motors off (push by hand)")
    ap.add_argument("--stay", action="store_true", help="don't drive back home at the end")
    a = ap.parse_args()

    if a.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(p.device, "-", p.description)
        return

    print(f"[motion] {MOTOR_TYPE}, {STEPS_PER_MM:g} steps/mm, travel {TRAVEL_MM[0]:.0f} x {TRAVEL_MM[1]:.0f} mm")
    m = Mover(port=a.port, dry_run=a.dry_run).connect()
    park = not a.stay
    try:
        if a.home:
            m.home()
        if a.zero:
            m.zero()
        if a.off:
            m.motors(False)
            park = False
            print("[motion] motors off")
        if a.jog:
            m.jog(a.jog[0].lower(), float(a.jog[1]))
            m.wait_idle()
            print("[motion] at", m.status()[:2], "steps")
        if a.goto:
            m.goto_desk(*a.goto, force=True)
            m.wait_idle()
        if a.square:
            X, Y = (int(t * STEPS_PER_MM) for t in TRAVEL_MM)
            for xs, ys in [(X, 0), (X, Y), (0, Y), (0, 0)]:
                print(f"[motion] corner -> {xs}, {ys} steps")
                m.goto_steps(xs, ys)
                m.wait_idle()
        if a.goto or a.jog:
            input("[motion] press Enter to finish (it will drive back home)... " if park else
                  "[motion] press Enter to finish... ")
    except KeyboardInterrupt:
        m.stop()
    finally:
        m.close(park=park)


if __name__ == "__main__":
    main()
