"""One-time patch: make phone_tracker_cv.py (and therefore the hybrid tracker) drive the gantry."""
import re
p = "phone_tracker_cv.py"
s = open(p).read()
if "grbl_controller" in s:
    print("already patched"); raise SystemExit
s = s.replace('SERIAL_PORT = "/dev/ttyUSB0"', 'SERIAL_PORT = "auto"           # "auto" finds the ESP32; or e.g. /dev/cu.usbserial-0001')
s = s.replace("SEND_SERIAL = False", "SEND_SERIAL = False              # True = actually move the charger (home first!)", 1)
old = s[s.index("def open_serial():"):s.index("# ---------------------------------------------------------------------------\n# Main")]
new = '''def open_serial():
    """Connect to the gantry controller and home it. Returns a Gantry (has .goto_desk and .close)."""
    import grbl_controller
    return grbl_controller.open_gantry(port=SERIAL_PORT, auto_home=True)


def send_target(ser, x_cm, y_cm):
    if ser is not None:
        ser.goto_desk(x_cm, y_cm)


'''
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
