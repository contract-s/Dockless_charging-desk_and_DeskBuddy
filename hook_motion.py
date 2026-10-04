"""
hook_motion.py  —  connect a tracker to the stepper motors (motion.py), one time.

    python hook_motion.py yolo_then_dark_tracker.py
    python hook_motion.py yolo_phone_tracker.py

It replaces the tracker's open_serial() and send_target() so every confirmed phone position
is turned into motor steps and sent to the ESP32, and sets SEND_SERIAL = True.
A backup is written next to it as <name>.before_motion.py.
"""

import re
import shutil
import sys

NEW_OPEN = '''def open_serial():
    import motion
    return motion.start(port="auto", use_limit_switches=False)   # True if you wired limit switches
'''

NEW_SEND = '''def send_target(ser, x_cm, y_cm):
    if ser is None:
        return
    ser.goto_desk(x_cm, y_cm)       # phone cm -> motor steps -> ESP32 (see motion.py)
'''


def replace_func(src, name, new):
    m = re.search(rf"^def {name}\(.*?(?=^\S)", src, flags=re.S | re.M)
    if not m:
        raise SystemExit(f"could not find def {name}() in the file")
    return src[:m.start()] + new + "\n\n" + src[m.end():].lstrip("\n")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    src = open(path).read()
    if "motion.start(" in src:
        print("already hooked up:", path)
        return
    shutil.copy(path, path.replace(".py", ".before_motion.py"))
    src = replace_func(src, "open_serial", NEW_OPEN)
    src = replace_func(src, "send_target", NEW_SEND)
    src, n = re.subn(r"^SEND_SERIAL = False", "SEND_SERIAL = True", src, flags=re.M)
    open(path, "w").write(src)
    compile(src, path, "exec")
    print(f"hooked up {path} (SEND_SERIAL {'set to True' if n else 'unchanged'}); "
          f"backup: {path.replace('.py', '.before_motion.py')}")


if __name__ == "__main__":
    main()
