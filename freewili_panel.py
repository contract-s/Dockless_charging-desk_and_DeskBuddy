"""
freewili_panel.py  —  the FREE-WILi as the desk's control panel.

    buttons   green = bring charger to phone    yellow = energy saved    blue = phone status
              red   = park / follow               gray   = park the charger   (see config.FREEWILI_BUTTONS)
    LEDs      blue moving | green charging | dim white idle   (purple listening, amber focus if enabled)
    screen    phone status, charger state, focus timer
    extras    (off by default) knock to talk: config.ENABLE_KNOCK, TV remote: config.ENABLE_IR

Uses the `freewili` Python library (pip install freewili, Python 3.10+). Talks USB serial.
The panel runs in its own thread, which owns the device; the rest of the desk only flags
"redraw" on the bus, so nothing else touches the serial port.

Test the board on its own:
    python freewili_panel.py --test         # LEDs cycle, text on screen, prints button presses
    python freewili_panel.py --learn-ir     # press remote buttons, copy the codes into config.IR_MAP
    python freewili_panel.py --accel        # print accelerometer force, to tune config.KNOCK_G
"""

import math
import threading
import time

import config
from desk_bus import bus, state

N_LEDS = 7
COLORS = {
    "listening": (90, 0, 120),
    "moving": (0, 40, 160),
    "charging": (0, 120, 20),
    "focus": (140, 70, 0),
    "idle": (12, 12, 12),
    "off": (0, 0, 0),
}
SPEAK_ACTIONS = {"focus_mode", "stop_focus", "park_charger", "follow_phone", "phone_status", "energy_report"}
BUTTONS = ("gray", "yellow", "green", "blue", "red")


def ok(result, what):
    """The freewili library returns Ok/Err results; log errors instead of raising."""
    if result is not None and hasattr(result, "is_err") and result.is_err():
        print(f"[freewili] {what} failed: {result.err()}")
        return False
    return True


class Panel:
    def __init__(self, brain=None, fw=None):
        self.brain = brain
        self.fw = fw
        self.buttons = {b: False for b in BUTTONS}
        self.dirty = threading.Event()
        self.dirty.set()
        self.last_leds = None
        self.last_text = None
        self.last_text_t = 0.0
        self.knocks = []
        self.accel_scale = 16384.0          # raw counts per g at the 2g range
        self.learn_ir = False
        self.print_accel = False
        bus.on("state_changed", lambda **kw: self.dirty.set())

    # ---------------------------------------------------------------- device
    def connect(self):
        from freewili import FreeWili
        if self.fw is None:
            self.fw = FreeWili.find_first().expect(
                "No FREE-WILi found (plugged in? firmware compatible with the freewili library?)")
        self.fw.open().expect("Could not open the FREE-WILi")
        print(f"[freewili] connected: {self.fw}")
        self.fw.set_event_callback(self.on_event)
        ok(self.fw.enable_button_events(True, 33), "button events")
        if config.ENABLE_IR or self.learn_ir:
            ok(self.fw.enable_ir_events(True), "IR events")
        if (config.ENABLE_KNOCK and config.ENABLE_VOICE) or self.print_accel:
            ok(self.fw.enable_accel_events(True, 33), "accel events")
        return self

    def close(self):
        if self.fw is None:
            return
        for name in ("enable_button_events", "enable_ir_events", "enable_accel_events"):
            try:
                getattr(self.fw, name)(False)
            except Exception:
                pass
        self.set_leds("off")
        self.fw.close()

    def start(self):
        self.connect()
        threading.Thread(target=self.run, name="freewili", daemon=True).start()
        return self

    def run(self):
        while True:
            try:
                self.fw.process_events()
                if self.dirty.is_set() or state.focus_until or state.moving:
                    self.render()
            except Exception as e:
                print("[freewili] loop error:", e)
                time.sleep(1)
            time.sleep(0.02)

    # ---------------------------------------------------------------- inputs
    def on_event(self, event_type, frame, data):
        name = getattr(event_type, "name", str(event_type))
        if name == "Button":
            self.on_buttons(data)
        elif name == "IR":
            self.on_ir(data)
        elif name == "Accel":
            self.on_accel(data)

    def on_buttons(self, data):
        for b in BUTTONS:
            now = bool(getattr(data, b, False))
            was = self.buttons[b]
            self.buttons[b] = now
            if now == was:
                continue
            action = config.FREEWILI_BUTTONS.get(b)
            print(f"[freewili] {b} {'pressed' if now else 'released'} -> {action}")
            if action == "push_to_talk":
                if now:
                    ok(self.fw.play_audio_tone(880, 0.08, 0.4), "beep")
                    bus.emit("listen_start", hold=True)
                else:
                    bus.emit("listen_stop")
            elif now and action:
                self.trigger(action)

    def on_ir(self, data):
        code = "0x" + bytes(getattr(data, "value", b"")).hex().upper()
        action = config.IR_MAP.get(code)
        if self.learn_ir or action is None:
            print(f"[freewili] IR code {code}" + ("" if action else "  (not mapped: add it to config.IR_MAP)"))
        if action and not self.learn_ir:
            if action == "listen":
                bus.emit("listen_start")
            else:
                self.trigger(action)

    def on_accel(self, data):
        rng = getattr(data, "g", 2.0) or 2.0
        self.accel_scale = 32768.0 / rng
        mag = math.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2) / self.accel_scale
        if self.print_accel:
            print(f"[freewili] |a| = {mag:.2f} g")
        if mag < config.KNOCK_G:
            return
        now = time.time()
        self.knocks = [t for t in self.knocks if now - t < config.KNOCK_WINDOW_S] + [now]
        # ignore samples of the same knock (< 120 ms apart)
        if len(self.knocks) >= 2 and self.knocks[-1] - self.knocks[-2] > 0.12:
            print("[freewili] double knock -> listening")
            self.knocks = []
            bus.emit("listen_start")

    def trigger(self, action):
        """Run an action off the panel thread (Spotify calls take a moment)."""
        if self.brain is None:
            return

        def run():
            reply = self.brain.run_action(action)
            if reply and action in SPEAK_ACTIONS:
                bus.emit("say", text=reply)
        threading.Thread(target=run, daemon=True).start()

    # ---------------------------------------------------------------- outputs
    def set_leds(self, mode, phase=0.0):
        r, g, b = COLORS[mode]
        if mode == "moving":                     # chase along the strip while the gantry moves
            colors = []
            for i in range(N_LEDS):
                k = 0.15 + 0.85 * max(0.0, math.cos((i / N_LEDS - phase) * 2 * math.pi)) ** 4
                colors.append((int(r * k), int(g * k), int(b * k)))
        else:
            colors = [(r, g, b)] * N_LEDS
        if colors == self.last_leds:
            return
        for i, (cr, cg, cb) in enumerate(colors):
            ok(self.fw.set_board_leds(i, cr, cg, cb), "LED")
        self.last_leds = colors

    def screen_text(self):
        s = state.summary()
        lines = ["SMART DESK", "Phone: " + s["phone"]]
        if s["focus"]:
            lines.append("FOCUS " + s["focus"])
        if s["listening"]:
            lines.append("Listening...")
        elif not s["tracking"]:
            lines.append("Charger parked")
        return "\n".join(lines)

    def render(self):
        self.dirty.clear()
        if state.listening:
            mode = "listening"
        elif state.moving:
            mode = "moving"
        elif state.coil_on:
            mode = "charging"
        elif state.focus_until:
            mode = "focus"
        else:
            mode = "idle"
        self.set_leds(mode, phase=(time.time() * 1.5) % 1.0)
        text = self.screen_text()
        if text != self.last_text and time.time() - self.last_text_t > 0.5:
            ok(self.fw.show_text_display(text), "display")
            self.last_text, self.last_text_t = text, time.time()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--learn-ir", action="store_true")
    ap.add_argument("--accel", action="store_true")
    a = ap.parse_args()
    if not (a.test or a.learn_ir or a.accel):
        raise SystemExit(__doc__)
    p = Panel()
    p.learn_ir, p.print_accel = a.learn_ir, a.accel
    p.connect()
    bus.on("listen_start", lambda **kw: print("[test] listen_start", kw))
    bus.on("listen_stop", lambda **kw: print("[test] listen_stop"))
    try:
        if a.test:
            ok(p.fw.show_text_display("SMART DESK\nFREE-WILi test"), "display")
            for mode in ("listening", "moving", "charging", "focus", "idle"):
                print("[test] LEDs:", mode)
                p.set_leds(mode)
                time.sleep(0.7)
            print("[test] press buttons / knock twice (Ctrl+C to quit)")
        while True:
            p.fw.process_events()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        p.close()


if __name__ == "__main__":
    main()
