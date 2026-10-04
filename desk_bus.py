"""
desk_bus.py  —  shared state + a tiny event bus, so the tracker, voice, FREE-WILi panel and
the Fetch.ai agent all see the same desk.

    from desk_bus import bus, state
    bus.on("phone_placed", lambda **kw: print("placed at", kw["x"], kw["y"]))
    bus.emit("phone_placed", x=30.0, y=22.0)

Events used:
    phone_placed(x, y)   phone_moved(x, y)   phone_lost()
    listen_start()       listen_stop()        state_changed()
    say(text)
"""

import json
import os
import threading
import time
from collections import defaultdict

import config


class Bus:
    def __init__(self):
        self._subs = defaultdict(list)
        self._lock = threading.Lock()

    def on(self, event, fn):
        with self._lock:
            self._subs[event].append(fn)

    def emit(self, event, **kw):
        with self._lock:
            subs = list(self._subs[event])
        for fn in subs:
            try:
                fn(**kw)
            except Exception as e:
                print(f"[bus] {event} handler {getattr(fn, '__name__', fn)} failed: {e}")


class DeskState:
    def __init__(self):
        self.lock = threading.RLock()
        self.phone_present = False
        self.phone_xy = None              # desk cm
        self.phone_since = None
        self.tracking_enabled = True      # False after "park": don't follow the phone
        self.coil_on = False
        self.moving = False
        self.listening = False
        self.now_playing = ""
        self.is_playing = False
        self.focus_until = None           # epoch seconds, or None
        self.mover = None                 # motion.Mover (or grbl Gantry), set by desk_brain
        self.energy = self._load_energy()
        self._coil_on_at = None

    # ---- energy bookkeeping (Sustainability) ----
    def _load_energy(self):
        try:
            with open(config.ENERGY_FILE) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"coil_on_s": 0.0, "started": time.time(), "sessions": 0}

    def _save_energy(self):
        try:
            with open(config.ENERGY_FILE, "w") as f:
                json.dump(self.energy, f)
        except OSError:
            pass

    def set_coil(self, on):
        with self.lock:
            if on == self.coil_on:
                return
            self.coil_on = on
            now = time.time()
            if on:
                self._coil_on_at = now
                self.energy["sessions"] += 1
            elif self._coil_on_at is not None:
                self.energy["coil_on_s"] += now - self._coil_on_at
                self._coil_on_at = None
                self._save_energy()

    def energy_report(self):
        """-> dict with hours tracked, hours charging, Wh used, Wh an always-on pad would waste."""
        with self.lock:
            now = time.time()
            on_s = self.energy["coil_on_s"] + ((now - self._coil_on_at) if self._coil_on_at else 0.0)
            total_s = max(1.0, now - self.energy["started"])
        idle_s = max(0.0, total_s - on_s)
        always_on_wh = (idle_s * config.COIL_IDLE_W) / 3600.0
        return dict(hours_tracked=total_s / 3600.0, hours_charging=on_s / 3600.0,
                    sessions=self.energy["sessions"], idle_wh_saved=always_on_wh)

    def summary(self):
        with self.lock:
            if self.phone_present and self.phone_xy:
                phone = f"on the desk at ({self.phone_xy[0]:.0f}, {self.phone_xy[1]:.0f}) cm"
                phone += ", charging" if self.coil_on else ""
            else:
                phone = "not on the desk"
            focus = None
            if self.focus_until:
                left = int(self.focus_until - time.time())
                focus = f"{left // 60}:{left % 60:02d}" if left > 0 else None
            return dict(phone=phone, tracking=self.tracking_enabled, coil=self.coil_on,
                        music=self.now_playing if self.is_playing else "paused",
                        focus=focus, listening=self.listening)


bus = Bus()
state = DeskState()
