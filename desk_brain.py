"""
desk_brain.py  —  runs the whole smart desk.

    camera tracker  ->  gantry moves the charger under the phone
    phone placed    ->  coil on, music resumes, "welcome back"
    phone picked up ->  coil off, music pauses
    voice (ElevenLabs), FREE-WILi panel, typed commands and the Fetch.ai agent all drive brain.py

Run:
    python desk_brain.py                      # everything
    python desk_brain.py --dry-run            # no ESP32: motion commands are only printed
    python desk_brain.py --no-camera --dry-run
        # no camera either. Type commands, or simulate the phone:  !placed 30 20   /   !lost

Start fetch_agent.py in a second terminal to control the desk from ASI:One.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from brain import Brain
from desk_bus import bus, state


def bg(fn, *args, **kw):
    """Run fn in a thread so the camera loop never waits on the network."""
    def run():
        try:
            fn(*args, **kw)
        except Exception as e:
            print(f"[desk] {getattr(fn, '__name__', fn)} failed: {e}")
    threading.Thread(target=run, daemon=True).start()


class Desk:
    def __init__(self, brain):
        self.brain = brain
        self.last_greet = 0.0
        bus.on("phone_placed", self.on_placed)
        bus.on("phone_moved", self.on_moved)
        bus.on("phone_lost", self.on_lost)

    def _coil(self, on):
        state.set_coil(on)
        m = state.mover
        if m is not None:
            bg(m.coil, on)

    def on_placed(self, x, y, **kw):
        with state.lock:
            state.phone_present, state.phone_xy, state.phone_since = True, (x, y), time.time()
        print(f"[desk] phone placed at ({x:.1f}, {y:.1f}) cm")
        self._coil(True)
        bus.emit("state_changed")
        if config.AUTO_MUSIC_ON_PLACE:
            bg(self._welcome)

    def _welcome(self):
        now = time.time()
        greet = now - self.last_greet > config.GREET_EVERY_S
        if greet:
            self.last_greet = now
            bus.emit("say", text="Welcome back. Charging your phone.")
        try:
            sp = self.brain.spotify
            if not sp.is_playing():
                sp.resume()
            self.brain.refresh_music_state()
        except Exception as e:
            print("[desk] could not resume music:", e)

    def on_moved(self, x, y, **kw):
        with state.lock:
            state.phone_xy = (x, y)
        state.moving = True
        bus.emit("state_changed")
        bg(self._settle)

    def _settle(self):
        time.sleep(1.5)
        state.moving = False
        bus.emit("state_changed")

    def on_lost(self, **kw):
        with state.lock:
            state.phone_present = False
        print("[desk] phone picked up")
        self._coil(False)
        bus.emit("state_changed")
        if config.PAUSE_ON_PICKUP:
            bg(self._pause)

    def _pause(self):
        try:
            self.brain.spotify.pause()
            self.brain.refresh_music_state()
        except Exception as e:
            print("[desk] could not pause music:", e)


# ---------------------------------------------------------------- command server (for fetch_agent.py)
def start_command_server(brain):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/status"):
                self._send(200, state.summary())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/command"):
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
            except ValueError:
                return self._send(400, {"error": "send JSON {\"text\": ...}"})
            reply = brain.handle_text(text, source="agent")
            if config.SPEAK_AGENT_REPLIES and reply:
                bus.emit("say", text=reply)
            self._send(200, {"reply": reply, "state": state.summary()})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(config.COMMAND_SERVER, Handler)
    threading.Thread(target=srv.serve_forever, name="command-server", daemon=True).start()
    print(f"[desk] command server on http://{config.COMMAND_SERVER[0]}:{config.COMMAND_SERVER[1]}")
    return srv


def music_poller(brain):
    """Keep 'now playing' fresh for the FREE-WILi screen."""
    while True:
        time.sleep(5)
        try:
            brain.refresh_music_state()
        except Exception:
            pass


def stdin_loop(brain):
    while True:
        try:
            line = input().strip()
        except EOFError:
            return
        if not line:
            continue
        if line.startswith("!placed"):
            parts = line.split()
            x, y = (float(parts[1]), float(parts[2])) if len(parts) >= 3 else (30.0, 22.0)
            bus.emit("phone_placed", x=x, y=y)
            if state.tracking_enabled and state.mover is not None:
                state.mover.goto_desk(x, y, force=True)
            bus.emit("phone_moved", x=x, y=y)
        elif line.startswith("!lost"):
            bus.emit("phone_lost")
        elif line.startswith("!listen"):
            bus.emit("listen_start")
        else:
            print(">>", brain.handle_text(line, source="typed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="don't talk to the ESP32, print motion commands")
    ap.add_argument("--port", default="auto", help="ESP32 serial port (default: auto-detect)")
    ap.add_argument("--home", action="store_true", help="home with limit switches instead of assuming HOME")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-panel", action="store_true", help="don't look for a FREE-WILi")
    ap.add_argument("--no-llm", action="store_true", help="keyword rules instead of ASI:One")
    a = ap.parse_args()

    import motion
    try:
        m = motion.Mover(port=a.port, dry_run=a.dry_run).connect()
        if a.home:
            m.home()
        else:
            print("[desk] assuming the charger is at HOME now (push it there by hand if not)")
            m.zero()
    except Exception as e:
        print(f"[desk] no gantry ({e}); continuing with a dry-run mover")
        m = motion.Mover(dry_run=True).connect()
    state.mover = m

    brain = Brain(use_llm=not a.no_llm)
    print(f"[desk] command AI: {'ASI:One' if brain.use_llm else 'keywords'}")
    Desk(brain)

    if not a.no_voice:
        from voice import Voice
        Voice(brain).start()
    if not a.no_panel:
        try:
            from freewili_panel import Panel
            Panel(brain).start()
        except Exception as e:
            print(f"[desk] FREE-WILi panel off ({e})")

    start_command_server(brain)
    threading.Thread(target=music_poller, args=(brain,), daemon=True).start()
    threading.Thread(target=stdin_loop, args=(brain,), daemon=True).start()
    print("[desk] ready. Type a command any time (or !placed x y / !lost / !listen).")

    try:
        if a.no_camera:
            while True:
                time.sleep(1)
        else:
            import phone_tracker_screen_off as tracker   # cv2 windows must stay on the main thread
            tracker.main(on_event=lambda name, **kw: bus.emit(name, **kw), state=state)
    except KeyboardInterrupt:
        pass
    finally:
        state.set_coil(False)
        m.close()


if __name__ == "__main__":
    main()
