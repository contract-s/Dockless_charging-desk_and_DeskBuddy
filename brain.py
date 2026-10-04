"""
brain.py  —  turns a command ("play some jazz and park the charger") into desk actions.

Every input goes through here: voice, FREE-WILi buttons, IR remote, the Fetch.ai / ASI:One agent,
and typed text. ASI:One (Fetch.ai's LLM) picks the actions; if it is unreachable or slow, simple
keyword rules take over so the desk always responds.

Test without any hardware:
    python brain.py              # type commands; motion is a dry run, Spotify is real if set up
    python brain.py --no-llm     # keyword rules only
"""

import json
import re
import threading
import time

import config
from desk_bus import bus, state

# name -> (argument hint, description)   (shown to the LLM)
ACTIONS = {
    "play_music": ('{"query": str}', 'play something on Spotify. query = genre/mood/playlist/"song by artist"/"liked"'),
    "pause_music": ("{}", "pause the music"),
    "resume_music": ("{}", "resume the music"),
    "toggle_music": ("{}", "play/pause"),
    "skip": ("{}", "next track"),
    "previous": ("{}", "previous track"),
    "volume": ('{"level": int 0-100}', "set the volume"),
    "volume_change": ('{"delta": int}', "louder (+15) / quieter (-15)"),
    "now_playing": ("{}", "say what song is playing"),
    "park_charger": ("{}", "send the charger home and stop following the phone"),
    "follow_phone": ("{}", "start following the phone again / bring the charger to the phone"),
    "phone_status": ("{}", "where is the phone, is it charging"),
    "focus_mode": ('{"minutes": int}', "start a focus session: focus music + countdown on the panel"),
    "stop_focus": ("{}", "end the focus session"),
    "energy_report": ("{}", "how much energy the desk saved by only powering the coil when a phone is there"),
}

SYSTEM_PROMPT = """You control a smart desk. A camera finds the user's phone and a motorised wireless
charger slides under it. The desk also plays the user's Spotify and has a focus timer.

Available actions (name: args - what it does):
{actions}

Current desk state: {state}

Reply with ONLY a JSON object, no prose around it:
{{"actions": [{{"name": "<action>", "args": {{...}}}}], "reply": "<one short friendly sentence to say out loud>"}}
Use an empty actions list for small talk. Never invent action names."""


class Brain:
    def __init__(self, spotify=None, use_llm=True):
        self._spotify = spotify
        self._spotify_error = None
        self.use_llm = use_llm and bool(config.ASI1_API_KEY)
        self.lock = threading.Lock()
        self.focus_timer = None
        self._llm = None

    # ---------------------------------------------------------------- helpers
    @property
    def spotify(self):
        if self._spotify is None and self._spotify_error is None:
            try:
                from spotify_control import Spotify
                self._spotify = Spotify()
            except Exception as e:
                self._spotify_error = str(e)
                print("[brain] Spotify unavailable:", e)
        if self._spotify is None:
            raise RuntimeError("Spotify isn't set up")
        return self._spotify

    def _llm_client(self):
        if self._llm is None:
            from openai import OpenAI
            self._llm = OpenAI(base_url=config.ASI1_BASE_URL, api_key=config.ASI1_API_KEY,
                               timeout=config.LLM_TIMEOUT_S, max_retries=0)
        return self._llm

    def refresh_music_state(self):
        try:
            playing, title = self.spotify.now_playing()
            with state.lock:
                state.is_playing, state.now_playing = playing, title
        except Exception:
            pass
        bus.emit("state_changed")

    # ---------------------------------------------------------------- parsing
    def plan_llm(self, text):
        actions = "\n".join(f"- {n}: {a} - {d}" for n, (a, d) in ACTIONS.items())
        msg = SYSTEM_PROMPT.format(actions=actions, state=json.dumps(state.summary()))
        r = self._llm_client().chat.completions.create(
            model=config.ASI1_MODEL,
            messages=[{"role": "system", "content": msg}, {"role": "user", "content": text}],
            temperature=0.2, max_tokens=300,
        )
        raw = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise ValueError(f"no JSON in LLM reply: {raw[:120]!r}")
        plan = json.loads(m.group(0))
        acts = [a for a in plan.get("actions", []) if isinstance(a, dict) and a.get("name") in ACTIONS]
        return acts, str(plan.get("reply") or "")

    @staticmethod
    def plan_keywords(text):
        t = text.lower().strip()
        acts = []

        def add(name, **args):
            acts.append({"name": name, "args": args})

        if re.search(r"\b(park|go home|home position|stop following)\b", t):
            add("park_charger")
        if re.search(r"\b(follow|find my phone|come back|charge my phone|bring the charger)\b", t):
            add("follow_phone")
        m = re.search(r"\bfocus\b(?:.*?(\d+)\s*min)?", t)
        if m and not re.search(r"\b(stop|end|cancel)\b", t):
            add("focus_mode", minutes=int(m.group(1)) if m.group(1) else config.FOCUS_MINUTES)
        elif m:
            add("stop_focus")
        if re.search(r"\b(skip|next)\b", t):
            add("skip")
        elif re.search(r"\b(previous|go back|last song)\b", t):
            add("previous")
        m = re.search(r"\bvolume\D*(\d{1,3})", t)
        if m:
            add("volume", level=int(m.group(1)))
        elif re.search(r"\b(louder|turn it up|volume up)\b", t):
            add("volume_change", delta=15)
        elif re.search(r"\b(quieter|softer|turn it down|volume down)\b", t):
            add("volume_change", delta=-15)
        if re.search(r"\b(pause|stop the music|stop music|be quiet|shut up)\b", t):
            add("pause_music")
        elif re.search(r"\b(resume|unpause|continue)\b", t):
            add("resume_music")
        else:
            m = re.search(r"\bplay\b\s*(?:some|me|my)?\s*(.*)", t)
            if m and not any(a["name"] == "focus_mode" for a in acts):
                q = re.split(r"\s+(?:and|then)\s+|[,.!?]", m.group(1))[0].strip() or None
                add("play_music", query=q) if q else add("resume_music")
        if re.search(r"(what('s| is) (playing|this song)|which song|song is this)", t):
            add("now_playing")
        if re.search(r"(where('s| is) my phone|phone status|is (it|my phone) charging)", t):
            add("phone_status")
        if re.search(r"\b(energy|power|save|saved|sustainab)", t):
            add("energy_report")
        return acts

    # ---------------------------------------------------------------- running
    def handle_text(self, text, source="text"):
        """Command in, reply sentence out. Safe to call from any thread."""
        text = (text or "").strip()
        if not text:
            return ""
        with self.lock:
            t0 = time.time()
            acts, llm_reply, how = None, "", "keywords"
            if self.use_llm:
                try:
                    acts, llm_reply = self.plan_llm(text)
                    how = "asi1"
                except Exception as e:
                    print(f"[brain] ASI:One failed ({e}); using keywords")
            if acts is None:
                acts = self.plan_keywords(text)
            print(f"[brain] {source}: {text!r} -> {how} {acts} ({time.time() - t0:.1f}s)")

            results = []
            for a in acts:
                r = self.run_action(a["name"], a.get("args") or {})
                if r:
                    results.append(r)
            if results:
                reply = " ".join(results)
            elif llm_reply:
                reply = llm_reply
            elif acts:
                reply = "Done."
            else:
                reply = "Sorry, I didn't catch that."
            return reply

    def run_action(self, name, args=None):
        """Run one action by name -> short reply sentence ("" = nothing to say)."""
        args = args or {}
        fn = getattr(self, "act_" + name, None)
        if fn is None:
            return f"I don't know how to {name}."
        try:
            return fn(**args) or ""
        except TypeError:
            return fn() or ""
        except Exception as e:
            print(f"[brain] {name} failed: {e}")
            return f"Sorry, {e}."
        finally:
            bus.emit("state_changed")

    # ---- music ----
    def act_play_music(self, query=None):
        what = self.spotify.play(query or config.DEFAULT_PLAYLIST_QUERY)
        self.refresh_music_state()
        return f"Playing {what}."

    def act_pause_music(self):
        self.spotify.pause()
        with state.lock:
            state.is_playing = False
        return "Paused."

    def act_resume_music(self):
        self.spotify.resume()
        self.refresh_music_state()
        return ""

    def act_toggle_music(self):
        r = self.spotify.toggle()
        self.refresh_music_state()
        return "Paused." if r == "paused" else ""

    def act_skip(self):
        self.spotify.next()
        time.sleep(0.4)
        self.refresh_music_state()
        return ""

    def act_previous(self):
        self.spotify.previous()
        time.sleep(0.4)
        self.refresh_music_state()
        return ""

    def act_volume(self, level=50):
        self.spotify.volume(int(level))
        return f"Volume {int(level)}."

    def act_volume_change(self, delta=15):
        sp = self.spotify
        cur = sp.saved_volume if sp.saved_volume is not None else (sp.current_volume() or 50)
        new = max(0, min(100, cur + int(delta)))
        if sp.saved_volume is not None:      # ducked right now: apply after unduck
            sp.saved_volume = new
        else:
            sp.volume(new)
        return f"Volume {new}."

    def act_now_playing(self):
        playing, title = self.spotify.now_playing()
        return f"This is {title}." if title else "Nothing is playing."

    # ---- charger ----
    def act_park_charger(self):
        with state.lock:
            state.tracking_enabled = False
            m = state.mover
        if m is not None:
            m.park()
        return "Parking the charger."

    def act_follow_phone(self):
        with state.lock:
            state.tracking_enabled = True
            m, xy = state.mover, state.phone_xy if state.phone_present else None
        if m is not None and xy:
            m.goto_desk(*xy, force=True)
            return "Coming to your phone."
        return "Following your phone again."

    def act_phone_status(self):
        return f"Your phone is {state.summary()['phone']}."

    # ---- focus ----
    def act_focus_mode(self, minutes=None):
        minutes = int(minutes or config.FOCUS_MINUTES)
        if self.focus_timer:
            self.focus_timer.cancel()
        with state.lock:
            state.focus_until = time.time() + minutes * 60
        self.focus_timer = threading.Timer(minutes * 60, self._focus_done)
        self.focus_timer.daemon = True
        self.focus_timer.start()
        try:
            self.spotify.play(config.FOCUS_PLAYLIST_QUERY)
            self.refresh_music_state()
        except Exception as e:
            print("[brain] focus music failed:", e)
        return f"Focus mode for {minutes} minutes. I'll let you know when it's time for a break."

    def act_stop_focus(self):
        if self.focus_timer:
            self.focus_timer.cancel()
            self.focus_timer = None
        with state.lock:
            state.focus_until = None
        return "Focus session ended."

    def _focus_done(self):
        with state.lock:
            state.focus_until = None
        bus.emit("state_changed")
        bus.emit("say", text="Focus session done. Nice work, take a five minute break.")

    # ---- sustainability ----
    def act_energy_report(self):
        r = state.energy_report()
        return (f"The coil has only been powered for {r['hours_charging']:.1f} of the last "
                f"{r['hours_tracked']:.1f} hours, across {r['sessions']} charging sessions. "
                f"Compared with a pad left on all the time, that's {r['idle_wh_saved']:.1f} watt-hours "
                f"of idle power not wasted.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="keyword rules only")
    ap.add_argument("--dry-run", action="store_true", help="connect a dry-run motion controller")
    a = ap.parse_args()
    if a.dry_run:
        import motion
        state.mover = motion.Mover(dry_run=True).connect()
    b = Brain(use_llm=not a.no_llm)
    print(f"[brain] LLM: {'ASI:One' if b.use_llm else 'off (keywords)'}. Type a command, Ctrl+C to quit.")
    try:
        while True:
            print(">>", b.handle_text(input("you: "), source="stdin"))
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
