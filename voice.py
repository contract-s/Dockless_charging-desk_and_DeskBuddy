"""
voice.py  —  the desk's ears and mouth, powered by ElevenLabs.

    listen:  mic -> ElevenLabs Scribe (speech to text) -> brain.handle_text() -> reply
    speak:   reply -> ElevenLabs TTS (Flash, low latency) -> speakers

A listen starts when something emits "listen_start" on the bus: the FREE-WILi green button
(held = push-to-talk), a double knock on the desk, or the spacebar in the camera window.
Music is ducked while the desk listens and talks.

Test:
    python voice.py --say "desk online"
    python voice.py --listen            # speak after the beep; prints the transcript
    python voice.py --loop              # Enter = listen, then runs the command through brain.py
"""

import io
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

import config
from desk_bus import bus, state

CHUNK_S = 0.1
TTS_RATE = 22050


class Voice:
    def __init__(self, brain=None):
        self.brain = brain
        self.jobs = queue.Queue()
        self.ptt = threading.Event()        # set while the push-to-talk button is held
        self.stop_now = threading.Event()
        self._client = None
        self.busy = False
        bus.on("listen_start", self._on_listen_start)
        bus.on("listen_stop", self._on_listen_stop)
        bus.on("say", lambda text, **kw: self.jobs.put(("say", text)))

    # ---------------------------------------------------------------- plumbing
    @property
    def client(self):
        if self._client is None:
            if not config.ELEVENLABS_API_KEY:
                raise RuntimeError("ELEVENLABS_API_KEY is not set")
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        return self._client

    def _on_listen_start(self, hold=False, **kw):
        if self.busy:
            return
        if hold:
            self.ptt.set()
        self.stop_now.clear()
        self.jobs.put(("listen", hold))

    def _on_listen_stop(self, **kw):
        self.ptt.clear()

    def start(self):
        threading.Thread(target=self._worker, name="voice", daemon=True).start()
        return self

    def _worker(self):
        while True:
            kind, arg = self.jobs.get()
            try:
                if kind == "listen":
                    self.listen_and_act(hold=arg)
                elif kind == "say":
                    self.speak(arg, duck=True)
            except Exception as e:
                print(f"[voice] {kind} failed: {e}")
            finally:
                self._set_listening(False)

    def _set_listening(self, on):
        with state.lock:
            changed = state.listening != on
            state.listening = on
        if changed:
            bus.emit("state_changed")

    def _spotify(self):
        try:
            return self.brain.spotify if self.brain else None
        except Exception:
            return None

    # ---------------------------------------------------------------- ears
    @staticmethod
    def beep(freq=880, dur=0.12):
        import sounddevice as sd
        t = np.linspace(0, dur, int(TTS_RATE * dur), False)
        tone = 0.25 * np.sin(2 * np.pi * freq * t) * np.hanning(t.size)
        sd.play(tone.astype(np.float32), TTS_RATE)
        sd.wait()

    def record(self, hold=False):
        """Record one command -> WAV bytes (or None if nothing was said)."""
        import sounddevice as sd
        import soundfile as sf

        rate, n = config.SAMPLE_RATE, int(config.SAMPLE_RATE * CHUNK_S)
        chunks, floor, heard, quiet_s, t0 = [], None, False, 0.0, time.time()
        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=n) as stream:
            while True:
                data, _ = stream.read(n)
                chunks.append(data.copy())
                rms = float(np.sqrt(np.mean(data ** 2)))
                elapsed = time.time() - t0
                if floor is None or elapsed < 0.3:
                    floor = rms if floor is None else 0.5 * (floor + rms)
                    continue
                loud = rms > max(floor * 3.0, 0.012)
                heard |= loud
                quiet_s = 0.0 if loud else quiet_s + CHUNK_S
                if self.stop_now.is_set() or elapsed >= config.LISTEN_MAX_S:
                    break
                if hold:
                    if not self.ptt.is_set() and elapsed > 0.4:
                        break
                elif heard and quiet_s >= config.LISTEN_SILENCE_S:
                    break
                elif not heard and elapsed > 4.0:
                    break
        if not heard and not hold:
            return None
        buf = io.BytesIO()
        sf.write(buf, np.concatenate(chunks), rate, format="WAV")
        buf.seek(0)
        buf.name = "command.wav"
        return buf

    def transcribe(self, wav):
        r = self.client.speech_to_text.convert(file=wav, model_id=config.ELEVENLABS_STT_MODEL,
                                               language_code="eng")
        return (getattr(r, "text", "") or "").strip()

    def listen(self, hold=False):
        """Beep, record, transcribe -> text ('' if nothing heard)."""
        self._set_listening(True)
        self.beep()
        wav = self.record(hold=hold)
        self._set_listening(False)
        if wav is None:
            return ""
        self.beep(660, 0.08)
        t0 = time.time()
        text = self.transcribe(wav)
        print(f"[voice] heard: {text!r} ({time.time() - t0:.1f}s)")
        return text

    def listen_and_act(self, hold=False):
        self.busy = True
        sp = self._spotify()
        if sp:
            sp.duck()
        try:
            text = self.listen(hold=hold)
            if not text:
                return
            reply = self.brain.handle_text(text, source="voice") if self.brain else ""
            if reply:
                self.speak(reply)
        finally:
            if sp:
                sp.unduck()
            self.busy = False

    # ---------------------------------------------------------------- mouth
    def speak(self, text, duck=False):
        if not text:
            return
        print(f"[voice] say: {text}")
        sp = self._spotify() if duck else None
        if sp:
            sp.duck()
        try:
            try:
                audio = b"".join(self.client.text_to_speech.convert(
                    text=text, voice_id=config.ELEVENLABS_VOICE_ID,
                    model_id=config.ELEVENLABS_TTS_MODEL, output_format=f"pcm_{TTS_RATE}"))
                import sounddevice as sd
                pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(pcm, TTS_RATE)
                sd.wait()
            except Exception as e:
                print(f"[voice] ElevenLabs TTS failed ({e}); using the system voice")
                if shutil.which("say"):
                    subprocess.run(["say", text])
        finally:
            if sp:
                sp.unduck()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--say")
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--loop", action="store_true", help="Enter = listen and run the command")
    a = ap.parse_args()
    if a.say:
        Voice().speak(a.say)
    elif a.listen:
        print("transcript:", Voice().listen())
    elif a.loop:
        from brain import Brain
        v = Voice(Brain())
        print("Press Enter, then speak. Ctrl+C to quit.")
        try:
            while True:
                input()
                v.listen_and_act()
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
