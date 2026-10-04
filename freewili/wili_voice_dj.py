"""
WiLi Voice DJ - say a song name into the FREE-WILi, laptop plays the MP3.

Commands:  "play <song name>"   "stop"   "pause"   "resume"   "next" / "shuffle"
Hold the GREEN button while speaking, then release.
"""

import difflib
import json
import pathlib
import queue
import random

import numpy as np
import pygame
from vosk import KaldiRecognizer, Model

from freewili import FreeWili
from freewili.framing import ResponseFrame
from freewili.types import AudioData, ButtonData, EventType

# ---------------- settings ----------------
MUSIC_DIR = pathlib.Path(r"C:\Users\suta2\Documents\freeWili\music")   # <-- folder with your .mp3 files
MODEL_DIR = "vosk-model-small-en-us-0.15"
PUSH_TO_TALK = True    # hold the GREEN button while speaking
FW_RATE = 8000         # FREE-WILi audio events are 16-bit mono @ 8 kHz
ASR_RATE = 16000       # Vosk models are trained on 16 kHz
# ------------------------------------------

songs = {
    p.stem.lower().replace("_", " ").replace("-", " "): p
    for p in MUSIC_DIR.rglob("*.mp3")
}
if not songs:
    raise SystemExit(f"No .mp3 files found in {MUSIC_DIR}")
print(f"Found {len(songs)} songs.")

pygame.mixer.init()
recognizer = KaldiRecognizer(Model(MODEL_DIR), ASR_RATE)

commands = queue.Queue()
state = {"listening": not PUSH_TO_TALK, "green": False, "dc": 0.0}


def to_pcm16k(samples):
    """Remove DC offset, upsample 8 kHz -> 16 kHz, return int16 bytes."""
    arr = np.asarray(samples, dtype=np.float64)
    state["dc"] = 0.95 * state["dc"] + 0.05 * arr.mean()
    arr -= state["dc"]
    n = len(arr)
    up = np.interp(np.arange(n * 2) / 2, np.arange(n), arr)
    return np.clip(up, -32768, 32767).astype("<i2").tobytes()


def on_event(event_type: EventType, frame: ResponseFrame, data) -> None:
    if event_type == EventType.Audio and isinstance(data, AudioData):
        if state["listening"] and data.data:
            if recognizer.AcceptWaveform(to_pcm16k(data.data)) and not PUSH_TO_TALK:
                commands.put(json.loads(recognizer.Result()).get("text", ""))

    elif event_type == EventType.Button and isinstance(data, ButtonData) and PUSH_TO_TALK:
        pressed = bool(data.green)
        if pressed and not state["green"]:
            recognizer.Reset()
            state["listening"] = True
            pygame.mixer.music.set_volume(0.2)
            print("Listening...")
        elif not pressed and state["green"]:
            state["listening"] = False
            commands.put(json.loads(recognizer.FinalResult()).get("text", ""))
            pygame.mixer.music.set_volume(1.0)
        state["green"] = pressed


def beep(fw, ok):
    fw.play_audio_tone(880 if ok else 220, 0.15, 0.5)


def play(path):
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    print(f"Now playing: {path.name}")


def handle(fw, text):
    text = text.strip().lower()
    if not text:
        return
    print(f"Heard: '{text}'")

    if text in ("stop", "stop music"):
        pygame.mixer.music.stop(); beep(fw, True)
    elif text in ("pause", "pause music"):
        pygame.mixer.music.pause(); beep(fw, True)
    elif text in ("resume", "continue", "unpause"):
        pygame.mixer.music.unpause(); beep(fw, True)
    elif text in ("next", "shuffle", "random", "play something"):
        play(random.choice(list(songs.values()))); beep(fw, True)
    else:
        query = text.removeprefix("play").strip()
        match = difflib.get_close_matches(query, songs.keys(), n=1, cutoff=0.4)
        if not match:
            match = [s for s in songs if any(w in s for w in query.split() if len(w) > 2)][:1]
        if match:
            play(songs[match[0]]); beep(fw, True)
        else:
            print(f"No song matches '{query}'"); beep(fw, False)


def main():
    fw = FreeWili.find_first().expect("Failed to find FreeWili")
    with fw:
        print(f"Connected to {fw}")
        fw.set_event_callback(on_event)
        fw.enable_audio_events(True).expect("Failed to enable audio events")
        if PUSH_TO_TALK:
            fw.enable_button_events(True, 33).expect("Failed to enable button events")
            print("Hold the GREEN button, say e.g. 'play bohemian rhapsody', then release.")
        try:
            while True:
                fw.process_events()
                while not commands.empty():
                    handle(fw, commands.get())
        except KeyboardInterrupt:
            print("\nBye!")
        finally:
            fw.enable_audio_events(False)
            if PUSH_TO_TALK:
                fw.enable_button_events(False)
            pygame.mixer.music.stop()


if __name__ == "__main__":
    main()