"""
Desk Assistant - the FREE-WILi announces iPhone messages and reads them aloud on request.

iPhone --BLE/ANCS--> Bottlenose (desk_ancs.yaml firmware) --20-pin header--> FREE-WILi --USB--> this script

A new message beeps, turns the LEDs blue, shows the sender and listens ~8 s for
"read my message" (set AUTO_READ = True to read immediately instead). After reading it listens ~8 s for:
  "repeat" / "say that again"                    -> reads it again
  "summarize my last three messages from Siddu"  -> Gemini summary of that sender's last N
                                                    (max HISTORY_MAX; reads them out if no API key)
  "tell me my messages" (any time, e.g. after GREEN) -> unread ones, else the latest few
Calls: only missed calls are announced ("You have a missed call from X"); incoming calls are ignored.
With AUTO_READ = False it waits for "read my message" / "who is it from" / "ignore" instead.
GREEN button = listen any time, RED = dismiss.
REPLIES_ENABLED: voice replies via Photon (photon_bridge/, contacts.json) - off by default.

Needs the FREE-WILi UART at 115200 8N1 with no flow control (its default; don't save the
"BottleNose" Orca setting, which switches it to 3 Mbps). Run inside .venv.
"""

import collections
import contextlib
import difflib
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import pyttsx3
import requests
import serial
import serial.tools.list_ports
from vosk import KaldiRecognizer, Model

from freewili import FreeWili
from freewili.framing import ResponseFrame
from freewili.types import AudioData, ButtonData, EventType, UART1Data

# ---------------- settings ----------------
HERE = pathlib.Path(__file__).resolve().parent


def load_env(path):
    """Minimal .env loader (KEY=value lines); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


load_env(HERE / ".env")
MODEL_DIR = HERE / "vosk-model-small-en-us-0.15"
# "usb" = Bottlenose USB-C (VID 0x303A), "header" = via FREE-WILi 20-pin UART, "auto" = usb if plugged in.
# Header caveat: with mic/button streams also on, the FREE-WILi drops most header UART events.
MESSAGE_SOURCE = "auto"
# True = upload speech to the FREE-WILi speaker (playback only works with event streams paused).
SPEAK_ON_FREEWILI = True
TTS_RATE = 150              # pyttsx3 words/min; slower is clearer on the small 8 kHz speaker
VOICE_LEVEL = 0.5           # peak level 0-1; higher distorts ("static") on the FREE-WILi speaker
HIGHPASS_HZ = 250           # cut bass the tiny speaker can't reproduce (it just buzzes)
LOWPASS_HZ = 3400           # cut harsh highs near the 4 kHz limit of 8 kHz audio (telephone band)
ELEVENLABS_RATE = 22050     # fetch higher quality, downsample cleanly ourselves
VOICE_SPEED = 0.9           # ElevenLabs speaking speed (0.7-1.2)
# ElevenLabs voice: used when ELEVENLABS_API_KEY is set, else (or on any error) falls back to pyttsx3.
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE = os.environ.get("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2"
AUTO_READ = False           # True = read every new message aloud immediately; False = ask first
LISTEN_WINDOW_S = 8.0       # time to say "read my message" after a message arrives
REPLIES_ENABLED = False     # voice replies via Photon; off - Photon can't send from your own number
REPLY_WINDOW_S = 8.0        # time to say "repeat" / "summarize" (or "send message") after a read
DICTATION_SILENCE_S = 8.0   # dictation ends after this much silence
DICTATION_MAX_S = 60.0
HISTORY_MAX = 10            # messages remembered per sender (cap for "summarize my last N")
SUMMARY_DEFAULT_N = 5       # when you don't say how many
# Summaries (needs GEMINI_API_KEY in .env); tried in order, next one if busy.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest"]
CONTACTS_FILE = HERE / "contacts.json"   # {"Sender Name": "+1734...", ...} - phone/email per sender
PHOTON_BRIDGE = "http://127.0.0.1:8787"  # photon_bridge/server.mjs (started automatically)
MAX_BODY_CHARS = 250        # long messages take a while to upload to the FREE-WILi
DEDUPE_S = 30.0             # ANCS can re-send the same notification
FW_RATE = 8000              # FREE-WILi mic events and speaker WAVs: 16-bit mono @ 8 kHz
ASR_RATE = 16000            # Vosk models are trained on 16 kHz
TTS_RAW = HERE / "tts_raw.wav"
TTS_FW = HERE / "deskmsg.wav"  # 8.3 name on the FREE-WILi: "deskmsg"
NUM_LEDS = 7
# ------------------------------------------

GRAMMAR = [
    "read my message", "read my messages", "read message", "read it",
    "tell me my messages", "tell me my message", "what are my messages",
    "who is it from", "who sent it",
    "ignore", "dismiss", "not now",
    *(["send message", "send a message", "send a reply", "reply", "respond"] if REPLIES_ENABLED else []),
    "repeat", "repeat that", "repeat the message", "repeat the messages", "repeat message",
    "say that again", "can you repeat that",
    "summarize", "summarize my messages", "summarize my last messages",
    "[unk]",
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
APP_NAMES = {
    "com.apple.MobileSMS": "Messages",
    "net.whatsapp.WhatsApp": "WhatsApp",
    "com.tinyspeck.chatlyio": "Slack",
    "com.hammerandchisel.discord": "Discord",
    "com.apple.mobilemail": "Mail",
    "com.google.Gmail": "Gmail",
    "com.facebook.Messenger": "Messenger",
    "com.burbn.instagram": "Instagram",
}
CALL_APPS = {"com.apple.mobilephone", "com.apple.facetime"}   # announced as calls, not read as texts
ANSI = re.compile(r"\x1b\[[0-9;]*m")

lines = queue.Queue()      # complete text lines from the Bottlenose
heard = queue.Queue()      # recognized phrases
buttons = queue.Queue()    # "green" / "red" presses
state = {
    "mode": "idle",        # idle | listening | speaking | dictating
    "window": "read",      # what the listen window is for: read | reply
    "deadline": 0.0,
    "reply_to": None,      # sender of the last message read aloud
    "dictated": [],        # dictation text so far
    "last_voice": 0.0,     # last time dictation heard a new word
    "partial": "",
    "free_text": "",       # free-form transcript of the last command (for summarize)
    "dc": 0.0,
    "uart_buf": b"",
    "green": False,
    "red": False,
}
pending = []               # [(app, sender, body)]
recent = {}                # dedupe key -> time first seen
history = {}               # first name (lower) -> {"name": longest name seen, "msgs": deque[(time, body)]}
latest = collections.deque(maxlen=HISTORY_MAX)   # most recent messages from anyone: (app, sender, body)
ringing = {}               # first name (lower) -> time their incoming call was seen
MISSED_CALL_WINDOW_S = 120 # a missed call must follow an incoming call within this long
RECENT_TO_READ = 3         # "tell me my messages" with nothing unread reads this many recent ones
last_spoken = []           # what "read my message" said last, for "repeat"

model = Model(str(MODEL_DIR))
recognizer = KaldiRecognizer(model, ASR_RATE, json.dumps(GRAMMAR))   # commands only
dictation = KaldiRecognizer(model, ASR_RATE)                         # free speech for replies
tts = pyttsx3.init()
tts.setProperty("rate", TTS_RATE)


# ---------------- input ----------------
def to_pcm16k(samples):
    """Remove DC offset, upsample 8 kHz -> 16 kHz, return int16 bytes."""
    arr = np.asarray(samples, dtype=np.float64)
    state["dc"] = 0.95 * state["dc"] + 0.05 * arr.mean()
    arr -= state["dc"]
    n = len(arr)
    up = np.interp(np.arange(n * 2) / 2, np.arange(n), arr)
    return np.clip(up, -32768, 32767).astype("<i2").tobytes()


def feed_uart(data: bytes) -> None:
    """Header UART arrives in small chunks; split it back into lines."""
    state["uart_buf"] += data
    *complete, state["uart_buf"] = state["uart_buf"].split(b"\n")
    for raw in complete:
        lines.put(raw.decode(errors="replace"))
    if len(state["uart_buf"]) > 4096:  # garbage with no newline
        state["uart_buf"] = b""


def on_event(event_type: EventType, frame: ResponseFrame, data) -> None:
    if isinstance(data, UART1Data):
        if data.data:
            feed_uart(data.data)

    elif event_type == EventType.Audio and isinstance(data, AudioData):
        if state["mode"] == "listening" and data.data:
            pcm = to_pcm16k(data.data)
            # The free recognizer listens too, so "summarize my last three messages from
            # Siddu" keeps the number and name the command grammar can't capture.
            if dictation.AcceptWaveform(pcm):
                text = json.loads(dictation.Result()).get("text", "")
                if text:
                    state["free_text"] = text
            if recognizer.AcceptWaveform(pcm):
                heard.put(json.loads(recognizer.Result()).get("text", ""))
        elif state["mode"] == "dictating" and data.data:
            if dictation.AcceptWaveform(to_pcm16k(data.data)):
                text = json.loads(dictation.Result()).get("text", "")
                if text:
                    state["dictated"].append(text)
                    state["last_voice"] = time.time()
                    print(f"  ...{text}")
                state["partial"] = ""
            else:
                partial = json.loads(dictation.PartialResult()).get("partial", "")
                if partial and partial != state["partial"]:
                    state["partial"] = partial
                    state["last_voice"] = time.time()

    elif event_type == EventType.Button and isinstance(data, ButtonData):
        for color in ("green", "red"):
            pressed = bool(getattr(data, color))
            if pressed and not state[color]:
                buttons.put(color)
            state[color] = pressed


def usb_reader() -> None:
    """MESSAGE_SOURCE = "usb": read the Bottlenose's own USB serial port."""
    while True:
        port = next((p.device for p in serial.tools.list_ports.comports() if p.vid == 0x303A), None)
        if not port:
            time.sleep(2)
            continue
        try:
            # Don't assert DTR/RTS: on the ESP32-C6 USB-JTAG port that holds the chip in reset.
            s = serial.Serial(None, 115200, timeout=1)
            s.port, s.dtr, s.rts = port, False, False
            s.open()
            print(f"Reading Bottlenose on {port}")
            with s:
                while True:
                    line = s.readline()
                    if line:
                        lines.put(line.decode(errors="replace"))
        except serial.SerialException as e:
            print(f"Bottlenose serial error ({e}); retrying")
            time.sleep(2)


# ---------------- output ----------------
def leds(fw, r, g, b):
    for i in range(NUM_LEDS):
        fw.set_board_leds(i, r, g, b)


@contextlib.contextmanager
def streams_paused(fw):
    """The FREE-WILi plays no sound while it streams mic/button/UART events to the PC,
    so pause them around anything that uses the speaker. (Header messages arriving during
    the pause are dropped - keep clips short.)"""
    fw.enable_audio_events(False)
    fw.enable_button_events(False)
    if MESSAGE_SOURCE == "header":
        fw.enable_uart_events(False)
    try:
        yield
    finally:
        fw.enable_audio_events(True)
        fw.enable_button_events(True, 33)
        if MESSAGE_SOURCE == "header":
            fw.enable_uart_events(True)


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FW_RATE)
        w.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())


def upload_beeps(fw):
    """play_audio_tone() fails on this firmware, so beeps are short WAVs played like speech."""
    for name, hz in (("beephi", 880), ("beeplo", 330)):
        t = np.arange(int(FW_RATE * 0.15)) / FW_RATE
        env = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fade in/out, no clicks
        path = HERE / f"{name}.wav"
        write_wav(path, np.sin(2 * np.pi * hz * t) * env * 0.8 * 32767)
        fw.send_file(path, None, None)


def beep(fw, ok=True):
    with streams_paused(fw):
        fw.play_audio_file("beephi" if ok else "beeplo")
        time.sleep(0.4)                     # let the beep finish before the next sound command


def show(fw, text):
    fw.show_text_display(text)  # Result ignored: the screen is a nice-to-have


def resample(sig, src_rate, dst_rate):
    """Band-limited (FFT) resample."""
    n_out = int(round(len(sig) * dst_rate / src_rate))
    spec = np.fft.rfft(sig)[: n_out // 2 + 1]
    return np.fft.irfft(spec, n_out) * (n_out / len(sig))


def synth_elevenlabs(text):
    """Text -> (float samples, rate) from ElevenLabs. Fetched at 22.05 kHz (not 8 kHz) so
    finish_audio() does a clean downsample; ElevenLabs' own 8 kHz output sounds hissy."""
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}",
        params={"output_format": f"pcm_{ELEVENLABS_RATE}"},
        headers={"xi-api-key": ELEVENLABS_KEY},
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {"stability": 0.6, "similarity_boost": 0.75, "speed": VOICE_SPEED},
        },
        timeout=20,
    )
    r.raise_for_status()
    return np.frombuffer(r.content, dtype="<i2").astype(np.float64), ELEVENLABS_RATE


def finish_audio(sig, rate) -> float:
    """Shape speech for the FREE-WILi's tiny speaker, downsample to 8 kHz, write TTS_FW.
    Returns seconds. Telephone band (250-3400 Hz) with smooth edges + headroom: bass makes
    the speaker buzz, and energy near 4 kHz / clipping is what sounds like static."""
    spec = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1 / rate)
    gain = np.ones_like(f)
    low = f < HIGHPASS_HZ
    gain[low] = (f[low] / HIGHPASS_HZ) ** 2
    lo, hi = LOWPASS_HZ - 400, LOWPASS_HZ + 200          # raised-cosine roll-off
    band = (f > lo) & (f < hi)
    gain[band] = 0.5 * (1 + np.cos(np.pi * (f[band] - lo) / (hi - lo)))
    gain[f >= hi] = 0
    spec *= gain
    n_out = int(round(len(sig) * FW_RATE / rate))
    out = np.fft.irfft(spec[: n_out // 2 + 1], n_out) * (n_out / len(sig))
    fade = min(len(out) // 2, int(0.01 * FW_RATE))         # no clicks at start/end
    out[:fade] *= np.linspace(0, 1, fade)
    out[len(out) - fade:] *= np.linspace(1, 0, fade)
    out *= VOICE_LEVEL * 32767 / max(np.abs(out).max(), 1)
    write_wav(TTS_FW, out)
    return len(out) / FW_RATE


def synth(text) -> float:
    """Text -> TTS_FW (8 kHz mono int16 WAV). Returns clip length in seconds."""
    if ELEVENLABS_KEY:
        try:
            return finish_audio(*synth_elevenlabs(text))
        except Exception as e:
            print(f"  ElevenLabs failed ({e}); using the built-in voice")
    tts.save_to_file(text, str(TTS_RAW))
    tts.runAndWait()
    with wave.open(str(TTS_RAW), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    sig = np.frombuffer(raw, dtype="<i2" if width == 2 else np.uint8).astype(np.float64)
    if width == 1:
        sig = (sig - 128) * 256
    if ch > 1:
        sig = sig.reshape(-1, ch).mean(axis=1)
    return finish_audio(sig, rate)


def speak(fw, text):
    """Say text on the FREE-WILi speaker (or the laptop). The mic is ignored meanwhile."""
    print(f"Speaking: {text}")
    state["mode"] = "speaking"
    secs = synth(text)
    played = False
    if SPEAK_ON_FREEWILI:
        with streams_paused(fw):
            t0 = time.time()
            sent = fw.send_file(TTS_FW, None, None)
            ok = False
            for attempt in range(3):         # the FREE-WILi sometimes rejects play (busy/other screen)
                if sent.is_ok() and fw.play_audio_file(TTS_FW.stem).is_ok():
                    ok = True
                    break
                print(f"  play rejected (attempt {attempt + 1}); resetting display and retrying")
                fw.reset_display()           # leave any text/menu screen that may block audio
                time.sleep(0.7)
            if ok:
                print(f"  (uploaded in {time.time() - t0:.1f}s)")
                played = True
                time.sleep(secs + 0.3)   # streams must stay off until the clip finishes
            else:
                print(f"  FREE-WILi playback failed ({sent}); using laptop speakers")
    if not played:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.Sound(str(TTS_FW)).play()
        time.sleep(secs + 0.3)
    fw.process_events()        # drain mic audio captured while speaking
    state["mode"] = "idle"


# ---------------- behaviour ----------------
def open_window(fw, kind="read", secs=LISTEN_WINDOW_S):
    """Listen for a command for `secs`. kind = what we're waiting for (read | reply)."""
    recognizer.Reset()
    dictation.Reset()
    state["free_text"] = ""
    state["mode"] = "listening"
    state["window"] = kind
    state["deadline"] = time.time() + secs
    hint = "read message" if kind == "read" else "repeat / summarize" + (" / reply" if REPLIES_ENABLED else "")
    print(f"Listening for {hint} ({secs:.0f}s)...")


def idle_leds(fw):
    if pending:
        leds(fw, 0, 0, 40)     # dim blue: something still waiting
    else:
        leds(fw, 0, 0, 0)


def handle_line(fw, line):
    line = ANSI.sub("", line).strip()
    # rfind: the USB log occasionally glues a cut-off line onto the next one
    # ("...from SiddWILIMSG|..."), so parse from the last tag in the line.
    hits = [(line.rfind(tag), tag) for tag in ("WILIMSG|", "WILICONN|", "WILIDISC|")]
    i, _ = max(hits)
    if i < 0:
        return
    parts = line[i:].split("|", 3)

    if parts[0] == "WILICONN":
        print(f"iPhone connected: {parts[1]}")
        beep(fw, True)
    elif parts[0] == "WILIDISC":
        print(f"iPhone disconnected: {parts[1]}")
        beep(fw, False)
    elif parts[0] == "WILIMSG" and len(parts) == 4:
        _, app_id, sender, body = parts
        if app_id in CALL_APPS:
            handle_call(fw, sender, body)
            return
        # The iPhone often sends the same text twice with different titles ("Siddu Kodali",
        # then "Siddu"), so dedupe on app + body only, keeping the longer name for replies.
        key = (app_id, body)
        if key in recent and time.time() - recent[key] <= DEDUPE_S:
            for i, (a, s, b) in enumerate(pending):
                if (a, b) == (APP_NAMES.get(app_id, a), body) and len(sender) > len(s):
                    pending[i] = (a, sender, b)
            if state["reply_to"] and len(sender) > len(state["reply_to"]) and sender.startswith(state["reply_to"]):
                state["reply_to"] = sender
            remember(sender, None)           # just upgrade the stored display name
            return
        now = time.time()
        for k in [k for k, t in recent.items() if now - t > DEDUPE_S]:
            del recent[k]
        if key in recent:
            return
        recent[key] = now
        app = APP_NAMES.get(app_id, app_id.rsplit(".", 1)[-1])
        print(f"New message ({app}) from {sender}: {body}")
        remember(sender, body)
        latest.append((app, sender, body))
        pending.append((app, sender, body))
        leds(fw, 0, 0, 255)
        beep(fw)
        show(fw, f"{app}: {sender}")
        if AUTO_READ:
            read_all(fw)
        else:
            open_window(fw)


def read_all(fw):
    if not pending:                         # "tell me my messages": re-read the latest ones
        if not latest:
            speak(fw, "You don't have any messages yet.")
            return
        pending.extend(list(latest)[-RECENT_TO_READ:])
    leds(fw, 0, 255, 0)
    last_spoken.clear()
    for app, sender, body in list(pending):
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + ". Message continues on your phone."
        last_spoken.append(f"{app} from {sender}. {body}")
        speak(fw, last_spoken[-1])
        state["reply_to"] = sender
    pending.clear()
    leds(fw, 0, 0, 255)                     # blue: you can say "send message" to reply
    open_window(fw, "reply", REPLY_WINDOW_S)


def repeat(fw):
    if not last_spoken:
        speak(fw, "There's nothing to repeat yet.")
        return
    leds(fw, 0, 255, 0)
    for text in last_spoken:
        speak(fw, text)
    leds(fw, 0, 0, 255)
    open_window(fw, "reply", REPLY_WINDOW_S)


# ---------------- calls ----------------
def handle_call(fw, sender, body):
    """Phone/FaceTime notifications: only missed calls are announced; incoming/active calls
    are ignored (the desk can't carry call audio - BLE has no voice channel).
    The iPhone re-sends old missed-call notifications (on reconnect or when they update), so a
    missed call only counts if we saw that person's incoming call shortly before it."""
    who = sender.split(" ")[0].lower()
    now = time.time()
    if "incoming" in body.lower():
        ringing[who] = now
        return
    if "missed" not in body.lower():
        return
    if now - ringing.get(who, 0) > MISSED_CALL_WINDOW_S:
        print(f"(old missed-call notification from {sender} - not announced)")
        return
    key = ("missed call", who)
    if key in recent and now - recent[key] <= DEDUPE_S:
        return
    ringing.pop(who, None)                  # one announcement per ring; later re-sends are old
    recent[key] = now
    print(f"Missed call from {sender}")
    leds(fw, 255, 120, 0)                   # orange: missed call
    beep(fw)
    show(fw, f"Missed call: {sender}")
    speak(fw, f"You have a missed call from {sender}.")
    leds(fw, 0, 0, 0)


# ---------------- history + summaries ----------------
def remember(sender, body):
    key = sender.strip().split(" ")[0].lower()
    entry = history.setdefault(key, {"name": sender, "msgs": collections.deque(maxlen=HISTORY_MAX)})
    if len(sender) > len(entry["name"]):
        entry["name"] = sender
    if body is not None:
        entry["msgs"].append((time.time(), body))


def parse_summary_request(text):
    """'summarize my last three messages from siddu' -> (history key or None, n)."""
    words = text.lower().split()
    n = next((NUMBER_WORDS[w] for w in words if w in NUMBER_WORDS), None)
    n = next((int(w) for w in words if w.isdigit()), n) or SUMMARY_DEFAULT_N
    who = None
    tail = words[words.index("from") + 1:] if "from" in words else words
    for w in tail:   # fuzzy: the free recognizer spells names creatively ("sid do")
        hit = difflib.get_close_matches(w, list(history), n=1, cutoff=0.6)
        if hit:
            who = hit[0]
            break
    if who is None and len(tail) >= 2:
        joined = difflib.get_close_matches("".join(tail[:2]), list(history), n=1, cutoff=0.6)
        who = joined[0] if joined else None
    return who, max(1, min(n, HISTORY_MAX))


def summarize(fw, free_text):
    who, n = parse_summary_request(free_text)
    if who is None and latest:                          # default: whoever texted last
        who = latest[-1][1].split(" ")[0].lower()
    if who is None or who not in history or not history[who]["msgs"]:
        speak(fw, "I don't have any messages from them yet.")
        return
    name, msgs = history[who]["name"], list(history[who]["msgs"])[-n:]
    print(f"Summarizing last {len(msgs)} message(s) from {name}")
    leds(fw, 0, 255, 255)                   # cyan: thinking
    summary = gemini_summary(name, [b for _, b in msgs])
    if summary is None:                     # no API key / API error: read them out instead
        summary = f"Your last {len(msgs)} from {name}: " + ". ".join(b for _, b in msgs)
    speak(fw, summary)
    state["reply_to"] = name
    leds(fw, 0, 0, 255)
    open_window(fw, "reply", REPLY_WINDOW_S)


def gemini_summary(name, bodies):
    """Short spoken summary from Gemini, or None if unavailable (caller reads messages instead)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("  (no GEMINI_API_KEY in .env - reading messages instead of summarizing)")
        return None
    from google import genai
    from google.genai import types
    listing = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(bodies))
    client = genai.Client(api_key=key)   # keep a reference: an inline Client is closed mid-request
    for model in GEMINI_MODELS:          # next model on overload (503) or any other error
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"Summarize these {len(bodies)} most recent messages from {name}, oldest first:\n{listing}",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You summarize text messages for a desk assistant that reads your answer aloud "
                        "through a small speaker. Reply in one to three short, plain sentences: no lists, "
                        "markdown, emoji or quotes. Mention anything that needs an answer or action."
                    ),
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"  Gemini {model} failed: {str(e)[:120]}")
    return None   # caller reads the messages out instead


# ---------------- replies (Photon) ----------------
def load_contacts():
    try:
        data = json.loads(CONTACTS_FILE.read_text("utf-8"))
        return {k.strip().lower(): v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"contacts.json unreadable: {e}")
        return {}


def find_contact(sender):
    """Phone/email for a sender: exact name, else a unique contact sharing the first name."""
    contacts = load_contacts()
    name = (sender or "").strip().lower()
    if name in contacts:
        return contacts[name]
    first = name.split(" ")[0] if name else ""
    matches = [v for k, v in contacts.items() if first and k.split(" ")[0] == first]
    return matches[0] if len(matches) == 1 else None


def start_reply(fw):
    sender = state["reply_to"]
    if not sender:
        speak(fw, "There's no message to reply to.")
        return
    if not find_contact(sender):
        print(f'Add "{sender}" with their phone number or email to {CONTACTS_FILE.name}')
        speak(fw, f"I don't have a number for {sender}.")
        leds(fw, 0, 0, 0)
        return
    speak(fw, "What would you like to send?")
    beep(fw)
    leds(fw, 255, 0, 255)                   # purple: recording your reply
    dictation.Reset()
    state.update(mode="dictating", dictated=[], partial="", last_voice=time.time())
    state["deadline"] = time.time() + DICTATION_MAX_S
    print(f"Dictating reply to {sender} (stops after {DICTATION_SILENCE_S:.0f}s of silence)...")


def finish_dictation(fw):
    final = json.loads(dictation.FinalResult()).get("text", "")
    text = " ".join(state["dictated"] + ([final] if final else [])).strip()
    state["mode"] = "idle"
    if not text:
        speak(fw, "I didn't hear anything. Reply cancelled.")
        leds(fw, 0, 0, 0)
        return
    text = text[0].upper() + text[1:]
    sender = state["reply_to"]
    to = find_contact(sender)
    print(f"Sending to {sender} ({to}): {text}")
    try:
        r = requests.post(f"{PHOTON_BRIDGE}/send", json={"to": to, "text": text}, timeout=30)
        result = r.json()
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    if result.get("ok"):
        leds(fw, 0, 255, 0)
        speak(fw, f"Sent to {sender}.")
        state["reply_to"] = None
    else:
        print(f"  Photon send failed: {result.get('error')}")
        leds(fw, 255, 0, 0)
        speak(fw, "Sorry, the message didn't send.")
    leds(fw, 0, 0, 0)


def start_photon_bridge():
    """Start photon_bridge/server.mjs unless it's already running. Returns the process (or None)."""
    try:
        if requests.get(f"{PHOTON_BRIDGE}/health", timeout=1).ok:
            print("Photon bridge already running")
            return None
    except requests.RequestException:
        pass
    if not os.environ.get("PHOTON_PROJECT_ID"):
        print("Photon not configured (.env) - replies disabled")
        return None
    bridge = HERE / "photon_bridge"
    if not (bridge / "node_modules").exists():
        print("Photon bridge not installed - run `npm install` in photon_bridge/. Replies disabled")
        return None
    print("Starting Photon bridge...")
    return subprocess.Popen(["node", "server.mjs"], cwd=bridge)


def who(fw):
    if not pending:
        speak(fw, "No new messages.")
        return
    senders = list(dict.fromkeys(s for _, s, _ in pending))
    n = len(pending)
    speak(fw, f"{n} message{'s' if n > 1 else ''} from {' and '.join(senders)}.")
    open_window(fw)


def dismiss(fw):
    pending.clear()
    state["mode"] = "idle"
    leds(fw, 0, 0, 0)
    beep(fw, False)
    print("Dismissed.")


def handle_heard(fw, text):
    """Voice command from the listen window (see GRAMMAR)."""
    text = text.strip().lower()
    if not text or text == "[unk]":
        return
    print(f"Heard: '{text}'")
    if text.startswith(("read", "tell", "what are")):
        state["mode"] = "idle"
        read_all(fw)
    elif text.startswith("who"):
        who(fw)
    elif text in ("ignore", "dismiss", "not now"):
        dismiss(fw)
    elif text.startswith(("repeat", "say that", "can you repeat")):
        state["mode"] = "idle"
        repeat(fw)
    elif text.startswith("summarize"):
        state["mode"] = "idle"
        free = state["free_text"] or json.loads(dictation.FinalResult()).get("text", "")
        print(f"  (heard as: '{free}')")
        summarize(fw, free)
    elif REPLIES_ENABLED and text.startswith(("send", "reply", "respond")):
        state["mode"] = "idle"
        start_reply(fw)


def main():
    fw = FreeWili.find_first().expect("Failed to find FREE-WILi")
    global MESSAGE_SOURCE
    if MESSAGE_SOURCE == "auto":
        plugged = any(p.vid == 0x303A for p in serial.tools.list_ports.comports())
        MESSAGE_SOURCE = "usb" if plugged else "header"
    with fw:
        print(f"Connected to {fw}")
        fw.set_event_callback(on_event)
        for disable in (fw.enable_audio_events, fw.enable_button_events, fw.enable_uart_events):
            disable(False)  # clear streams a killed run may have left on (they block uploads/sound)
        upload_beeps(fw)
        fw.enable_audio_events(True).expect("Failed to enable audio events")
        fw.enable_button_events(True, 33).expect("Failed to enable button events")
        if MESSAGE_SOURCE == "usb":
            threading.Thread(target=usb_reader, daemon=True).start()
            print("Listening for messages on the Bottlenose USB cable")
        else:
            fw.enable_uart_events(True).expect("Failed to enable UART events")
            print("Listening for messages on the 20-pin header")
        leds(fw, 0, 0, 0)
        print(f"Voice: {'ElevenLabs' if ELEVENLABS_KEY else 'built-in (set ELEVENLABS_API_KEY for ElevenLabs)'}")
        bridge = None
        if REPLIES_ENABLED and "--speaker-test" not in sys.argv:
            bridge = start_photon_bridge()
            contacts = load_contacts()
            print(f"Replies: {len(contacts)} contact(s) in {CONTACTS_FILE.name}" if contacts
                  else f"Replies: no {CONTACTS_FILE.name} yet - add senders' numbers to reply")
        print(f"Summaries: {'Gemini' if os.environ.get('GEMINI_API_KEY') else 'no GEMINI_API_KEY - reads messages out instead'}")
        print("Ready. GREEN = talk, RED = dismiss. Ctrl+C to quit.")
        if "--test" in sys.argv:   # fake a message without needing a real text
            lines.put("WILIMSG|com.apple.MobileSMS|Test Sender|This is a test message from the desk.")
        try:
            if "--speaker-test" in sys.argv:   # check FREE-WILi file playback by ear
                print("Speaker test: listen to the FREE-WILi now")
                secs = synth("Messages from Ayan Siraj. Hey, are you coming to the demo?")
                with streams_paused(fw):
                    fw.send_file(TTS_FW, None, None)
                    steps = [
                        ("1) a beep", lambda: fw.play_audio_file("beephi")),
                        ("2) the number forty-two spoken", lambda: fw.play_audio_number_as_speech(42)),
                        ("3) our speech file", lambda: fw.play_audio_file(TTS_FW.stem)),
                    ]
                    for label, play in steps:
                        print(f"Playing {label} ... {play()}")
                        time.sleep(max(3, secs + 1) if "speech" in label else 3)
                return
            while True:
                fw.process_events()
                while not lines.empty():
                    handle_line(fw, lines.get())
                while not buttons.empty():
                    button = buttons.get()
                    if state["mode"] == "dictating":
                        if button == "green":        # done talking: send now
                            finish_dictation(fw)
                        else:                        # red: cancel the reply
                            state["mode"] = "idle"
                            leds(fw, 0, 0, 0)
                            speak(fw, "Reply cancelled.")
                    elif button == "green":
                        beep(fw)
                        open_window(fw, "reply" if state["reply_to"] and not pending else "read")
                    else:
                        dismiss(fw)
                while not heard.empty():
                    handle_heard(fw, heard.get())
                now = time.time()
                if state["mode"] == "listening" and now > state["deadline"]:
                    handle_heard(fw, json.loads(recognizer.FinalResult()).get("text", ""))
                    if state["mode"] == "listening":
                        state["mode"] = "idle"
                        print("Stopped listening.")
                        idle_leds(fw)
                if state["mode"] == "dictating" and (
                    now - state["last_voice"] > DICTATION_SILENCE_S or now > state["deadline"]
                ):
                    finish_dictation(fw)
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nBye!")
        finally:
            fw.enable_uart_events(False)
            fw.enable_audio_events(False)
            fw.enable_button_events(False)
            leds(fw, 0, 0, 0)
            if bridge:
                bridge.terminate()


if __name__ == "__main__":
    main()
