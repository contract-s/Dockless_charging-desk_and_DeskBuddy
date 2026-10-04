"""
Desk Assistant - the FREE-WILi announces iPhone messages and reads them aloud on request.

iPhone --BLE/ANCS--> Bottlenose (desk_ancs.yaml firmware) --20-pin header--> FREE-WILi --USB--> this script

A new message turns the LEDs blue, beeps, shows the sender, and listens for ~8 s:
  "read my message" -> reads every pending message aloud
  "who is it from"  -> says who sent them
  "ignore" / "dismiss" / "not now" -> clears them
GREEN button = listen any time, RED button = dismiss.

Needs the FREE-WILi UART at 115200 8N1 with no flow control (its default; don't save the
"BottleNose" Orca setting, which switches it to 3 Mbps). Run inside .venv.
"""

import contextlib
import json
import os
import pathlib
import queue
import re
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
AUTO_READ = True            # True = read every new message aloud immediately; False = ask first
LISTEN_WINDOW_S = 8.0
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
    "who is it from", "who sent it",
    "ignore", "dismiss", "not now",
    "[unk]",
]
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
ANSI = re.compile(r"\x1b\[[0-9;]*m")

lines = queue.Queue()      # complete text lines from the Bottlenose
heard = queue.Queue()      # recognized phrases
buttons = queue.Queue()    # "green" / "red" presses
state = {
    "mode": "idle",        # idle | listening | speaking
    "deadline": 0.0,
    "dc": 0.0,
    "uart_buf": b"",
    "green": False,
    "red": False,
}
pending = []               # [(app, sender, body)]
recent = {}                # (app, sender, body) -> time first seen

recognizer = KaldiRecognizer(Model(str(MODEL_DIR)), ASR_RATE, json.dumps(GRAMMAR))
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
            if recognizer.AcceptWaveform(to_pcm16k(data.data)):
                heard.put(json.loads(recognizer.Result()).get("text", ""))

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
        time.sleep(0.25)


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
            if sent.is_ok() and fw.play_audio_file(TTS_FW.stem).is_ok():
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
def open_window(fw):
    recognizer.Reset()
    state["mode"] = "listening"
    state["deadline"] = time.time() + LISTEN_WINDOW_S
    print(f"Listening ({LISTEN_WINDOW_S:.0f}s)...")


def idle_leds(fw):
    if pending:
        leds(fw, 0, 0, 40)     # dim blue: something still waiting
    else:
        leds(fw, 0, 0, 0)


def handle_line(fw, line):
    line = ANSI.sub("", line).strip()
    for tag in ("WILIMSG|", "WILICONN|", "WILIDISC|"):
        i = line.find(tag)
        if i >= 0:
            break
    else:
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
        key = (app_id, sender, body)
        now = time.time()
        for k in [k for k, t in recent.items() if now - t > DEDUPE_S]:
            del recent[k]
        if key in recent:
            return
        recent[key] = now
        app = APP_NAMES.get(app_id, app_id.rsplit(".", 1)[-1])
        print(f"New message ({app}) from {sender}: {body}")
        pending.append((app, sender, body))
        leds(fw, 0, 0, 255)
        beep(fw)
        show(fw, f"{app}: {sender}")
        if AUTO_READ:
            read_all(fw)
        else:
            open_window(fw)


def read_all(fw):
    if not pending:
        speak(fw, "No new messages.")
        return
    leds(fw, 0, 255, 0)
    for app, sender, body in list(pending):
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS].rsplit(" ", 1)[0] + ". Message continues on your phone."
        speak(fw, f"{app} from {sender}. {body}")
    pending.clear()
    leds(fw, 0, 0, 0)


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
    text = text.strip().lower()
    if not text or text == "[unk]":
        return
    print(f"Heard: '{text}'")
    if text.startswith("read"):
        state["mode"] = "idle"
        read_all(fw)
    elif text.startswith("who"):
        who(fw)
    elif text in ("ignore", "dismiss", "not now"):
        dismiss(fw)


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
                    if buttons.get() == "green":
                        beep(fw)
                        open_window(fw)
                    else:
                        dismiss(fw)
                while not heard.empty():
                    handle_heard(fw, heard.get())
                if state["mode"] == "listening" and time.time() > state["deadline"]:
                    handle_heard(fw, json.loads(recognizer.FinalResult()).get("text", ""))
                    if state["mode"] == "listening":
                        state["mode"] = "idle"
                        print("Stopped listening.")
                        idle_leds(fw)
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nBye!")
        finally:
            fw.enable_uart_events(False)
            fw.enable_audio_events(False)
            fw.enable_button_events(False)
            leds(fw, 0, 0, 0)


if __name__ == "__main__":
    main()
