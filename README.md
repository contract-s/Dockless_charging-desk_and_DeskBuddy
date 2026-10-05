# charging-desk and desk-buddy

This repo has two projects:

| Project | Folder | What it is |
|---|---|---|
| **DeskBuddy** | [`freewili/`](freewili) | A desk companion on a FREE-WiLi. It announces your iPhone messages and missed calls, then reads them aloud only when you ask. |
| **Dockless** | repo root | A prototype desk that tracks your phone with a camera and moves a charger under it on a two-axis stepper gantry. |

---

## DeskBuddy

### How it works

```
iPhone --BLE (ANCS)--> Bottlenose Orca (ESP32-C6, ESPHome) --UART--> FREE-WiLi --USB--> desk_assistant.py (laptop)
```

1. The Bottlenose Orca runs [`desk_ancs.yaml`](freewili/desk_ancs.yaml), an ESPHome config using [esphome-ancs](https://github.com/wonderslug/esphome-ancs). It pairs with the iPhone over Bluetooth and subscribes to its notifications (Apple ANCS).
2. For each notification it sends a line such as `WILIMSG|com.apple.MobileSMS|Sender|message text` over UART at 115200 baud (GPIO16 TX / GPIO17 RX on the 20-pin header).
3. `desk_assistant.py` reads those lines from the FREE-WiLi. It can also read them from the Bottlenose's own USB port: `MESSAGE_SOURCE = "auto"` picks USB when it sees an Espressif device (VID `0x303A`) and the header otherwise.
4. When a message arrives, DeskBuddy beeps, turns the LEDs blue and shows the sender on the FREE-WiLi screen. Then it listens for about 8 seconds.
5. Speech recognition is offline: [Vosk](https://alphacephei.com/vosk/) runs on the FREE-WiLi microphone audio, upsampled from 8 kHz to 16 kHz.
6. Replies are spoken with **ElevenLabs** TTS. The audio is band-limited (250 Hz–3.4 kHz), downsampled to an 8 kHz WAV and played on the FREE-WiLi speaker. `pyttsx3` is the offline fallback.
7. Summaries come from **Google Gemini**, which falls back through `gemini-flash-latest`, `gemini-2.5-flash` and `gemini-flash-lite-latest`.

### Voice commands

| Say | What happens |
|---|---|
| "read my message" | Reads the newest message |
| "repeat" / "say that again" | Reads it again |
| "tell me my messages" | Reads unread messages, or the latest few if none are unread |
| "summarize my last three messages from Siddu" | Gemini summary of that sender's last N messages (default 5, history keeps 10) |

**Buttons:** GREEN = listen at any time. RED = dismiss.

**Calls:** Only missed calls are announced ("You have a missed call from X"). Incoming calls are ignored, and a missed-call notification only counts if that person's call came in within the last 2 minutes, so old ones the iPhone re-sends aren't repeated.

### LED colors

| Color | Meaning |
|---|---|
| Blue | New message |
| Green | Listening / reading |
| Orange | Missed call |
| Cyan | Thinking (Gemini) |
| Dim blue | Something is still unread |

### Setup

**1. Flash the Bottlenose Orca** (once)

```bash
pip install esphome
cd freewili
esphome run desk_ancs.yaml
```

`components/ancs_bt/` is a small shim that keeps the ANCS component working on ESPHome 2026.9 and later. After flashing, pair the iPhone with the `superdesk` device in Bluetooth settings and allow notifications.

**2. Python environment**

```bash
cd freewili
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install freewili vosk numpy pyttsx3 requests pyserial google-genai
```

Download [`vosk-model-small-en-us-0.15`](https://alphacephei.com/vosk/models) and unzip it into `freewili/`.

**3. API keys**

```bash
cp .env.example .env
```

Fill in `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` (optional) and `GEMINI_API_KEY`. Without ElevenLabs it falls back to `pyttsx3`. Without Gemini, summaries are off.

**4. FREE-WiLi**

Set the FREE-WiLi UART to 115200 8N1 so it passes the Bottlenose's lines through. Then plug the FREE-WiLi into the laptop over USB. If you use the Bottlenose's USB port as the message source, plug that in too; both need data cables.

### Run

```bash
python desk_assistant.py                 # normal
python desk_assistant.py --test          # fake an incoming message, no phone needed
python desk_assistant.py --speaker-test  # check the FREE-WiLi speaker
```

### Other files in `freewili/`

- `wili_voice_dj.py`: a side experiment. Hold GREEN and say "play <song>", and the laptop plays a matching MP3 with pygame (set `MUSIC_DIR` to your music folder first).
- `photon_bridge/` and `contacts.example.json`: an optional Node bridge (spectrum-ts, port 8787) for replying to iMessages by voice. It is **off by default** (`REPLIES_ENABLED = False`) and was not used in the demo.

---

## Dockless (charging desk)

> **Status: prototype.** Phone tracking and the motor firmware each work on their own. The full camera-to-gantry loop is still being tuned.

### How it works

```
camera --> phone tracker (YOLOv8 + OpenCV) --> homography: pixels -> desk cm
       --> motion.py: cm -> motor steps --> USB serial --> ESP32 (charger_mover.ino) --> 2x NEMA17
```

- **Detection:** `phone_tracker_screen_off.py` is a hybrid tracker. It combines YOLOv8n ("cell phone") with an OpenCV dark-rectangle detector from `phone_tracker_cv.py`, so a face-down phone or one with the screen off is still found. `yolo_phone_tracker.py` is the YOLO-only version.
- **Calibration:** you click the 4 desk corners once. `cv2.findHomography` maps camera pixels to a 24 × 18 in (60.96 × 45.72 cm) desk surface, and the result is saved to `calibration.json`.
- **Motion:** `motion.py` converts desk cm into absolute step counts. The rate is 80 steps/mm: 200-step motors × 1/16 microstepping ÷ GT2 20-tooth pulleys (40 mm/rev). Travel is 434 × 242 mm. The result is sent to the ESP32.
- **Gantry:** crossed rods. The X motor (top track) moves the vertical rod, the Y motor (left track) moves the horizontal rod, and the charger sits where they cross.

### Hardware

- ESP32 Dev Module (ELEGOO ESP-WROOM-32)
- 2 × NEMA17 steppers with A4988 drivers at 1/16 microstepping
- GT2 belts and 20T pulleys
- Webcam mounted above the desk
- Wireless (MagSafe) charger

| Signal | ESP32 pin |
|---|---|
| X STEP / DIR | 26 / 27 |
| Y STEP / DIR | 25 / 33 |
| Driver ENABLE (shared, active LOW) | 13 |
| Limit switches (optional, off by default) | X 4, Y 16 |

### Firmware

Open `charger_mover/charger_mover.ino` in the **Arduino IDE**. Install **AccelStepper** (by Mike McCauley) from the Library Manager, select **ESP32 Dev Module** and upload.

Serial protocol (115200 baud, one command per line, replies `ok` or `error: ...`):

| Command | Action |
|---|---|
| `G <x_steps> <y_steps>` | Move to absolute step position |
| `Z` | Set the current position as 0,0 |
| `H` | Home with limit switches (if enabled) |
| `S` | Stop (decelerate) |
| `?` | Status: `pos <x> <y> moving <0/1>` |
| `E <0/1>` | Motors off / on |

### Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy pyserial

# test the gantry on its own
python motion.py --list-ports
python motion.py --zero              # charger is at home: call it 0,0
python motion.py --jog x 50          # move X by 50 mm
python motion.py --goto 30 20        # go to (30 cm, 20 cm) on the desk
python motion.py --square            # visit the 4 corners of the travel
python motion.py --dry-run --goto 30 20   # print steps without moving

# track the phone
python yolo_phone_tracker.py --calibrate  # click the 4 desk corners once
python phone_tracker_screen_off.py        # hybrid tracker
```

### Legacy files

These are from an earlier GRBL / FluidNC approach and are not used by the current code: `grbl_controller.py`, `charging_desk_fluidnc.yaml`, `hook_motion.py`, `patch_ptc.py`, `yolo_phone_tracker.before_motion.py`.

---

## Tech stack

**DeskBuddy:** Python, FREE-WiLi, Bottlenose Orca (ESP32-C6), ESPHome, Bluetooth LE / Apple ANCS, Vosk, ElevenLabs, Google Gemini

**Dockless:** Python, OpenCV, YOLOv8 (Ultralytics), C++ / Arduino IDE, ESP32, AccelStepper, NEMA17 + A4988

Built at MHacks 2026.
