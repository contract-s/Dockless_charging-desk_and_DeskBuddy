# charging-desk

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

A plexiglass desk with a wireless (Qi/MagSafe) charger underneath that moves to wherever you set your
phone down. You can talk to it, it plays your Spotify, and a FREE-WILi on the desk edge is its control panel.

An overhead camera finds the phone, a 2-axis gantry under the pane slides the charger beneath it, and
charging starts automatically.

## What it does
- **Follows your phone:** YOLOv8 + a dark-rectangle detector find the phone, even with the screen off.
  A homography maps pixels to desk cm, and the ESP32 gantry moves the charger there.
- **Knows when you sit down:** set the phone down and the charger powers on, your music resumes and the
  desk says "welcome back". Pick the phone up and the music pauses and the charger powers off.
- **Voice (ElevenLabs):** hold the FREE-WILi green button, knock twice on the desk, or press space.
  Ask things like "play some jazz", "skip", "volume 30", "where's my phone?", "park the charger",
  "focus for 25 minutes" or "how much energy did we save?". Speech-to-text uses ElevenLabs Scribe and
  replies use ElevenLabs Flash.
- **Understands requests (Fetch.ai ASI:One):** the `asi1` model turns what you say into desk actions.
  Keyword rules take over if it is unreachable.
- **Chat with your desk from anywhere (Fetch.ai agent):** `fetch_agent.py` is registered on Agentverse
  with the Agent Chat Protocol, so you can control the desk from ASI:One.
- **FREE-WILi control panel:** the buttons play/pause, skip, park and start focus mode. The LEDs show
  state (purple listening, blue moving, green charging, amber focus), the screen shows phone, song and
  focus timer, and any IR remote can be mapped to actions.
- **Sustainability:** the charger is only powered while a phone is on it (optional relay, `P` command).
  The desk tracks energy an always-on pad would have wasted.

## Fetch.ai agent
| Agent | Address |
|---|---|
| `smart-charging-desk` | `agent1q...` *(printed by `python fetch_agent.py` on startup: paste it here)* |

In ASI:One, ask: "Is my phone charging on my smart desk?", "Play lofi on my desk", or
"Start a 25 minute focus session on my desk".

## Files
| File | What |
|---|---|
| `desk_brain.py` | **Main program.** Runs the tracker, voice, FREE-WILi panel and command server together |
| `phone_tracker_screen_off.py` | Camera → phone position (YOLO + dark-rectangle hybrid) |
| `motion.py` | Desk cm → stepper steps → ESP32 (`charger_mover/charger_mover.ino`) |
| `brain.py` | Command → actions (ASI:One + keyword fallback). Every input goes through here |
| `voice.py` | ElevenLabs speech-to-text and voice replies |
| `spotify_control.py` | Spotify Web API (Premium) |
| `freewili_panel.py` | FREE-WILi buttons / knock / IR in, LEDs / screen out |
| `fetch_agent.py` | Fetch.ai uAgent (chat protocol, Agentverse mailbox) → desk |
| `desk_bus.py` | Shared desk state + events |
| `config.py` | All settings. Keys go in `.env` (see `.env.example`) |

## Setup
Needs Python 3.10+ (`brew install python@3.12`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in Spotify, ElevenLabs, ASI:One keys and an agent seed
```

1. **Camera calibration (once):** `python yolo_phone_tracker.py --calibrate`, then click the 4 corners of the pane.
2. **Gantry:** flash `charger_mover/charger_mover.ino`, then `python motion.py --zero --jog x 20` to check direction.
3. **Spotify:** open the Spotify app, then run `python spotify_control.py status` (logs in once in the browser).
4. **Voice:** `python voice.py --say "desk online"`.
5. **FREE-WILi:** `python freewili_panel.py --test`.

## Run
```bash
python desk_brain.py                 # push the charger to HOME first
python fetch_agent.py                # second terminal: the ASI:One agent
```
Testing without hardware: `python desk_brain.py --no-camera --dry-run`. Type commands, or
`!placed 30 20` / `!lost` / `!listen` to simulate the phone.

The camera is mounted ~80 cm above an 18 x 24 in pane. Change `DESK_WIDTH_CM` / `DESK_DEPTH_CM` in
`phone_tracker_cv.py` if your pane differs.
