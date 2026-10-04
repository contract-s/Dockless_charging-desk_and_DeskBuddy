"""
config.py  —  keys (from .env) and the smart desk's behaviour settings, in one place.

Copy .env.example to .env and fill in the keys. Everything else is tuned here.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ============================== KEYS ==============================
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_STT_MODEL = "scribe_v2"
ELEVENLABS_TTS_MODEL = "eleven_flash_v2_5"     # lowest latency

ASI1_API_KEY = os.getenv("ASI1_API_KEY", "")
ASI1_BASE_URL = "https://api.asi1.ai/v1"
ASI1_MODEL = "asi1"
LLM_TIMEOUT_S = 6.0          # after this, fall back to keyword matching

AGENT_SEED = os.getenv("AGENT_SEED", "")
AGENT_NAME = "smart-charging-desk"
AGENT_PORT = 8001

# ============================== BEHAVIOUR ==============================
# Demo scope: charging + music from the FREE-WILi. Flip these on for the extras.
ENABLE_VOICE = False         # ElevenLabs push-to-talk (or run desk_brain.py --voice)
ENABLE_KNOCK = False         # double-knock to talk (needs voice)
ENABLE_IR = False            # TV remote via the FREE-WILi IR receiver

# Spotify device to play on. "" = this laptop's Spotify app (never the phone on the desk).
# Otherwise part of the device name as Spotify shows it, e.g. "MacBook".
SPOTIFY_DEVICE_NAME = ""

COMMAND_SERVER = ("127.0.0.1", 8765)   # fetch_agent.py talks to the desk here
SPEAK_AGENT_REPLIES = True   # say ASI:One chat replies out loud too (fun in the demo)

LOST_AFTER_S = 3.0           # phone unseen this long = picked up
AUTO_MUSIC_ON_PLACE = True   # resume Spotify when the phone is set down
PAUSE_ON_PICKUP = True       # pause Spotify when the phone is picked up
GREET_EVERY_S = 120          # don't say "welcome back" more often than this

DEFAULT_PLAYLIST_QUERY = "lofi beats"
FOCUS_PLAYLIST_QUERY = "deep focus"
FOCUS_MINUTES = 25
DUCK_VOLUME = 15             # Spotify volume while the desk is listening / talking

LISTEN_MAX_S = 6.0           # longest voice command
LISTEN_SILENCE_S = 1.2       # stop recording after this much quiet (when not push-to-talk)
SAMPLE_RATE = 16000

# Energy story (Sustainability). A Qi/MagSafe pad left plugged in idles at roughly 0.3-1 W;
# while charging it pulls ~15 W from the wall, of which the phone gets ~11 W.
COIL_IDLE_W = 0.5
COIL_ACTIVE_W = 15.0
ENERGY_FILE = "energy.json"

# FREE-WILi panel. Button names as reported by freewili.read_all_buttons() (lower-case colour).
FREEWILI_BUTTONS = {
    "green": "previous",          # previous song   ("push_to_talk" if ENABLE_VOICE)
    "yellow": "toggle_music",     # play / pause
    "blue": "skip",               # next song
    "red": "toggle_follow",       # park the charger / follow the phone again
    "gray": "play_music",         # start DEFAULT_PLAYLIST_QUERY
    "white": "play_music",        # read_all_buttons() calls the 5th button White
}
KNOCK_G = 1.8                # accelerometer spike (in g) that counts as a knock
KNOCK_WINDOW_S = 0.6         # two knocks within this = start listening

# IR remote: run `python freewili_panel.py --learn-ir`, press buttons on any remote,
# and paste the printed codes here. Values are action names (see brain.ACTIONS).
IR_MAP = {
    # "0x20DF10EF": "toggle_music",
}
