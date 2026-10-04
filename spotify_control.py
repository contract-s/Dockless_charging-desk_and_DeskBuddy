"""
spotify_control.py  —  play / pause / skip / volume on your Spotify (Premium) from Python.

Setup (once):
    1. https://developer.spotify.com/dashboard -> Create app, tick "Web API",
       Redirect URI: http://127.0.0.1:8888/callback
    2. Put the Client ID / Secret in .env (see .env.example)
    3. Open the Spotify desktop app on this laptop (it is the speaker the desk plays to)
    4. python spotify_control.py status     -> a browser opens once to log in; token is cached

Test:
    python spotify_control.py play lofi
    python spotify_control.py play liked
    python spotify_control.py pause | resume | next | prev | volume 40 | status
"""

import sys
import threading

import config

SCOPES = "user-modify-playback-state user-read-playback-state user-read-currently-playing user-library-read"


class SpotifyError(RuntimeError):
    pass


class Spotify:
    def __init__(self):
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        if not config.SPOTIFY_CLIENT_ID:
            raise SpotifyError("SPOTIFY_CLIENT_ID is not set (copy .env.example to .env)")
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=SCOPES,
            open_browser=True,
            cache_path=".cache-spotify",
        ), requests_timeout=8)
        self.lock = threading.Lock()
        self.saved_volume = None

    # ---- devices ----
    def device_id(self):
        """The active device, else this laptop's Spotify app, else any device."""
        devices = self.sp.devices().get("devices", [])
        if not devices:
            raise SpotifyError("No Spotify device found. Open the Spotify app on the laptop.")
        for d in devices:
            if d.get("is_active"):
                return d["id"]
        for d in devices:
            if d.get("type") == "Computer":
                return d["id"]
        return devices[0]["id"]

    # ---- playback ----
    def play(self, query=None):
        """query: None = resume, "liked" = your Liked Songs, otherwise searched as playlist then track."""
        with self.lock:
            dev = self.device_id()
            if not query:
                self.sp.start_playback(device_id=dev)
                return "resuming"
            q = query.strip()
            if q.lower() in ("liked", "liked songs", "my music", "my songs", "favorites"):
                items = self.sp.current_user_saved_tracks(limit=10).get("items", [])
                uris = [it["track"]["uri"] for it in items if it.get("track")]
                if not uris:
                    raise SpotifyError("No liked songs found")
                self.sp.start_playback(device_id=dev, uris=uris)
                return "your liked songs"
            # Prefer a playlist for vibe requests ("lofi", "jazz"); fall back to a track.
            res = self.sp.search(q=q, type="playlist,track", limit=5)
            playlists = [p for p in (res.get("playlists") or {}).get("items", []) if p]
            tracks = [t for t in (res.get("tracks") or {}).get("items", []) if t]
            looks_like_song = " by " in q.lower()
            if tracks and (looks_like_song or not playlists):
                t = tracks[0]
                self.sp.start_playback(device_id=dev, uris=[t["uri"]])
                return f"{t['name']} by {t['artists'][0]['name']}"
            if playlists:
                p = playlists[0]
                self.sp.start_playback(device_id=dev, context_uri=p["uri"])
                return f"the playlist {p['name']}"
            raise SpotifyError(f"Nothing found for {q!r}")

    def pause(self):
        with self.lock:
            try:
                self.sp.pause_playback()
            except Exception as e:
                if "Restriction violated" not in str(e):   # already paused
                    raise

    def resume(self):
        return self.play(None)

    def toggle(self):
        if self.is_playing():
            self.pause()
            return "paused"
        return self.resume()

    def next(self):
        with self.lock:
            self.sp.next_track()

    def previous(self):
        with self.lock:
            self.sp.previous_track()

    def volume(self, percent):
        with self.lock:
            self.sp.volume(int(max(0, min(100, percent))))

    def current_volume(self):
        pb = self.sp.current_playback()
        return (pb or {}).get("device", {}).get("volume_percent")

    def duck(self, level=None):
        """Lower the music while the desk listens/talks; unduck() restores it."""
        try:
            if self.saved_volume is None and self.is_playing():
                self.saved_volume = self.current_volume()
                if self.saved_volume is not None:
                    self.volume(min(self.saved_volume, level if level is not None else config.DUCK_VOLUME))
        except Exception as e:
            print("[spotify] duck failed:", e)

    def unduck(self):
        try:
            if self.saved_volume is not None:
                self.volume(self.saved_volume)
        except Exception as e:
            print("[spotify] unduck failed:", e)
        self.saved_volume = None

    def is_playing(self):
        pb = self.sp.current_playback()
        return bool(pb and pb.get("is_playing"))

    def now_playing(self):
        """-> (is_playing, "Song by Artist") or (False, "")"""
        pb = self.sp.current_playback()
        if not pb or not pb.get("item"):
            return False, ""
        it = pb["item"]
        artist = it["artists"][0]["name"] if it.get("artists") else ""
        return bool(pb.get("is_playing")), f"{it['name']} by {artist}".strip()


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    s = Spotify()
    cmd, rest = sys.argv[1].lower(), " ".join(sys.argv[2:])
    if cmd == "play":
        print("playing", s.play(rest or None))
    elif cmd == "pause":
        s.pause()
    elif cmd == "resume":
        print(s.resume())
    elif cmd in ("next", "skip"):
        s.next()
    elif cmd in ("prev", "previous"):
        s.previous()
    elif cmd == "volume":
        s.volume(int(rest))
    elif cmd == "status":
        print("devices:", [(d["name"], d["type"], d["is_active"]) for d in s.sp.devices()["devices"]])
        print("now playing:", s.now_playing())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
