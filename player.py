"""CHUNK 9: WAV playback through ALSA (optional)."""
import os, threading, time, wave
import numpy as np
from settings import BASE, CFG
from state import state, state_lock

try:
    import sounddevice as sd
except Exception:
    sd = None

_stop = threading.Event()

def available():
    return sd is not None

def _set_playing(v):
    with state_lock: state["playing"] = v

def _worker():
    path = os.path.join(BASE, CFG["record_file"])
    if sd is None or not os.path.exists(path):
        _set_playing(False); return
    try:
        wf = wave.open(path, "rb")
        def cb(out, frames, t, status):
            raw = wf.readframes(frames)
            if _stop.is_set() or not raw: raise sd.CallbackStop()
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            out[:, 0] = np.pad(a, (0, max(0, frames - a.size)))[:frames]
        with sd.OutputStream(samplerate=wf.getframerate(), channels=1, dtype="float32",
                             callback=cb, device=CFG.get("playback_device")):
            while not _stop.is_set(): time.sleep(0.1)
        wf.close()
    except Exception as e:
        print("Playback error:", e)
    finally:
        _set_playing(False)

def start():
    _stop.clear(); _set_playing(True)
    threading.Thread(target=_worker, daemon=True).start()

def stop():
    _stop.set(); _set_playing(False)
