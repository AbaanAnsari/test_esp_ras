"""CHUNK 7: WAV recording of the beamformed output."""
import os, threading, wave
import numpy as np
from settings import BASE, CFG, FS

_lock = threading.Lock()
_frames = []

def clear():
    with _lock: _frames.clear()

def add(block):
    with _lock: _frames.append(block.copy())

def save():
    with _lock:
        if not _frames: return None
        data = np.concatenate(_frames); _frames.clear()
    pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
    path = os.path.join(BASE, CFG["record_file"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(FS)
        wf.writeframes(pcm.tobytes())
    print("Saved:", path)
    return path
