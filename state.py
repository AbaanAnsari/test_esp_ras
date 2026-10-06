"""CHUNK 2: shared state used by receiver, processor and web."""
import threading, time, queue
import numpy as np
from settings import BLOCK

state_lock = threading.Lock()
state = {
    "running": True, "packets": 0, "packet_loss": 0, "late_packets": 0,
    "dropped_blocks": 0, "last_sequence": None,
    "doa_deg": 0.0, "doa_confidence": 0.0, "gcc_score": 0.0, "beam_rms": 0.0,
    "recording": False, "playing": False,
    "last_update": time.time(), "status": "waiting for ESP32",
}
latest = {"beam": np.zeros(BLOCK, dtype=np.float32)}
block_q = queue.Queue(maxsize=50)   # receiver -> processor, shape (CH, BLOCK)

def bump(key, n=1):
    with state_lock:
        state[key] += n
