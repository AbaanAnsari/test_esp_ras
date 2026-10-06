"""CHUNK 1: configuration and array geometry."""
import json, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "config.json")) as f:
    CFG = json.load(f)

FS = int(CFG["sample_rate"])
CH = int(CFG["channels"])
PACKET_FRAMES = int(CFG["packet_frames"])      # frames per UDP packet (160 = 10 ms)
BLOCK = int(CFG.get("block_frames", 320))      # DSP block (320 = 20 ms)
NFFT = int(CFG["fft_size"])
RADIUS = float(CFG["array_radius_m"])
C = float(CFG["speed_of_sound_mps"])
LOW_HZ = float(CFG["bandpass_low_hz"])
HIGH_HZ = float(CFG["bandpass_high_hz"])
MAX_LAG = float(CFG["gcc_max_lag_ms"]) * 1e-3
DOA_STEP = float(CFG["doa_grid_step_deg"])
DOA_MIN_CONF = float(CFG.get("doa_min_confidence", 0.25))
GAIN = np.asarray(CFG["channel_gain"], dtype=np.float32)
CAL_DELAY = np.asarray(CFG.get("channel_delay_samples", [0] * CH), dtype=float)
UDP_HOST = CFG["udp_host"]
UDP_PORT = int(CFG["udp_port"])
RCVBUF = int(CFG.get("rcvbuf_bytes", 1 << 20))

MAGIC = 0x53484134
PACKET_VERSION = 1

MIC_ANGLES = np.deg2rad(np.asarray(CFG["mic_angles_deg"], dtype=float))
MIC_POS = np.column_stack([RADIUS * np.cos(MIC_ANGLES), RADIUS * np.sin(MIC_ANGLES)])

assert NFFT >= BLOCK + 8, "fft_size must exceed block_frames by a few samples"
assert BLOCK % PACKET_FRAMES == 0, "block_frames must be a multiple of packet_frames"
