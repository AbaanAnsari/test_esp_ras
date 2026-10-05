#!/usr/bin/env python3

import json
import os
import socket
import struct
import threading
import time
import wave
from collections import deque

import numpy as np
from scipy import signal
from flask import Flask, jsonify, render_template, Response

try:
    import sounddevice as sd
except Exception:
    sd = None


BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "config.json"), "r") as f:
    CFG = json.load(f)

FS = int(CFG["sample_rate"])
CH = int(CFG["channels"])
FRAME = int(CFG["packet_frames"])

RADIUS = float(CFG["array_radius_m"])
MIC_ANGLES = np.deg2rad(np.asarray(CFG["mic_angles_deg"], dtype=float))
C = float(CFG["speed_of_sound_mps"])

NFFT = int(CFG["fft_size"])
HOP = int(CFG["hop_size"])

LOW_HZ = float(CFG["bandpass_low_hz"])
HIGH_HZ = float(CFG["bandpass_high_hz"])

MAX_LAG = float(CFG["gcc_max_lag_ms"]) * 1e-3

DOA_STEP = float(CFG["doa_grid_step_deg"])

GAIN = np.asarray(CFG["channel_gain"], dtype=np.float32)
CHANNEL_DELAY = np.asarray(
    CFG.get("channel_delay_samples", [0, 0, 0, 0]),
    dtype=float
)

UDP_PORT = int(CFG["udp_port"])

MAGIC = 0x53484134
HEADER = struct.Struct("<IIHBB")


# ============================================================
# Array geometry
# ============================================================

MIC_POS = np.column_stack(
    [
        RADIUS * np.cos(MIC_ANGLES),
        RADIUS * np.sin(MIC_ANGLES)
    ]
)


# ============================================================
# Global state
# ============================================================

state_lock = threading.Lock()

state = {
    "running": True,
    "packets": 0,
    "packet_loss": 0,
    "last_sequence": None,
    "doa_deg": 0.0,
    "doa_confidence": 0.0,
    "gcc_score": 0.0,
    "beam_rms": 0.0,
    "recording": False,
    "playing": False,
    "last_update": time.time(),
    "status": "waiting for ESP32"
}

audio_queue = deque(maxlen=50)

latest_channels = np.zeros((CH, FRAME), dtype=np.float32)
latest_beam = np.zeros(FRAME, dtype=np.float32)

record_lock = threading.Lock()
recording_frames = []


# ============================================================
# Streaming band-pass filters
# ============================================================

# 2nd-order Butterworth high-pass + low-pass.
# Filter state is maintained channel by channel.
sos_hp = signal.butter(
    2,
    LOW_HZ,
    btype="highpass",
    fs=FS,
    output="sos"
)

sos_lp = signal.butter(
    4,
    HIGH_HZ,
    btype="lowpass",
    fs=FS,
    output="sos"
)

zi_hp = np.zeros((CH, sos_hp.shape[0], 2), dtype=np.float64)
zi_lp = np.zeros((CH, sos_lp.shape[0], 2), dtype=np.float64)


def preprocess(x):
    """
    x: shape [4, N]
    returns filtered float32 [4, N]
    """
    global zi_hp, zi_lp

    x = x.astype(np.float64)

    # Remove DC using block mean.
    x -= np.mean(x, axis=1, keepdims=True)

    y = np.empty_like(x)

    for ch in range(CH):
        y[ch], zi_hp[ch] = signal.sosfilt(
            sos_hp, x[ch], zi=zi_hp[ch]
        )

        y[ch], zi_lp[ch] = signal.sosfilt(
            sos_lp, y[ch], zi=zi_lp[ch]
        )

    y *= GAIN[:, None]

    return y.astype(np.float32)


# ============================================================
# GCC-PHAT
# ============================================================

def gcc_phat(sig, refsig, fs, max_tau=None, interp=1):
    """
    Returns:
        tau: estimated delay in seconds
        peak: normalized PHAT peak magnitude
    """
    n = sig.size + refsig.size

    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), 1e-12)

    cc = np.fft.irfft(R, n=n)

    max_shift = n // 2

    if max_tau is not None:
        max_shift = min(
            int(interp * fs * max_tau),
            max_shift
        )

    cc = np.concatenate(
        (cc[-max_shift:], cc[:max_shift + 1])
    )

    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    peak = float(
        np.max(np.abs(cc)) /
        (np.mean(np.abs(cc)) + 1e-12)
    )

    return tau, peak


def pairwise_gcc(x):
    pairs = []
    peaks = []

    for i in range(CH):
        for j in range(i + 1, CH):
            tau, peak = gcc_phat(
                x[i],
                x[j],
                FS,
                MAX_LAG,
                interp=1
            )
            pairs.append((i, j, tau))
            peaks.append(peak)

    return pairs, float(np.mean(peaks))


# ============================================================
# DOA
# ============================================================

def direction_vector(theta):
    return np.array(
        [np.cos(theta), np.sin(theta)],
        dtype=np.float64
    )


def expected_tau(i, j, theta):
    """
    Far-field delay for direction theta.

    Positive theta is counter-clockwise from MIC0's +X axis.
    """
    u = direction_vector(theta)

    # propagation delay difference at microphones
    return float(
        np.dot(MIC_POS[j] - MIC_POS[i], u) / C
    )


def doa_from_gcc(pairs):
    """
    Scan 0..358 degrees and find the direction whose predicted
    pair delays best agree with GCC-PHAT measured delays.
    """
    if not pairs:
        return 0.0, 0.0

    thetas = np.deg2rad(
        np.arange(0.0, 360.0, DOA_STEP)
    )

    scores = np.zeros_like(thetas)

    for k, theta in enumerate(thetas):

        score = 0.0
        weight = 0.0

        for i, j, tau in pairs:

            pred = expected_tau(i, j, theta)

            # Gaussian-like delay consistency.
            sigma = 1.0 / FS

            err = tau - pred

            score += np.exp(
                -0.5 * (err / sigma) ** 2
            )

            weight += 1.0

        scores[k] = score / max(weight, 1.0)

    idx = int(np.argmax(scores))

    doa = float(np.rad2deg(thetas[idx]))
    confidence = float(scores[idx])

    return doa, confidence


# ============================================================
# Frequency-domain delay-and-sum beamformer
# ============================================================

def beamform(x, doa_deg):
    """
    Frequency-domain steering beamformer.

    Returns one mono block.
    """
    n = x.shape[1]

    # Zero-pad to NFFT.
    block = np.zeros((CH, NFFT), dtype=np.float64)
    block[:, :n] = x

    X = np.fft.rfft(block, n=NFFT, axis=1)

    freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)

    theta = np.deg2rad(doa_deg)
    u = direction_vector(theta)

    delays = (
        MIC_POS @ u
    ) / C

    delays = delays - np.mean(delays)

    delays += CHANNEL_DELAY / FS

    # Steering:
    # compensate propagation phase.
    steering = np.exp(
        1j * 2.0 * np.pi *
        freqs[None, :] *
        delays[:, None]
    )

    Y = np.sum(
        X * steering,
        axis=0
    ) / CH

    y = np.fft.irfft(
        Y,
        n=NFFT
    )

    # Extract original block.
    y = y[:n]

    # Prevent clipping.
    peak = np.max(np.abs(y)) + 1e-12

    if peak > 0.98:
        y = y * (0.98 / peak)

    return y.astype(np.float32)


# ============================================================
# Recording
# ============================================================

def save_recording():
    global recording_frames

    with record_lock:
        if not recording_frames:
            return

        data = np.concatenate(
            recording_frames
        ).astype(np.float32)

        recording_frames = []

    data16 = np.clip(
        data * 32767.0,
        -32768,
        32767
    ).astype(np.int16)

    path = os.path.join(
        BASE,
        CFG["record_file"]
    )

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(data16.tobytes())

    print("Saved:", path)


# ============================================================
# Playback
# ============================================================

play_thread = None
play_stop = threading.Event()


def playback_worker():
    global state

    path = os.path.join(
        BASE,
        CFG["record_file"]
    )

    if not os.path.exists(path):
        with state_lock:
            state["playing"] = False
        return

    if sd is None:
        with state_lock:
            state["playing"] = False
        return

    try:
        wf = wave.open(path, "rb")

        sr = wf.getframerate()

        def callback(outdata, frames, time_info, status):
            if play_stop.is_set():
                raise sd.CallbackStop()

            raw = wf.readframes(frames)

            if not raw:
                raise sd.CallbackStop()

            arr = np.frombuffer(
                raw,
                dtype=np.int16
            ).astype(np.float32) / 32768.0

            if len(arr) < frames:
                arr = np.pad(
                    arr,
                    (0, frames - len(arr))
                )

            outdata[:, 0] = arr[:frames]

        with sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            callback=callback,
            device=CFG.get("playback_device")
        ):
            while not play_stop.is_set():
                time.sleep(0.1)

        wf.close()

    except Exception as e:
        print("Playback error:", e)

    finally:
        with state_lock:
            state["playing"] = False


# ============================================================
# UDP receiver
# ============================================================

def receiver_loop():

    global latest_channels
    global latest_beam
    global recording_frames

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        4 * 1024 * 1024
    )

    sock.bind(
        (CFG["udp_host"], UDP_PORT)
    )

    print(
        f"UDP receiver listening on "
        f"{CFG['udp_host']}:{UDP_PORT}"
    )

    expected = None

    while True:

        try:
            data, addr = sock.recvfrom(4096)

            if len(data) < HEADER.size:
                continue

            magic, seq, frames, channels, version = \
                HEADER.unpack_from(data)

            if magic != MAGIC:
                continue

            if version != 1:
                continue

            if channels != 4:
                continue

            payload = data[HEADER.size:]

            expected_bytes = (
                frames *
                channels *
                2
            )

            if len(payload) != expected_bytes:
                continue

            if expected is not None and seq > expected:
                with state_lock:
                    state["packet_loss"] += seq - expected

            expected = seq + 1

            raw = np.frombuffer(
                payload,
                dtype="<i2"
            )

            x = raw.reshape(
                frames,
                channels
            ).T.astype(np.float32)

            x /= 32768.0

            y = preprocess(x)

            pairs, gcc_score = pairwise_gcc(y)

            doa, confidence = doa_from_gcc(pairs)

            beam = beamform(
                y,
                doa
            )

            with state_lock:

                state["packets"] += 1
                state["last_sequence"] = int(seq)
                state["doa_deg"] = round(doa, 1)
                state["doa_confidence"] = round(
                    confidence, 4
                )
                state["gcc_score"] = round(
                    gcc_score, 3
                )
                state["beam_rms"] = round(
                    float(np.sqrt(
                        np.mean(beam * beam)
                    )),
                    5
                )
                state["status"] = (
                    f"ESP32 {addr[0]}"
                )
                state["last_update"] = time.time()

            latest_channels = y
            latest_beam = beam

            # Keep latest block for optional API waveform.
            audio_queue.append(beam.copy())

            with state_lock:
                rec = state["recording"]

            if rec:
                with record_lock:
                    recording_frames.append(
                        beam.copy()
                    )

        except Exception as e:
            print("Receiver error:", e)
            time.sleep(0.1)


# ============================================================
# Flask dashboard
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, "templates")
)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():

    with state_lock:
        s = dict(state)

    # Convert geometry to dashboard-friendly coordinates.
    s["mic_positions"] = [
        {
            "x": float(p[0]),
            "y": float(p[1]),
            "angle": float(CFG["mic_angles_deg"][i])
        }
        for i, p in enumerate(MIC_POS)
    ]

    return jsonify(s)


@app.route("/api/waveform")
def api_waveform():

    if latest_beam.size == 0:
        arr = []
    else:
        # Downsample for dashboard.
        arr = latest_beam[::4].tolist()

    return jsonify({
        "samples": arr
    })


@app.route("/api/record/start", methods=["POST"])
def record_start():

    global recording_frames

    with record_lock:
        recording_frames = []

    with state_lock:
        state["recording"] = True

    return jsonify({"ok": True})


@app.route("/api/record/stop", methods=["POST"])
def record_stop():

    with state_lock:
        state["recording"] = False

    save_recording()

    return jsonify({"ok": True})


@app.route("/api/play/start", methods=["POST"])
def play_start():

    global play_thread

    if sd is None:
        return jsonify({
            "ok": False,
            "error": "sounddevice is not installed"
        })

    play_stop.clear()

    with state_lock:
        state["playing"] = True

    play_thread = threading.Thread(
        target=playback_worker,
        daemon=True
    )

    play_thread.start()

    return jsonify({"ok": True})


@app.route("/api/play/stop", methods=["POST"])
def play_stop_route():

    play_stop.set()

    with state_lock:
        state["playing"] = False

    return jsonify({"ok": True})


@app.route("/api/config")
def api_config():

    return jsonify({
        "sample_rate": FS,
        "channels": CH,
        "radius_m": RADIUS,
        "mic_angles_deg": CFG["mic_angles_deg"],
        "fft_size": NFFT,
        "hop_size": HOP,
        "low_hz": LOW_HZ,
        "high_hz": HIGH_HZ
    })


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    t = threading.Thread(
        target=receiver_loop,
        daemon=True
    )

    t.start()

    print()
    print("======================================")
    print(" Smart Hearing Aid Raspberry Pi 5")
    print("======================================")
    print(
        f"Dashboard: http://0.0.0.0:"
        f"{CFG['web_port']}"
    )
    print(
        f"DOA array radius: {RADIUS} m"
    )
    print(
        f"Mic angles: {CFG['mic_angles_deg']}"
    )
    print()

    app.run(
        host=CFG["web_host"],
        port=int(CFG["web_port"]),
        threaded=True
    )
