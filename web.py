"""CHUNK 10: Flask dashboard + JSON API (same endpoints as before)."""
import os
from flask import Flask, jsonify, render_template
from settings import BASE, CFG, FS, CH, RADIUS, NFFT, LOW_HZ, HIGH_HZ, MIC_POS
from state import state, state_lock, latest
import recorder, player

app = Flask(__name__, template_folder=os.path.join(BASE, "templates"))

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/state")
def api_state():
    with state_lock: s = dict(state)
    s["mic_positions"] = [{"x": float(p[0]), "y": float(p[1]), "angle": float(CFG["mic_angles_deg"][i])}
                          for i, p in enumerate(MIC_POS)]
    return jsonify(s)

@app.route("/api/waveform")
def api_waveform():
    return jsonify({"samples": latest["beam"][::4].tolist()})

@app.route("/api/record/start", methods=["POST"])
def record_start():
    recorder.clear()
    with state_lock: state["recording"] = True
    return jsonify({"ok": True})

@app.route("/api/record/stop", methods=["POST"])
def record_stop():
    with state_lock: state["recording"] = False
    recorder.save()
    return jsonify({"ok": True})

@app.route("/api/play/start", methods=["POST"])
def play_start():
    if not player.available():
        return jsonify({"ok": False, "error": "sounddevice is not installed"})
    player.start()
    return jsonify({"ok": True})

@app.route("/api/play/stop", methods=["POST"])
def play_stop():
    player.stop()
    return jsonify({"ok": True})

@app.route("/api/config")
def api_config():
    return jsonify({"sample_rate": FS, "channels": CH, "radius_m": RADIUS,
                    "mic_angles_deg": CFG["mic_angles_deg"], "fft_size": NFFT,
                    "low_hz": LOW_HZ, "high_hz": HIGH_HZ})
