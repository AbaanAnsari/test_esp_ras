"""CHUNK 8: filter -> GCC-PHAT -> DOA -> beamform, in its own thread."""
import queue
import numpy as np
from settings import DOA_MIN_CONF
from state import state, state_lock, latest, block_q
from dsp_filter import Preprocessor
from beamformer import Beamformer
from doa import pairwise_gcc, doa_from_gcc
import recorder


def processor_loop():
    pre, bf = Preprocessor(), Beamformer()
    steer = 0.0
    while True:
        try:
            x = block_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            y = pre(x)
            taus, gcc_score = pairwise_gcc(y)
            doa, conf = doa_from_gcc(taus)
            if conf >= DOA_MIN_CONF:          # only re-steer when the estimate is trustworthy
                steer = doa
            beam = bf.process(y, steer)

            with state_lock:
                state["doa_deg"] = round(steer, 1)
                state["doa_confidence"] = round(conf, 4)
                state["gcc_score"] = round(gcc_score, 3)
                state["beam_rms"] = round(float(np.sqrt(np.mean(beam * beam))), 5)
                rec = state["recording"]
            latest["beam"] = beam
            if rec:
                recorder.add(beam)
        except Exception as e:
            print("Processor error:", e)
