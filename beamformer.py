"""CHUNK 5: delay-and-sum with overlap-add (no block-edge clicks, correct steering sign)."""
import numpy as np
from settings import FS, CH, BLOCK, NFFT, MIC_POS, C, CAL_DELAY

_FREQS = np.fft.rfftfreq(NFFT, 1.0 / FS)
LATENCY = 4          # constant samples of delay so every channel delay is >= 0


class Beamformer:
    def __init__(self):
        self.carry = np.zeros(NFFT - BLOCK)

    def process(self, x, doa_deg):
        """x: [CH, BLOCK] -> mono [BLOCK]. Output is delayed by LATENCY samples."""
        th = np.deg2rad(doa_deg)
        u = np.array([np.cos(th), np.sin(th)])        # unit vector toward the source

        # A mic nearer the source hears it EARLIER, so it must be DELAYED by p.u/c.
        d = (MIC_POS @ u) / C * FS + CAL_DELAY + LATENCY      # samples, > 0

        X = np.fft.rfft(x.astype(np.float64), n=NFFT, axis=1)
        steer = np.exp(-2j * np.pi * _FREQS[None, :] * d[:, None] / FS)
        y = np.fft.irfft(np.sum(X * steer, axis=0) / CH, n=NFFT)

        out = y[:BLOCK].copy()
        out[:self.carry.size] += self.carry            # overlap-add previous tail
        self.carry = y[BLOCK:].copy()
        return np.clip(out, -0.98, 0.98).astype(np.float32)
