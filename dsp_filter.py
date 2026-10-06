"""CHUNK 3: streaming band-pass (state kept across blocks, no per-block mean steps)."""
import numpy as np
from scipy import signal
from settings import FS, CH, LOW_HZ, HIGH_HZ, GAIN

_sos_hp = signal.butter(2, LOW_HZ, btype="highpass", fs=FS, output="sos")
_sos_lp = signal.butter(4, HIGH_HZ, btype="lowpass", fs=FS, output="sos")


class Preprocessor:
    def __init__(self):
        self.zi_hp = np.zeros((_sos_hp.shape[0], CH, 2))
        self.zi_lp = np.zeros((_sos_lp.shape[0], CH, 2))

    def __call__(self, x):
        """x: [CH, N] float -> filtered float32 [CH, N]. The 150 Hz high-pass removes DC."""
        y, self.zi_hp = signal.sosfilt(_sos_hp, x.astype(np.float64), axis=1, zi=self.zi_hp)
        y, self.zi_lp = signal.sosfilt(_sos_lp, y, axis=1, zi=self.zi_lp)
        y *= GAIN[:, None]
        return y.astype(np.float32)
