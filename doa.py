"""CHUNK 4: GCC-PHAT pair delays and DOA scan (vectorised)."""
import numpy as np
from settings import FS, CH, MAX_LAG, DOA_STEP, MIC_POS, C

PAIRS = [(i, j) for i in range(CH) for j in range(i + 1, CH)]
_THETAS = np.deg2rad(np.arange(0.0, 360.0, DOA_STEP))
_U = np.vstack([np.cos(_THETAS), np.sin(_THETAS)])                 # [2, T]
_D = np.array([MIC_POS[j] - MIC_POS[i] for i, j in PAIRS])         # [P, 2]
_PRED = (_D @ _U) / C                                              # [P, T] seconds
_SIGMA = 1.0 / FS


def gcc_phat(sig, ref):
    """Delay (s) of `sig` relative to `ref`, and a peak-sharpness score."""
    n = sig.size + ref.size
    R = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, n)

    m = min(int(FS * MAX_LAG), n // 2 - 1)
    cc = np.concatenate((cc[-m:], cc[:m + 1]))
    k = int(np.argmax(cc))
    shift = float(k - m)
    if 0 < k < cc.size - 1:                       # parabolic sub-sample refinement
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            shift += 0.5 * (a - c) / den
    return shift / FS, float(cc[k] / (np.mean(np.abs(cc)) + 1e-12))


def pairwise_gcc(x):
    taus, peaks = [], []
    for i, j in PAIRS:
        t, p = gcc_phat(x[i], x[j])
        taus.append(t); peaks.append(p)
    return np.asarray(taus), float(np.mean(peaks))


def doa_from_gcc(taus):
    """Return (azimuth deg, confidence 0..1)."""
    err = taus[:, None] - _PRED
    scores = np.mean(np.exp(-0.5 * (err / _SIGMA) ** 2), axis=0)
    idx = int(np.argmax(scores))
    return float(np.rad2deg(_THETAS[idx])), float(scores[idx])
