import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def window(sequence, window, alpha):
    if window:
        window = sg.tukey(len(sequence), alpha)
        return sequence * window
    else:
        return sequence

def chirp(f1, f2, T, fs, mode=0, window=False, alpha=0.04):
    N = int(fs * T)
    period = 1 / fs
    n = np.arange(N)

    if mode == 0:  # Linear
        k = (f2 - f1) / N
        freqs = f1 + k * n
        phase = 2 * np.pi * np.cumsum(freqs) / fs
        return window(np.sin(phase))

    elif mode == 1:  # Exponential
        if f1 <= 0 or f2 <= 0:
            raise ValueError("Exponential chirp requires positive start and end frequencies.")
        r = (f2 / f1)
        freqs = f1 * (r ** (n / N))
        phase = 2 * np.pi * np.cumsum(freqs) / fs
        return window(np.sin(phase))

    elif mode == 2:  # Hyperbolic
        denom = f1 + (f2 - f1) * n / N
        freqs = (f1 * f2) / denom
        phase = 2 * np.pi * np.cumsum(freqs) / fs
        return window(np.sin(phase))

    else:
        raise ValueError("Mode must be 0 (linear), 1 (exponential), or 2 (hyperbolic).")
    return