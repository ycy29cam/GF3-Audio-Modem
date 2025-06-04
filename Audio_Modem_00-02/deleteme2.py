import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
from scipy import signal, fft
from scipy.io.wavfile import read
from transmitter_00_03 import generate_chirp, WAV_TX, output, Q_COL        
import transmitter_00_03 as tx 

# ------------------ USER-DEFINED PARAMETERS ------------------

k = 1000        # Chirp rate (Hz/s)
f0 = 20         # Start frequency (Hz)
T = 0.5         # Half-duration (signal goes from -T to T)
N = 1000        # Number of samples per half
t = np.linspace(-T, T, 2 * N, endpoint=False)
dt = t[1] - t[0]  # Sample spacing

# ------------------ CHIRP GENERATION ------------------

# === Uncomment to define your own chirp ===
chirp = output["waveform"][1]
t = np.linspace(0, np.count(chirp) * dt, len(chirp), endpoint=False)

# Default linear chirp
# chirp = np.exp(1j * (2 * np.pi * f0 * t + np.pi * k * t**2))

# Automatically generate anti-chirp (conjugate chirp)
anti_chirp = np.conj(chirp)

# ------------------ CHANNEL MODEL ------------------

# === You can skip this if using a real received signal ===
channel = np.zeros_like(chirp)
delay_indices = [N, N + 100, N + 250]  # Channel taps at 0s, ~0.05s, ~0.125s
amplitudes = [1.0, 0.6, 0.3]
for idx, amp in zip(delay_indices, amplitudes):
    channel[idx] = amp

# Simulated received signal
received = scipy.signal.convolve(chirp, channel, mode='full')

# === Uncomment to use your own received signal ===
sample_rate, received  = read("tx_sequence.wav")

# ------------------ MATCHED FILTER ------------------

matched = scipy.signal.convolve(received, anti_chirp[::-1], mode='full')

# ------------------ TIME AXES ------------------

t_rx = np.linspace(-2*T, 2*T, len(received))
t_matched = np.linspace(-3*T, 3*T, len(matched))

# ------------------ CHANNEL PEAK MARKERS ------------------

# Compute true tap locations for simulated channel
matched_peak_locations = [(idx - len(chirp)) * dt for idx in delay_indices]
matched_abs = np.abs(matched) / np.max(np.abs(matched))
channel_plot = np.zeros_like(matched_abs)
for tau, amp in zip(matched_peak_locations, amplitudes):
    closest_idx = np.argmin(np.abs(t_matched - tau))
    channel_plot[closest_idx] = amp / max(amplitudes)  # Normalize

# ------------------ PLOTTING ------------------

plt.figure(figsize=(12, 5))
plt.plot(t_matched, matched_abs, label='Matched filter output', linewidth=2)
plt.plot(t_matched, channel_plot, '--o', label='Original channel taps', color='red', markersize=6)

plt.title("Matched Filter Output with Channel Overlay")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
