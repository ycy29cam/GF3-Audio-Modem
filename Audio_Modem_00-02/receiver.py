import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.io.wavfile import read
from transmitter import *

# ------------------------------------------------
#   1.  General parameters (can be overridden by CLI flags)
# ------------------------------------------------
FS              = 48_000         # audio sample-rate (Hz)
FFT_LEN         = 8192           # size of one OFDM symbol (must be even)
CP_LEN          = FFT_LEN // 4   # cyclic-prefix length
CHIRP_LEN_S     = 2              # chirp duration (seconds)
SILENCE_LEN_S   = 1.0  
F0, F1          = 20, 15000      # chirp start / end frequencies (Hz)
TX_REPS         = 4              # 1 pilot + 3 identical data blocks
WAV_TX          = 'tx_sequence.wav'
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

def record_audio(expected_len:int, fs:int=FS) -> np.ndarray:
    print(f"Recording ≈{expected_len/fs:.2f} s …")
    rec = sd.rec(expected_len, samplerate=fs, channels=1,
                 dtype='float32').squeeze()
    sd.wait()
    sf.write(WAV_RX, rec, fs)
    return rec

def synchronise(rx:np.ndarray,
                chirp_up:np.ndarray,
                chirp_down:np.ndarray) -> tuple[np.ndarray,int,int]:
    """Find payload boundaries using both chirps (silence is ignored)."""
    corr_up   = signal.correlate(rx, chirp_up,   mode='valid')
    peak_up   = np.argmax(corr_up)                  # strongest match

    corr_down = signal.correlate(rx, chirp_down, mode='valid')
    search_from = peak_up + len(chirp_up)
    peak_down_candidates = np.where(
        corr_down > 0.8*corr_down.max())[0]
    peak_down = peak_down_candidates[
        peak_down_candidates > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down                    # start of down-chirp
    payload       = rx[start_payload:end_payload]
    return payload, start_payload, end_payload

def split_ofdm_blocks(payload:np.ndarray,
                      fft_len:int=FFT_LEN,
                      cp_len:int=CP_LEN,
                      reps:int=TX_REPS) -> np.ndarray:
    """Layout: CP+FFT_LEN + (reps-1)*FFT_LEN."""
    expected = cp_len + reps*fft_len
    if len(payload) < expected:
        raise RuntimeError(f"Payload too short ({len(payload)} < {expected})")

    blocks = []
    idx = cp_len
    for _ in range(reps):
        blocks.append(payload[idx : idx+fft_len])
        idx += fft_len
    return np.stack(blocks)

def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    """FFT and keep positive sub-carriers (1 … N/2-1)."""
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2]

def channel_estimate(rx_fd:np.ndarray, pilot:np.ndarray) -> np.ndarray:
    eps = 1e-12
    H_hat = rx_fd[0] / (pilot + eps)
    np.save(CHAN_NPY, H_hat)
    return H_hat

def equalise(rx_fd:np.ndarray, H:np.ndarray) -> np.ndarray:
    return rx_fd / H

def spectrum_plot(sig:np.ndarray, fs:int=FS):
    f, Pxx = signal.welch(sig, fs, nperseg=4096)
    plt.figure(); plt.semilogy(f, Pxx)
    plt.title("Received PSD"); plt.xlabel("Hz"); plt.ylabel("PSD [V²/Hz]")
    plt.tight_layout(); plt.show()

def constellation_plot(eq_fd:np.ndarray):
    col = np.load(COLMAP_NPY)
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(eq_fd.real, eq_fd.imag, c=col,
                s=10, alpha=.85, edgecolors='none')
    plt.title("Equalised constellation"); plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

# record_audio(480000)

SAMPLE_RATE, recording = read('tx_sequence.wav')

chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

print(len(recording))
sync = synchronise(recording, chirp_up, chirp_down)
print(sync)

split_block = split_ofdm_blocks(sync[0])
freq_block = freq_domain(split_block)
channel = channel_estimate(freq_block, np.load("pilot_symbols.npy"))
eq_block = equalise(freq_block, channel)
spectrum_plot(recording)
constellation_plot(eq_block)