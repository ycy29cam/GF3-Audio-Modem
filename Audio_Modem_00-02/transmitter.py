import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft

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

Q_COL = { (0,0):'#d62728',  # red
          (0,1):'#1f77b4',  # blue
          (1,1):'#2ca02c',  # green
          (1,0):'#ff7f0e'}  # orange

def generate_chirp(f0:int, f1:int, dur_s:float, fs:int=FS) -> np.ndarray:
    """Linear up-chirp"""
    t = np.arange(int(dur_s*fs)) / fs
    return signal.chirp(t, f0, t[-1], f1).astype(np.float32)

def random_bitpairs(n_pairs:int) -> np.ndarray:
    return np.random.randint(0, 2, size=(n_pairs,2), dtype=np.int8)

def qpsk_gray(bitpairs:np.ndarray) -> np.ndarray:
    """
    Gray-coded QPSK anticlockwise starting bottom-left 00.
    Returns complex array, also saves colour map for later plotting.
    """
    mapping = { (0,0):-1-1j, (0,1):-1+1j, (1,1):1+1j, (1,0):1-1j }
    sym = np.array([mapping[tuple(b)] for b in bitpairs], dtype=np.complex64)
    colours = np.array([Q_COL[tuple(b)] for b in bitpairs])
    np.save(COLMAP_NPY, colours)
    return sym

def to_real_ofdm_block(freq_syms:np.ndarray, n:int=FFT_LEN) -> np.ndarray:
    """Places complex symbols on positive sub-carriers, mirrors for real IFFT."""
    block = np.zeros(n, dtype=np.complex64)
    # index 0 and N/2 left zero for DC & Nyquist
    half = n//2
    block[1:half] = freq_syms            # positive tones
    block[half+1:] = np.conj(freq_syms[::-1])  # mirror (neg. tones)
    time_dom = fft.ifft(block).real.astype(np.float32)  # guaranteed real  :contentReference[oaicite:0]{index=0}
    return time_dom

def add_cyclic_prefix(x:np.ndarray, cp_len:int=CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])

def prepare_tx_sequence() -> dict:
    # ------------- build pieces -------------
    silence     = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

    n_qpsk      = FFT_LEN//2 - 1
    bits        = random_bitpairs(n_qpsk)
    pilot       = qpsk_gray(bits)            # also stores colour map
    np.save(PILOT_NPY, pilot)

    blk_td      = to_real_ofdm_block(pilot)
    blk_td_cp   = add_cyclic_prefix(blk_td)  # CP only on first block

    sequence = np.concatenate([
        silence,
        chirp_up,
        blk_td_cp,
        np.tile(blk_td, TX_REPS-1),          # three CP-less copies
        chirp_down
    ])

    # write & visualise
    sf.write(WAV_TX, sequence, FS)
    plt.figure(figsize=(10,3))
    plt.plot(sequence, lw=.7)
    plt.title("Transmit waveform (time domain)")
    plt.xlabel("sample"); plt.ylabel("amplitude")
    plt.tight_layout(); plt.show()

    info = {
        "leading_silence_samples": len(silence),
        "chirp_samples"          : len(chirp_up),
        "ofdm_block_len"         : len(blk_td),
        "ofdm_block_len_prefix"  : len(blk_td_cp),
        "cp_len"                 : CP_LEN,
        "block_real?"            : np.isrealobj(blk_td),
        "total_ofdm_length"      : len(sequence) - len(chirp_up) - len(chirp_down) - len(silence),
        "final_len"              : len(sequence)
    }
    return dict(waveform=sequence, info=info)

def play_audio(sig:np.ndarray, fs:int=FS):
    sd.play(sig, fs); sd.wait()

output = prepare_tx_sequence()
play_audio(output["waveform"])
print(output["waveform"])
print(output["info"])