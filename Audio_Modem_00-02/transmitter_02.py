"""
Transmitter -- up-chirp + 4 pilot OFDM blocks + 1 data OFDM block + down-chirp
Every OFDM block, the data block *and* the down-chirp are preceded by a cyclic
prefix.  Only the up-chirp has no prefix.

Block order (after leading silence):

      ┌──── prefix ────┐
  0 -►│ CP │  pilot #1 │
  1 -►│ CP │  pilot #2 │
  2 -►│ CP │   DATA    │   (random seed 1, colour map saved separately)
  3 -►│ CP │  pilot #3 │
  4 -►│ CP │  pilot #4 │
  5 -►│ CP │ down-chirp│
"""

# ------------------------------------------------------------
# 0.  Imports & global parameters
# ------------------------------------------------------------
import numpy as np
import sounddevice as sd, soundfile as sf, matplotlib.pyplot as plt
from scipy import signal, fft

FS              = 48_000
FFT_LEN         = 8192
CP_LEN          = FFT_LEN // 4
CHIRP_LEN_S     = 2.0
SILENCE_LEN_S   = 1.0
F0,  F1         = 20, 15000      # up-chirp sweep
TARGET_PEAK     = 0.80
CHIRP_ATTEN     = 0.80

# -------- I/O file names --------
WAV_TX          = "tx_sequence.wav"
PILOT_NPY       = "pilot_symbols.npy"
COLMAP_NPY      = "colour_map.npy"
DATA_SYMS_NPY   = "data_symbols.npy"
DATA_COLMAP_NPY = "data_colour_map.npy"

# -------- QPSK Gray mapping & palette --------
Q_COL = {(0,0): "#d62728",  # red
         (0,1): "#1f77b4",  # blue
         (1,1): "#2ca02c",  # green
         (1,0): "#ff7f0e"}  # orange


# ------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------
def generate_chirp(f0, f1, dur, fs=FS):
    t = np.arange(int(dur*fs)) / fs
    return (CHIRP_ATTEN * signal.chirp(t, f0, t[-1], f1)).astype(np.float32)


def random_bitpairs(n):                        # n×2 array of {0,1}
    return np.random.randint(0, 2, size=(n, 2), dtype=np.int8)


def qpsk_gray(bitpairs, save_colour_path=None):
    """Return QPSK-Gray symbols & their colours; optionally save the colours."""
    mapping = {(0,0): -1-1j, (0,1): -1+1j,
               (1,1):  1+1j, (1,0):  1-1j}
    syms    = np.array([mapping[tuple(b)] for b in bitpairs], np.complex64)
    colours = np.array([Q_COL[tuple(b)]   for b in bitpairs])
    if save_colour_path:
        np.save(save_colour_path, colours)
    return syms, colours


def to_real_ofdm_block(freq_syms):
    """Hermitian mirror → real-valued TD block, then peak-normalise."""
    n      = FFT_LEN
    half   = n // 2
    X      = np.zeros(n, np.complex64)
    X[1:half]   = freq_syms
    X[half+1:]  = np.conj(freq_syms[::-1])
    x_td   = fft.ifft(X).real.astype(np.float32)
    x_td  *= TARGET_PEAK / (np.max(np.abs(x_td)) + 1e-12)
    return x_td


def add_cyclic_prefix(x_td, cp_len=CP_LEN):
    return np.concatenate([x_td[-cp_len:], x_td])


# ------------------------------------------------------------
# 2.  Build the complete transmit waveform
# ------------------------------------------------------------
def prepare_tx_sequence():
    silence     = np.zeros(int(SILENCE_LEN_S*FS), np.float32)
    chirp_up    = generate_chirp(F0,  F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1,  F0, CHIRP_LEN_S)

    n_qpsk = FFT_LEN//2 - 1

    # ---- pilot constellation (seed 0, saved) ----
    np.random.seed(0)
    bits_pilot, _ = random_bitpairs(n_qpsk), None
    pilot_syms, pilot_col = qpsk_gray(bits_pilot, save_colour_path=COLMAP_NPY)
    np.save(PILOT_NPY, pilot_syms)

    pilot_td     = to_real_ofdm_block(pilot_syms)
    pilot_cp_blk = add_cyclic_prefix(pilot_td)         # re-use for all pilots

    # ---- data constellation (seed 1, saved separately) ----
    np.random.seed(1)
    bits_data = random_bitpairs(n_qpsk)
    data_syms, data_col = qpsk_gray(bits_data)          # don’t overwrite pilot col
    np.save(DATA_SYMS_NPY,   data_syms)
    np.save(DATA_COLMAP_NPY, data_col)

    data_cp_blk = add_cyclic_prefix(to_real_ofdm_block(data_syms))

    # ---- CP on down-chirp ---------------------------------
    chirp_down_cp = np.concatenate([chirp_down[-CP_LEN:], chirp_down])

    # ---- final sequence -----------------------------------
    sequence = np.concatenate([
        silence,
        chirp_up,
        pilot_cp_blk,              # pilot #1
        pilot_cp_blk,              # pilot #2
        data_cp_blk,               # DATA
        pilot_cp_blk,              # pilot #3
        pilot_cp_blk,              # pilot #4
        chirp_down_cp
    ])

    sf.write(WAV_TX, sequence, FS)

    # quick visual check
    plt.figure(figsize=(10,3))
    plt.plot(sequence, lw=.7)
    plt.title("Transmit waveform"); plt.xlabel("sample"); plt.ylabel("amplitude")
    plt.tight_layout(); plt.show()

    return {"waveform": sequence,
            "info": {"blocks": 5, "waveform_len": len(sequence)}}


# ------------------------------------------------------------
# 3.  Playback (comment out if not needed in lab)
# ------------------------------------------------------------
tx = prepare_tx_sequence()
sd.play(tx["waveform"], FS); sd.wait()
print("\nTx meta-info:", tx["info"])
