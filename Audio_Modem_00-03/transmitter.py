import json, numpy as np, sounddevice as sd, soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft

# ------------------------------------------------------------
#                      PARAMETERS
# ------------------------------------------------------------
FS              = 48_000
FFT_LEN         = 8192
CP_LEN          = FFT_LEN // 4
CHIRP_LEN_S     = 2.0
SILENCE_LEN_S   = 1.0
F0,  F1         = 20, 15000          # chirp sweep (Hz)
TARGET_PEAK     = 0.80               # OFDM peak normalisation
CHIRP_ATTEN     = 0.80               # chirp amplitude scale

# files
WAV_TX          = "tx_sequence.wav"
PILOT_NPY       = "pilot_symbols.npy"
COLMAP_NPY      = "colour_map.npy"
DATA_SYMS_NPY   = "data_symbols.npy"
DATA_COLMAP_NPY = "data_colour_map.npy"

# colour palette (QPSK Gray)
Q_COL = {(0,0): "#d62728",   # red
         (0,1): "#1f77b4",   # blue
         (1,1): "#2ca02c",   # green
         (1,0): "#ff7f0e"}   # orange

# ------------------------------------------------------------
#                    BASIC UTILITIES
# ------------------------------------------------------------
def generate_chirp(f0, f1, dur, fs=FS):
    t = np.arange(int(dur*fs)) / fs
    return (CHIRP_ATTEN*signal.chirp(t, f0, t[-1], f1)).astype(np.float32)

def random_bitpairs(n):
    return np.random.randint(0, 2, size=(n, 2), dtype=np.int8)

def qpsk_gray(bitpairs, save_colour=True):
    mapping = {(0,0): -1-1j, (0,1): -1+1j,
               (1,1):  1+1j, (1,0):  1-1j}
    syms    = np.array([mapping[tuple(b)] for b in bitpairs], np.complex64)
    colours = np.array([Q_COL[tuple(b)]   for b in bitpairs])
    if save_colour:
        np.save(COLMAP_NPY, colours)
    return syms, colours

def to_real_ofdm_block(freq_syms, n=FFT_LEN):
    """Hermitian-symmetry → real TD block, peak-normalised."""
    half        = n // 2
    X           = np.zeros(n, np.complex64)
    X[1:half]   = freq_syms
    X[half+1:]  = np.conj(freq_syms[::-1])
    x           = fft.ifft(X).real.astype(np.float32)
    x          *= TARGET_PEAK / (np.max(np.abs(x)) + 1e-12)
    return x

def add_cyclic_prefix(x, cp_len=CP_LEN):
    return np.concatenate([x[-cp_len:], x])

# ------------------------------------------------------------
#               BUILD COMPLETE TX WAVEFORM
# ------------------------------------------------------------
def prepare_tx_sequence():
    silence    = np.zeros(int(SILENCE_LEN_S*FS), np.float32)
    chirp_up   = generate_chirp(F0,  F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1,  F0, CHIRP_LEN_S)

    n_qpsk = FFT_LEN//2 - 1

    # ---- Pilot constellation (seed 0) ----
    np.random.seed(0)
    bits_pilot        = random_bitpairs(n_qpsk)
    pilot_syms, _col  = qpsk_gray(bits_pilot, save_colour=True)   # colour map saved
    np.save(PILOT_NPY, pilot_syms)

    pilot_td          = to_real_ofdm_block(pilot_syms)
    pilot_td_cp       = add_cyclic_prefix(pilot_td)

    # ---- Data constellation (seed 1) ----
    np.random.seed(1)
    bits_data         = random_bitpairs(n_qpsk)
    data_syms, data_c = qpsk_gray(bits_data, save_colour=False)   # don’t overwrite pilot colours
    np.save(DATA_SYMS_NPY,   data_syms)
    np.save(DATA_COLMAP_NPY, data_c)

    data_td           = to_real_ofdm_block(data_syms)
    data_td_cp        = add_cyclic_prefix(data_td)

    # ---- Assemble final waveform ----
    # Block order: 0 pilot-CP | 1 pilot | 2 DATA-CP | 3 pilot-CP | 4 pilot
    sequence = np.concatenate([
        silence,
        chirp_up,
        pilot_td_cp,      # block 0  (CP)
        pilot_td,         # block 1  (no CP)
        data_td_cp,       # block 2  (CP)
        pilot_td_cp,      # block 3  (CP)
        pilot_td,         # block 4  (no CP)
        chirp_down
    ])

    sf.write(WAV_TX, sequence, FS)

    # quick sanity plot
    plt.figure(figsize=(9,3))
    plt.plot(sequence, lw=.7)
    plt.title("Transmit waveform"); plt.xlabel("sample"); plt.ylabel("amplitude")
    plt.tight_layout(); plt.show()

    meta = {
        "blocks_total"  : 5,
        "pilot_blocks"  : 4,
        "data_index"    : 2,
        "cp_blocks"     : [0, 2, 3],
        "waveform_len"  : len(sequence)
    }
    return {"waveform": sequence, "info": meta}

# ------------------------------------------------------------
#                 PLAY (for lab testing)
# ------------------------------------------------------------
def play_audio(sig, fs=FS):
    sd.play(sig, fs); sd.wait()

tx = prepare_tx_sequence()
play_audio(tx["waveform"])
print(json.dumps(tx["info"], indent=2))