import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft
import ldpc_jossy as ldpc_jossy

# Parameters
FS = 48_000  # audio sample-rate (Hz)
FFT_LEN = 8192  # size of one OFDM symbol (must be even)
CP_LEN = FFT_LEN // 4  # cyclic-prefix length
CHIRP_LEN_S = 1 / 2  # chirp duration (seconds)
SILENCE_LEN_S = 1.0
F0, F1 = 3000, 0  # chirp start / end frequencies (Hz)
TX_REPS = 5  # 1 pilot + 7 identical data blocks (Comment seems outdated, it's TX_REPS pilot/data pairs)
WAV_TX = 'tx_sequence.wav'
WAV_RX = 'rx_recording.wav'
PILOT_NPY = 'pilot_symbols.npy'
DATA_NPY = 'data_symbols.npy'  # Will now store frequency-domain data symbols
COLMAP_NPY = 'colour_map.npy'
CHAN_NPY = 'channel_estimate.npy'
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"


CHIRP_ATTEN = 0.80  # scale applied to both chirps
TARGET_PEAK = 0.80  # peak of every OFDM block after scaling
#LENGTH_TOL = 10
LENGTH_TOL = 50


Q_COL = {(0, 0): '#d62728',  # red
         (0, 1): '#1f77b4',  # blue
         (1, 1): '#2ca02c',  # green
         (1, 0): '#ff7f0e'}  # orange


# LDPC params
LDPC_Z = 81
LDPC_N = 24 * LDPC_Z         # = 1944
LDPC_K = LDPC_N // 2         # = 972

my_ldpc = ldpc_jossy.code(standard='802.11n', rate='1/2', z=LDPC_Z)

# Sanity‐check: each OFDM symbol will carry TWO codewords, each of length=LDPC_N=1944 bits,
# so total QPSK symbols per OFDM = 2*(LDPC_N/2) = 2*972 = 1944.  We must have <= 4095 available subcarriers.
assert 2 * (LDPC_N // 2) <= (FFT_LEN // 2 - 1)


#-----------------------------------------------------------------------------------------#


#def generate_chirp(f0, f1, dur, fs=FS):
#    t = np.arange(int(dur * fs)) / fs
#    return (CHIRP_ATTEN * signal.chirp(t, f0, t[-1], f1)).astype(np.float32)

def generate_chirp(f0=F0, f1=F1, dur=CHIRP_LEN_S) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * FS), endpoint=False)
    k = (f1 - f0) / dur  # Sweep rate (Hz/s)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    signal = np.sin(phase)
    return (CHIRP_ATTEN * signal).astype(np.float32)


def random_bitpairs(n, seed_no=42):
    """ generates a choppable random sequence of bit pairs
        Aruments:
        n (int): number of bit pairs to generate, if not specified, defaults to maximum length of block emission
        seed_no (int): random seed for reproducibility"""
    np.random.seed(seed_no)
    return np.random.randint(0, 2, size=(n, 2), dtype=np.int8)


def qpsk_gray(bitpairs):
    mapping = {(0, 0): 1 + 1j, (0, 1): 1 - 1j, (1, 1): -1 - 1j, (1, 0): -1 + 1j}
    syms = np.array([mapping[tuple(b)] for b in bitpairs], np.complex64)
    colours = np.array([Q_COL[tuple(b)] for b in bitpairs])
    return syms, colours  # return colours to replicate later


def to_real_ofdm_block(useful_freq_symbols, n=FFT_LEN):
    """Convert a sequence of 'useful' frequency-domain symbols into
    a time-domain OFDM block with cyclic prefix (IFFT + real conversion).
    """
    # Number of occupied subcarriers
    N = len(useful_freq_symbols)

    # Prepare full FFT-bin vector
    X = np.zeros(n, np.complex64)

    # Map the 'useful' subcarriers into bins 1 through N
    X[1: 1 + N] = useful_freq_symbols

    # Mirror those into the negative frequencies for a real time-signal
    X[-N:] = np.conj(useful_freq_symbols[::-1])

    # IFFT and take real part
    x = fft.ifft(X).real.astype(np.float32)

    # Peak-normalise
    peak_val = np.max(np.abs(x))
    if peak_val > 1e-9:
        x *= TARGET_PEAK / peak_val
    elif TARGET_PEAK == 0:
        # Edge-case: if you really want silence
        x = np.zeros_like(x)

    return x


def add_cyclic_prefix(x: np.ndarray, cp_len: int = CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])


def prepare_tx_sequence(plot=False) -> dict:
    # ------------- build pieces -------------
    silence = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

    n_qpsk = FFT_LEN // 2 - 1

    # 0) Generate & store all info bits (systematic LDPC code)
    # We need one 2*K-bit message per OFDM *data* block, and there are TX_REPS * 4 of those
    rng = np.random.RandomState(seed=24)
    num_data_blocks = TX_REPS * 4
    # Each data-block carries TWO LDPC info halves, so 2*LDPC_K bits
    info_bits = rng.randint(0, 2, size=(num_data_blocks, 2 * LDPC_K), dtype=np.int8)

    # ------------- build data blocks (now TD without CP) and collect data freq symbols -------------
    data_blocks = []
    data_freq_symbols_ = []
    #pilot_colours = []
    #pilot_freq_symbols = []
    #time_pilot_blocks_no_CP = []

    # Loop over each repetition, encoding two K‐bit info blocks per OFDM symbol
    """for rep in range(TX_REPS):
    # 1) Extract the two info‐bit chunks (each length K=972)
           ib1 = info_bits[rep, : LDPC_K]
           ib2 = info_bits[rep, LDPC_K: 2 * LDPC_K]
    # 2) LDPC‐encode each chunk into an N=1944‐bit codeword
           cw1 = my_ldpc.encode(ib1)
           cw2 = my_ldpc.encode(ib2)

      # 3) Interleave bit‐pairs for QPSK mapping
           bits_for_qpsk = np.vstack([cw1, cw2]).reshape(-1, 2).astype(int)

      # 4) Map to complex QPSK symbols
           freq_syms, _ = qpsk_gray(bits_for_qpsk)
           data_freq_symbols_.append(freq_syms)

      # 5) Form the time‐domain OFDM block (IFFT + CP)
           block_no_cp = to_real_ofdm_block(freq_syms)
           data_blocks.append(block_no_cp) """

    # We have num_data_blocks = TX_REPS * 4 total blocks
    for blk_idx in range(num_data_blocks):
        # 1) Extract the two info‐bit chunks for this data‐block
        ib1 = info_bits[blk_idx, : LDPC_K]
        ib2 = info_bits[blk_idx, LDPC_K: 2 * LDPC_K]

        # 2) LDPC‐encode each chunk into an N=1944‐bit codeword
        cw1 = my_ldpc.encode(ib1)
        cw2 = my_ldpc.encode(ib2)

        # 3) Interleave bit‐pairs for QPSK mapping
        bits_for_qpsk = np.vstack([cw1, cw2]).reshape(-1, 2).astype(int)

        # 4) Map to complex QPSK symbols
        freq_syms, _ = qpsk_gray(bits_for_qpsk)
        data_freq_symbols_.append(freq_syms)

        # 5) Form the time‐domain OFDM block (IFFT + CP)
        data_blocks.append(to_real_ofdm_block(freq_syms))

    #np.save(COLMAP_NPY, np.array(pilot_colours, dtype=object))
    #np.save(PILOT_NPY, pilot_freq_symbols)
    #np.save(PILOT_TIME_NO_CP_NPY, time_pilot_blocks_no_CP)

    # ------------- build sequence --> 'payload' list will contain TD blocks WITHOUT CP -------------
    payload = []  # no CP
    payload_type = []
    time_pilot_blocks_no_cp = []
    pilot_freq_symbols = []

    for rep in range(TX_REPS):
        # 1) Regenerate a fresh pilot for this rep
        pilot_bits, _ = qpsk_gray(random_bitpairs(n_qpsk, seed_no=4242 + rep))
        pilot_td = to_real_ofdm_block(pilot_bits)
        payload.append(pilot_td)
        payload_type.append('pilot')

        time_pilot_blocks_no_cp.append(pilot_td)
        pilot_freq_symbols.append(pilot_bits)

        # 2) Append the 4 data blocks for this rep
        base = rep * 4
        for j in range(4):
            payload.append(data_blocks[base + j])
            payload_type.append('data')

    np.save(PILOT_TIME_NO_CP_NPY, np.array(time_pilot_blocks_no_cp, dtype=object))
    np.save(PILOT_NPY, np.array(pilot_freq_symbols, dtype=object))

    sequence = [
        silence,
        chirp_up,
        *payload,  # Unpacks TD blocks (pilots & data), all WITHOUT CP
        chirp_down,
        silence
    ]

    for i in range(2, len(sequence) - 2):
        sequence[i] = add_cyclic_prefix(sequence[i])
    sf.write(WAV_TX, np.concatenate(sequence), FS)  # Write concatenated sequence

    # ------------- plot function -------------
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(np.concatenate(sequence), lw=0.7)  # Plot concatenated sequence
        plt.title("Transmit waveform (time domain)")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.tight_layout()
        plt.show()

    # ------------- Correct calculation for total_ofdm_length -------------
    calculated_total_ofdm_length = len(payload_type) * (FFT_LEN + CP_LEN)

    for i, syms in enumerate(data_freq_symbols_):
        print(f" data block {i}: {len(syms)} symbols")

    # ------------- flat dictionary ---> key{waveform} is a flattened sequence, key{waveform_blocks} is unflattened -------------
    info = {
        "leading_silence_samples": len(silence),
        "chirp_samples": len(chirp_up),  # Length of chirp signal (no CP)
        "ofdm_block_len": FFT_LEN,
        "ofdm_block_len_with_cp": (len(sequence[2])),  # Length of first payload block (with CP)
        "cp_len": CP_LEN,
        # sanity‐check: is the very first payload block real?
        "block_real?": np.isrealobj(sequence[2]),  # sequence[0]=silence, [1]=chirp_up, [2]=first pilot+CP
        "total_ofdm_length": calculated_total_ofdm_length,
        "final_len": len(np.concatenate(sequence)),
        "no_of_payload_blocks": len(payload_type),
        "waveform_blocks": sequence,
        "payload_type_list": payload_type,
        #"payload_data_blocks": np.array(data_freq_symbols_),
        "payload_data_blocks": np.stack(data_freq_symbols_),

        "payload_info_bits" : info_bits

    }
    return {"waveform": np.concatenate(sequence), **info}

# This was the original line for testing, kept for consistency:
output = prepare_tx_sequence(True)

if __name__ == "__main__":

    pass

# def play_audio(sig:np.ndarray, fs:int=FS): # Definition was in original but commented out
# sd.play(sig, fs); sd.wait()


## SHOULD OUTPUT:
#colour_map.npy        ←  length=4095 array of ints (unchanged)
#data_symbols.npy      ←  shape = (TX_REPS, 1944)   # ← changed!
#pilot_symbols.npy     ←  shape = (4095,)          # unchanged
#tx_sequence.wav       ←  real-valued samples (unchanged format)