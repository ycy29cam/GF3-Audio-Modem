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
DATA_NPY = 'data_symbols.npy'  # Will now store frequency-domain data symbols#
COLMAP_NPY = 'colour_map.npy'
CHAN_NPY = 'channel_estimate.npy'

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


def generate_chirp(f0, f1, dur, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return (CHIRP_ATTEN * signal.chirp(t, f0, t[-1], f1)).astype(np.float32)


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
    """ converts a sequence of useful frequency symbols to a real OFDM block
    """
    half = n // 2
    X = np.zeros(n, np.complex64)
    X[1:half] = useful_freq_symbols
    X[half + 1:] = np.conj(useful_freq_symbols[::-1])
    x = fft.ifft(X).real.astype(np.float32)
    # peak normalisation
    peak_val = np.max(np.abs(x))
    if peak_val > 1e-9:  # Avoid division by zero for unstable scaling
        x *= TARGET_PEAK / peak_val
    elif TARGET_PEAK == 0:
        x = np.zeros_like(x)
    # If peak_val is very small and TARGET_PEAK is not zero, x remains small
    return x


def add_cyclic_prefix(x: np.ndarray, cp_len: int = CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])


def prepare_tx_sequence(plot=False) -> dict:
    # ------------- build pieces -------------
    silence = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

    n_qpsk = FFT_LEN // 2 - 1

    # Build TWO 1944‐bit LDPC codewords per OFDM symbol, map each codeword → 972 QPSK symbols, concatenate those
    # 972+972 = 1944 symbols, and then scatter them into the 4095 subcarriers

    # First, compute how many QPSK symbols we need per OFDM symbol:
    n_data_subc = FFT_LEN // 2 - 1                 # = 4095, total QPSK slots per OFDM symbol
    symbols_per_ofdm = 2 * (LDPC_N // 2)           # = 2*(1944/2) = 1944 QPSK symbols
    assert symbols_per_ofdm == 1944
    assert symbols_per_ofdm <= n_data_subc        # ensures we actually have enough subcarriers

    # (2‐a) Generate TOTAL_INFO_BITS = TX_REPS * (2 * LDPC_K):
    # Because each OFDM symbol carries 2 codewords, each of length K=972 info bits,
    # so per OFDM symbol we need 2*972 = 1944 info bits.
    total_info_bits = TX_REPS * 2 * LDPC_K        # = TX_REPS * (2*972)
    info_bits = np.random.RandomState(24).randint(0, 2, size=(total_info_bits,), dtype=np.int8)

    data_blocks = []  # This list will now store TIME DOMAIN data blocks WITHOUT CP
    _data_freq_symbols_for_info_dict = []  # To store data QPSK symbols for the output dictionary and .npy file

    for i in range(TX_REPS):
        # SLICE OUT the next (2*LDPC_K)=1944 info bits for this OFDM symbol:
        start_info = i * 2 * LDPC_K
        end_info = (i + 1) * 2 * LDPC_K
        this_two_info = info_bits[start_info:end_info]  # shape = (1944,)

        # SPLIT into two length‐K chunks of 972 bits:
        first_info_block = this_two_info[0:LDPC_K]  # bits 0..971
        second_info_block = this_two_info[LDPC_K: 2 * LDPC_K]  # bits 972..1943

        # LDPC‐ENCODE each 972‐bit chunk → 1944‐bit codeword
        cw1 = my_ldpc.encode(first_info_block)  # shape = (1944,)
        cw2 = my_ldpc.encode(second_info_block)  # shape = (1944,)

        # Reshape each codeword into QPSK‐bit‐pairs: (1944 bits → 972 pairs)
        cw1_pairs = np.asarray(cw1, dtype=np.int8).reshape((LDPC_N // 2, 2))
        cw2_pairs = np.asarray(cw2, dtype=np.int8).reshape((LDPC_N // 2, 2))

        # CALL qpsk_gray(...) on each (→ returns 972 symbols + their colours)
        syms1, colours1 = qpsk_gray(cw1_pairs)  # shape of syms1 = (972,) complex
        syms2, colours2 = qpsk_gray(cw2_pairs)  # shape of syms2 = (972,) complex

        # CONCATENATE syms1 || syms2 → gives 1944 QPSK symbols. We'll scatter these
        # into the first 1944 data‐subcarriers.  For any remaining subcarriers (positions 1944..4094),
        # we insert zeros (no data).  That way each OFDM symbol remains length=4095 after pilots,
        # but only the first 1944 are “active” data, the rest = 0.
        payload_syms = np.concatenate([syms1, syms2])  # shape = (1944,)

        # (Optionally) concatenate their colours so that Rx can colour‐code the scatter plot consistently:
        payload_colours = np.concatenate([colours1, colours2])  # shape = (1944,)

        # ZERO‐PAD to length n_data_subc = 4095, so that the final array “data_subcarrier_syms”
        # is length 4095:
        zeros_to_pad = n_data_subc - symbols_per_ofdm  # = 4095 - 1944 = 2151
        if zeros_to_pad < 0:
            raise ValueError("ERROR: symbols_per_ofdm > available data‐subcarriers")
        padding = np.zeros(zeros_to_pad, dtype=complex)  # zeros for the “unused carriers”
        data_subcarrier_syms = np.concatenate([payload_syms, padding])  # length = 4095

        # SAVE the first-1944 symbols for the “info dict” so Rx can compare them:
        _data_freq_symbols_for_info_dict.append(payload_syms)

        # BUILD the time‐domain OFDM block (without CP) by calling your existing helper:
        block_no_cp = to_real_ofdm_block(data_subcarrier_syms)
        data_blocks.append(block_no_cp)
        #     • to_real_ofdm_block(...) will do an IFFT of length=FFT_LEN, place these 4095 symbols
        #       on the correct subcarriers (1..2047 and 2048..4094), then return a real‐valued time‐domain
        #       block of length=FFT_LEN.


    # Original: np.save(DATA_NPY, data_blocks) # Saved TD blocks with CP
    # Corrected: Save FREQUENCY DOMAIN data symbols
    np.save(DATA_NPY, np.array(_data_freq_symbols_for_info_dict))  # shape=(TX_REPS, 1944)


    # ------------- build pilot blocks (TD without CP) ------------
    pilot_bits = random_bitpairs(n_qpsk)  # Original used default seed_no=42
    freq_pilot, colour = qpsk_gray(pilot_bits)
    pilot = to_real_ofdm_block(freq_pilot)  # 'pilot' is TD pilot block WITHOUT CP (this was correct)
    np.save(COLMAP_NPY, colour)
    np.save(PILOT_NPY, freq_pilot)  # Saves pilot frequency symbols

    # ------------- build sequence --> 'payload' list will contain TD blocks WITHOUT CP -------------
    payload = []  # This list will contain TD blocks (pilots and data), all WITHOUT CP
    # The original local variable 'payload_data_blocks' is no longer needed here for its old purpose.
    # The 'payload_data_blocks' key in the output 'info' dict will use _data_freq_symbols_for_info_dict.
    payload_type = []
    for i in range(TX_REPS):
        payload.append(pilot)  # pilot is TD, no CP
        payload_type.append('pilot')
        payload.append(data_blocks[i])  # data_blocks[i] is now TD, no CP
        # Original: payload_data_blocks.append(data_blocks[i]) # This was for TD with CP data
        payload_type.append('data')

    sequence = [
        silence,
        chirp_up,
        *payload,  # Unpacks TD blocks (pilots & data), all WITHOUT CP
        chirp_down,
        silence
    ]

    # This loop now correctly adds CP to all OFDM blocks and the trailing chirp_down
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
    # Sum of lengths of OFDM payload blocks (pilots and data) *after* CP has been added.
    # These are sequence[2] through sequence[2 + len(payload_type) - 1].
    # All these blocks now have length (FFT_LEN + CP_LEN).
    # len(payload_type) is the total number of OFDM payload blocks (2 * TX_REPS).
    calculated_total_ofdm_length = len(payload_type) * (FFT_LEN + CP_LEN)

    # ------------- flat dictionary ---> key{waveform} is a flattened sequence, key{waveform_blocks} is unflattened -------------
    info = {
        "leading_silence_samples": len(silence),
        "chirp_samples": len(chirp_up),  # Length of chirp signal (no CP)
        "ofdm_block_len": FFT_LEN,
        "ofdm_block_len_with_cp": (len(sequence[2])),  # Length of first payload block (with CP)
        "cp_len": CP_LEN,
        "block_real?": np.isrealobj(pilot),  # 'pilot' is TD, no CP
        "total_ofdm_length": calculated_total_ofdm_length,  # CORRECTED
        "final_len": len(np.concatenate(sequence)),  # Use concatenated sequence
        "no_of_payload_blocks": len(payload_type),  # Total pilot + data blocks
        "waveform_blocks": sequence,
        "payload_type_list": payload_type,
        # Corrected: 'payload_data_blocks' key now gets frequency domain QPSK symbols for data blocks
        "payload_data_blocks": np.array(_data_freq_symbols_for_info_dict),
    }
    return {"waveform": np.concatenate(sequence), **info}  # Use concatenated sequence

# This was the original line for testing, kept for consistency:
output = prepare_tx_sequence(True)

if __name__ == "__main__":

    pass

# def play_audio(sig:np.ndarray, fs:int=FS): # Definition was in original but commented out
# sd.play(sig, fs); sd.wait()


## THIS CODE SHOULD OUTPUT:
#colour_map.npy        ←  length=4095 array of ints (unchanged)
#data_symbols.npy      ←  shape = (TX_REPS, 1944)   # ← changed!
#pilot_symbols.npy     ←  shape = (4095,)          # unchanged
#tx_sequence.wav       ←  real-valued samples (unchanged format)
