import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft


FS              = 48_000         # audio sample-rate (Hz)
FFT_LEN         = 8192           # size of one OFDM symbol (must be even)
CP_LEN          = FFT_LEN // 4   # cyclic-prefix length
CHIRP_LEN_S     = 0.5              # chirp duration (seconds)
SILENCE_LEN_S   = 1.0
F0, F1          = 20, 15000      # chirp start / end frequencies (Hz)
TX_REPS         = 5              # 1 pilot + 7 identical data blocks (Comment seems outdated, it's TX_REPS pilot/data pairs)
WAV_TX          = 'tx_sequence.wav'
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
DATA_NPY        = 'data_symbols.npy' # Will now store frequency-domain data symbols
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

CHIRP_ATTEN      = 0.80            # scale applied to both chirps
TARGET_PEAK      = 0.80            # peak of every OFDM block after scaling
LENGTH_TOL       = 10

Q_COL = { (0,0):'#d62728',  # red
          (0,1):'#1f77b4',  # blue
          (1,1):'#2ca02c',  # green
          (1,0):'#ff7f0e'}  # orange

def generate_chirp(f0, f1, dur, fs=FS):
    t = np.arange(int(dur*fs))/fs
    return (CHIRP_ATTEN*signal.chirp(t, f0, t[-1], f1)).astype(np.float32)

def random_bitpairs(n, seed_no=42):
    """ generates a choppable random sequence of bit pairs
        Aruments:
        n (int): number of bit pairs to generate, if not specified, defaults to maximum length of block emission
        seed_no (int): random seed for reproducibility"""
    np.random.seed(seed_no)
    return np.random.randint(0, 2, size=(n,2), dtype=np.int8)

def qpsk_gray(bitpairs):
    mapping = {(0,0):1+1j, (0,1):1-1j, (1,1):-1-1j, (1,0):-1+1j}
    syms    = np.array([mapping[tuple(b)] for b in bitpairs], np.complex64)
    colours = np.array([Q_COL[tuple(b)]  for b in bitpairs])
    return syms, colours      # return colours to replicate later

def to_real_ofdm_block(useful_freq_symbols, n=FFT_LEN):
    """ converts a sequence of useful frequency symbols to a real OFDM block
    """
    half = n//2
    X = np.zeros(n, np.complex64)
    X[1:half]   = useful_freq_symbols
    X[half+1:]  = np.conj(useful_freq_symbols[::-1])
    x = fft.ifft(X).real.astype(np.float32)
    # peak normalisation
    peak_val = np.max(np.abs(x))
    if peak_val > 1e-9: # Avoid division by zero for unstable scaling
        x *= TARGET_PEAK/peak_val
    elif TARGET_PEAK == 0:
        x = np.zeros_like(x)
    # If peak_val is very small and TARGET_PEAK is not zero, x remains small
    return x

def add_cyclic_prefix(x:np.ndarray, cp_len:int=CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])


def prepare_tx_sequence(plot = False) -> dict:
    # ------------- build pieces -------------
    silence     = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

    n_qpsk = FFT_LEN//2 - 1

    # ------------- build data blocks (now TD without CP) and collect data freq symbols -------------
    # Original: long_bits = random_bitpairs(n = (TX_REPS * (FFT_LEN + CP_LEN)), seed_no=24)
    # Corrected 'n' for random_bitpairs for all data bits:
    _data_long_bits = random_bitpairs(n=(TX_REPS * n_qpsk), seed_no=24)
    
    data_blocks = []  # This list will now store TIME DOMAIN data blocks WITHOUT CP
    _data_freq_symbols_for_info_dict = [] # To store data QPSK symbols for the output dictionary and .npy file

    for i in range(TX_REPS):
        # Original: bits = long_bits[i * n_qpsk : (i + 1) * n_qpsk]
        bits = _data_long_bits[i * n_qpsk : (i + 1) * n_qpsk]
        syms, _ = qpsk_gray(bits)
        _data_freq_symbols_for_info_dict.append(syms) # Store freq symbols
        
        # Original: block = add_cyclic_prefix(to_real_ofdm_block(syms))
        # Corrected: create TD block WITHOUT CP
        block_no_cp = to_real_ofdm_block(syms)
        data_blocks.append(block_no_cp)
    
    # Original: np.save(DATA_NPY, data_blocks) # Saved TD blocks with CP
    # Corrected: Save FREQUENCY DOMAIN data symbols
    np.save(DATA_NPY, np.array(_data_freq_symbols_for_info_dict))

    # ------------- build pilot blocks (TD without CP) ------------
    pilot_bits = random_bitpairs(n_qpsk) # Original used default seed_no=42
    freq_pilot, colour = qpsk_gray(pilot_bits)
    pilot = to_real_ofdm_block(freq_pilot) # 'pilot' is TD pilot block WITHOUT CP (this was correct)
    np.save(COLMAP_NPY, colour)
    np.save(PILOT_NPY, freq_pilot) # Saves pilot frequency symbols

    # ------------- build sequence --> 'payload' list will contain TD blocks WITHOUT CP -------------
    payload = [] # This list will contain TD blocks (pilots and data), all WITHOUT CP
    # The original local variable 'payload_data_blocks' is no longer needed here for its old purpose.
    # The 'payload_data_blocks' key in the output 'info' dict will use _data_freq_symbols_for_info_dict.
    payload_type = []
    for i in range(TX_REPS):
        payload.append(pilot) # pilot is TD, no CP
        payload_type.append('pilot')
        payload.append(data_blocks[i]) # data_blocks[i] is now TD, no CP
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

    sf.write(WAV_TX, np.concatenate(sequence), FS) # Write concatenated sequence
    
    # ------------- plot function -------------
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(np.concatenate(sequence), lw=0.7) # Plot concatenated sequence
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
        "chirp_samples"          : len(chirp_up), # Length of chirp signal (no CP)
        "ofdm_block_len"         : FFT_LEN,
        "ofdm_block_len_with_cp" : (len(sequence[2])), # Length of first payload block (with CP)
        "cp_len"                 : CP_LEN,
        "block_real?"            : np.isrealobj(pilot), # 'pilot' is TD, no CP
        "total_ofdm_length"      : calculated_total_ofdm_length, # CORRECTED
        "final_len"              : len(np.concatenate(sequence)), # Use concatenated sequence
        "no_of_payload_blocks"   : len(payload_type),  # Total pilot + data blocks
        "waveform_blocks"        : sequence,
        "payload_type_list"      : payload_type,
        # Corrected: 'payload_data_blocks' key now gets frequency domain QPSK symbols for data blocks
        "payload_data_blocks"    : np.array(_data_freq_symbols_for_info_dict),
    }
    return { "waveform": np.concatenate(sequence), **info} # Use concatenated sequence

# This was the original line for testing, kept for consistency:
output = prepare_tx_sequence(True)

def play_audio(sig:np.ndarray, fs:int=FS):
    sd.play(sig, fs); sd.wait()

play_audio(output["waveform"], FS)
print(output)