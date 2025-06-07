import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal, fft


FS              = 48_000         # audio sample-rate (Hz)
FFT_LEN         = 8192           # size of one OFDM symbol (must be even)
CP_LEN          = FFT_LEN // 4   # cyclic-prefix length
CHIRP_LEN_S     = 1/2           # chirp duration (seconds)
SILENCE_LEN_S   = 1.0
F0, F1          = 3000,0     # chirp start / end frequencies (Hz)
TX_REPS         = 5              # 1 pilot + 7 identical data blocks (Comment seems outdated, it's TX_REPS pilot/data pairs)
WAV_TX          = 'tx_sequence.wav'
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
DATA_NPY        = 'data_symbols.npy' # Will now store frequency-domain data symbols
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"

CHIRP_ATTEN      = 0.80            # scale applied to both chirps
TARGET_PEAK      = 0.80            # peak of every OFDM block after scaling
LENGTH_TOL       = 10

Q_COL = { (0,0):'#d62728',  # red
          (0,1):'#1f77b4',  # blue
          (1,1):'#2ca02c',  # green
          (1,0):'#ff7f0e'}  # orange

# def generate_chirp(f0, f1, dur, fs=FS):
#     t = np.arange(int(dur*fs))/fs
#     return (CHIRP_ATTEN*signal.chirp(t, f0, t[-1], f1)).astype(np.float32)

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
    peak_val = np.max(np.abs(x))
    if peak_val > 1e-9: # Avoid division by zero for unstable scaling
        x *= TARGET_PEAK/peak_val
    elif TARGET_PEAK == 0:
        x = np.zeros_like(x)
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
    data_long_bits = random_bitpairs(n=(n_qpsk*200), seed_no=24)
    data_blocks = []  
    data_freq_symbols_ = [] 
    
    for i in range(200):
        bits = data_long_bits[i * n_qpsk : (i + 1) * n_qpsk]
        freq_pilot, _ = qpsk_gray(bits)
        data_freq_symbols_.append(freq_pilot) 
        block_no_cp = to_real_ofdm_block(freq_pilot)
        data_blocks.append(block_no_cp)


    np.save(DATA_NPY, np.array(data_freq_symbols_))

    # ------------- build pilot blocks array, (TD without CP) ------------
    pilot_long_bits = random_bitpairs(n =(n_qpsk * 200))
    time_pilot_blocks_no_CP = []
    pilot_freq_symbols = [] 
    pilot_colours = []


    for pilot in range(200):
        pilot_bits = pilot_long_bits[i * n_qpsk : (i + 1) * n_qpsk]
        freq_pilot, colour = qpsk_gray(bits)
        pilot_freq_symbols.append(freq_pilot)
        pilot_colours.append(colour)
        block_no_cp = to_real_ofdm_block(freq_pilot)
        time_pilot_blocks_no_CP.append(block_no_cp)


    np.save(COLMAP_NPY, np.array(pilot_colours, dtype=object))
    np.save(PILOT_NPY, pilot_freq_symbols)
    np.save(PILOT_TIME_NO_CP_NPY, time_pilot_blocks_no_CP)

    # ------------- build sequence --> 'payload' list will contain TD blocks WITHOUT CP -------------
    payload = [] # no CP
    payload_type = []
    for i in range(TX_REPS):
        payload.append(time_pilot_blocks_no_CP[i])
        payload_type.append('pilot')
        for j in range(4):
            payload.append(data_blocks[i * 4 + j])
            payload_type.append('data')


    sequence = [
        silence,
        chirp_up,
        *payload,  # Unpacks TD blocks (pilots & data), all WITHOUT CP
        chirp_down,
        silence
    ]

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
    calculated_total_ofdm_length = len(payload_type) * (FFT_LEN + CP_LEN)
    
    # ------------- flat dictionary ---> key{waveform} is a flattened sequence, key{waveform_blocks} is unflattened -------------
    info = {
        "leading_silence_samples": len(silence),
        "chirp_samples"          : len(chirp_up), # Length of chirp signal (no CP)
        "ofdm_block_len"         : FFT_LEN,
        "ofdm_block_len_with_cp" : (len(sequence[2])), # Length of first payload block (with CP)
        "cp_len"                 : CP_LEN,
        "block_real?"            : np.isrealobj(payload.append(time_pilot_blocks_no_CP[2])), # 'pilot' is TD, no CP
        "total_ofdm_length"      : calculated_total_ofdm_length, 
        "final_len"              : len(np.concatenate(sequence)), 
        "no_of_payload_blocks"   : len(payload_type),
        "waveform_blocks"        : sequence,
        "payload_type_list"      : payload_type,
        "payload_data_blocks"    : np.array(data_freq_symbols_),
    }
    return { "waveform": np.concatenate(sequence), **info} 


output = prepare_tx_sequence(True)
if __name__ == "__main__":
    # output = prepare_tx_sequence(True)
    pass


