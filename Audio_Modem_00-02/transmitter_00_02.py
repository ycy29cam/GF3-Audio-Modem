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
F0, F1          = 20, 10000      # chirp start / end frequencies (Hz)
TX_REPS         = 10              # 1 pilot + 7 identical data blocks 
WAV_TX          = 'tx_sequence.wav'
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
DATA_NPY        = 'data_symbols.npy'
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
    x *= TARGET_PEAK/np.max(np.abs(x))
    return x

def add_cyclic_prefix(x:np.ndarray, cp_len:int=CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])


def prepare_tx_sequence(plot = False) -> dict:
    # ------------- build pieces -------------
    silence     = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = add_cyclic_prefix(generate_chirp(F1, F0, CHIRP_LEN_S))


    # ------------- build data blocks, returns data_blocks 2D array -------------
    long_bits = random_bitpairs(n = (TX_REPS * (FFT_LEN + CP_LEN)), seed_no=24)
    data_blocks = []
    n_qpsk = FFT_LEN//2 - 1
    for i in range(TX_REPS):
        bits = long_bits[i * n_qpsk : (i + 1) * n_qpsk]
        syms, _ = qpsk_gray(bits)
        block = add_cyclic_prefix(to_real_ofdm_block(syms))
        data_blocks.append(block)
    np.save(DATA_NPY, data_blocks)

    # ------------- build pilot blocks, returns pilot ------------
    pilot_bits = random_bitpairs(n_qpsk)
    freq_pilot, colour = qpsk_gray(pilot_bits)
    pilot = to_real_ofdm_block(freq_pilot)            
    np.save(COLMAP_NPY, colour)                   
    np.save(PILOT_NPY, freq_pilot)

    
    # ------------- build sequence --> var(sequence) is a 2D array with time signal blocks in it -------------
    payload = []
    payload_type = []
    for i in range(TX_REPS):
        payload.append(pilot)
        payload_type.append('pilot')
        payload.append(data_blocks[i])
        payload_type.append('data')


    sequence = [
        silence,
        chirp_up,
        *payload,  # interleave pilot and data blocks
        chirp_down,
        silence
    ]

    for i in range(2, len(sequence) - 1):
        sequence[i] = add_cyclic_prefix(sequence[i])
    

    sf.write(WAV_TX, np.concatenate(sequence), FS)
# ------------- plot function -------------
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(np.concatenate(sequence), lw=0.7)
        plt.title("Transmit waveform (time domain)")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.tight_layout()
        plt.show()

# ------------- flat dictionary ---> key{waveform} is a flattened sequence, key{waveform_blocks} is unflattened -------------
    info = {
        "leading_silence_samples": len(silence),
        "chirp_samples"          : len(chirp_up),
        "ofdm_block_len"         : FFT_LEN,
        "ofdm_block_len_with_cp" : (len(sequence[2])),
        "cp_len"                 : CP_LEN,
        "block_real?"            : np.isrealobj(pilot),
        "total_ofdm_length"      : len(np.concatenate(sequence)) - len(chirp_up) - len(add_cyclic_prefix(chirp_down)) - len(silence),
        "final_len"              : len(np.concatenate(sequence)),
        "no_of_payload_blocks"   : len(sequence) - 3,  # excluding silence and chirps
        "waveform_blocks"        : sequence,
        "payload_type_list"      : payload_type,
    }
    return { "waveform": np.concatenate(sequence), **info}

# def prepare_tx_sequence() -> dict:
#     # ------------- build pieces -------------
#     silence     = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
#     chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
#     chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

#     n_qpsk      = FFT_LEN//2 - 1
#     pilot_bits        = random_bitpairs(n_qpsk)
#     data_bits         = random_bitpairs(n_qpsk, seed_no=24)  # different seed for data
#     pilot, colour       = qpsk_gray(pilot_bits)            # also stores colour map
#     np.save(COLMAP_NPY, colour)                   # where pilot colours get saved go in
#     data, colour       = qpsk_gray(data_bits)            # also stores colour map
    
#     np.save(PILOT_NPY, pilot)
#     np.save(DATA_NPY, data)  # save data symbols for later use

#     blk_td      = to_real_ofdm_block(pilot)
#     data_td   = to_real_ofdm_block(data)
#     blk_td_cp   = add_cyclic_prefix(blk_td)  # CP only on first block 
#     data_td_cp  = add_cyclic_prefix(data_td)  # CP only on first block

    
#     # ------------- build sequence -------------
#     sequence = np.concatenate([
#         silence,
#         chirp_up,
#         np.tile(blk_td_cp, 2),         # 2 pilot blocks with CP
#         data_td_cp,                    # 1 data block with CP
#         np.tile(blk_td_cp, 2),         # 2 more pilot blocks with CP
#         add_cyclic_prefix(chirp_down)  # down-chirp with CP
#     ])
#     # plot of the waveform

#     # plt = plt.figure(figsize=(10,3))
#     sf.write(WAV_TX, sequence, FS)
#     plt.figure(figsize=(10,3))
#     plt.plot(sequence, lw=.7)
#     plt.title("Transmit waveform (time domain)")
#     plt.xlabel("sample"); plt.ylabel("amplitude")
#     plt.tight_layout(); plt.show()

#     info = {
#         "leading_silence_samples": len(silence),
#         "chirp_samples"          : len(chirp_up),
#         "ofdm_block_len"         : len(blk_td),
#         "ofdm_block_len_prefix"  : len(blk_td_cp),
#         "cp_len"                 : CP_LEN,
#         "block_real?"            : np.isrealobj(blk_td),
#         "total_ofdm_length"      : len(sequence) - len(chirp_up) - len(add_cyclic_prefix(chirp_down)) - len(silence),
#         "final_len"              : len(sequence)
#     }
#     return dict(waveform=sequence, info=info)

def play_audio(sig:np.ndarray, fs:int=FS):
    sd.play(sig, fs); sd.wait()

output = prepare_tx_sequence(True)

if __name__ == "__main__":

    # play_audio(output["waveform"])
    pass
