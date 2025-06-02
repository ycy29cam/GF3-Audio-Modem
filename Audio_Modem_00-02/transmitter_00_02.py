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
TX_REPS         = 8              # 1 pilot + 7 identical data blocks
WAV_TX          = 'tx_sequence.wav'
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
DATA_NPY        = 'data_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

CHIRP_ATTEN      = 0.80            # scale applied to both chirps
TARGET_PEAK      = 0.80            # peak of every OFDM block after scaling
LENGTH_TOL       = 512  

Q_COL = { (0,0):'#d62728',  # red
          (0,1):'#1f77b4',  # blue
          (1,1):'#2ca02c',  # green
          (1,0):'#ff7f0e'}  # orange

def generate_chirp(f0, f1, dur, fs=FS):
    t = np.arange(int(dur*fs))/fs
    return (CHIRP_ATTEN*signal.chirp(t, f0, t[-1], f1)).astype(np.float32)

def random_bitpairs(n, seed_no=42):
    np.random.seed(seed_no)  # for reproducibility
    return np.random.randint(0, 2, size=(n,2), dtype=np.int8) #worth adding a seed for reproducibility
 
def qpsk_gray(bitpairs):
    mapping = {(0,0):1+1j, (0,1):1-1j, (1,1):-1-1j, (1,0):-1+1j}
    syms    = np.array([mapping[tuple(b)] for b in bitpairs], np.complex64)
    colours = np.array([Q_COL[tuple(b)]  for b in bitpairs])
    np.save(COLMAP_NPY, colours)
    return syms, colours      # return colours to replicate later

def to_real_ofdm_block(freq_syms, n=FFT_LEN):
    half = n//2
    X = np.zeros(n, np.complex64)
    X[1:half]   = freq_syms
    X[half+1:]  = np.conj(freq_syms[::-1])
    x = fft.ifft(X).real.astype(np.float32)
    # peak normalisation
    x *= TARGET_PEAK/np.max(np.abs(x))
    return x

def add_cyclic_prefix(x:np.ndarray, cp_len:int=CP_LEN) -> np.ndarray:
    return np.concatenate([x[-cp_len:], x])

def prepare_tx_sequence() -> dict:
    # ------------- build pieces -------------
    silence     = np.zeros(int(SILENCE_LEN_S * FS), np.float32)
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

    n_qpsk      = FFT_LEN//2 - 1
    pilot_bits        = random_bitpairs(n_qpsk)            # where data symbols go in
    data_bits         = random_bitpairs(n_qpsk, seed_no=24)  # different seed for data
    pilot, colour       = qpsk_gray(pilot_bits)            # also stores colour map
    data, colour       = qpsk_gray(data_bits)            # also stores colour map
    
    np.save(PILOT_NPY, pilot)
    np.save(DATA_NPY, data)  # save data symbols for later use

    blk_td      = to_real_ofdm_block(pilot)
    data_td   = to_real_ofdm_block(data)
    blk_td_cp   = add_cyclic_prefix(blk_td)  # CP only on first block 
    data_td_cp  = add_cyclic_prefix(data_td)  # CP only on first block

    
    # ------------- build sequence -------------
    sequence = np.concatenate([
        silence,
        chirp_up,
        np.tile(blk_td_cp, 2),         # 2 pilot blocks with CP
        data_td_cp,                    # 1 data block with CP
        np.tile(blk_td_cp, 2),         # 2 more pilot blocks with CP
        add_cyclic_prefix(chirp_down)  # down-chirp with CP
    ])
    # plot of the waveform

    # plt = plt.figure(figsize=(10,3))
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
        "total_ofdm_length"      : len(sequence) - len(chirp_up) - len(add_cyclic_prefix(chirp_down)) - len(silence),
        "final_len"              : len(sequence)
    }
    return dict(waveform=sequence, info=info)

def play_audio(sig:np.ndarray, fs:int=FS):
    sd.play(sig, fs); sd.wait()

output = prepare_tx_sequence()
print(output["waveform"])
print(output["info"])

if __name__ == "__main__":
    play_audio(output["waveform"])
