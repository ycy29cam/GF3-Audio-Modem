import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from scipy.io.wavfile import read
from transmitter_00_03 import generate_chirp, WAV_TX, output, Q_COL
import transmitter_00_03 as tx

FS              = tx.FS
FFT_LEN         = tx.FFT_LEN
CP_LEN          = tx.CP_LEN
CHIRP_LEN_S     = tx.CHIRP_LEN_S
SILENCE_LEN_S   = tx.SILENCE_LEN_S
F0, F1          = tx.F0, tx.F1
TX_REPS         = tx.TX_REPS
WAV_TX          = WAV_TX
WAV_RX          = 'rx_recording.wav'
WAV_RX_1        = 'rx_recording_group1.wav'
WAV_RX_3        = 'rx_recording_group3.wav'
PILOT_NPY       = 'pilot_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"

CHIRP_ATTEN     = tx.CHIRP_ATTEN
TARGET_PEAK     = tx.TARGET_PEAK
LENGTH_TOL      = tx.LENGTH_TOL

def record_audio(expected_len:int, fs:int=FS) -> np.ndarray:
    print(f"Recording ≈{expected_len/fs:.2f} s …")
    rec = sd.rec(expected_len, samplerate=fs, channels=1,
                 dtype='float32').squeeze()
    sd.wait()
    sf.write(WAV_RX, rec, fs)
    return rec

def load_wav(path):
    data, sr = sf.read(path, always_2d=False)
    assert sr == FS, "sample-rate mismatch"
    return data.astype(np.float32)

def start_end_synchronise(rx: np.ndarray,
                          chirp_up: np.ndarray,
                          chirp_down: np.ndarray) -> tuple[np.ndarray, int, int, int]:

    corr_up   = signal.correlate(rx, chirp_up, mode='valid')
    peak_up   = np.argmax(corr_up)
    corr_down = signal.correlate(rx, chirp_down, mode='valid')


    search_from = peak_up + len(chirp_up)
    peak_down_locs = np.where(corr_down > 0.2 * corr_down.max())[0]
    peak_down = peak_down_locs[peak_down_locs > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down + 4

    plt.plot(corr_up, label='up-chirp correlation')
    plt.plot(corr_down, label='down-chirp correlation')
    plt.plot(rx * 5000, label='received signal', alpha=0.5)
    plt.axvline(start_payload, color='red', linestyle='--', label='start_payload')
    plt.axvline(end_payload, color='red', linestyle='--', label='end_payload')

    print("start_payload:", start_payload, "end_payload:", end_payload)

    payload = rx[start_payload:end_payload]
    print(payload.shape)

    block_len = 10240
    n_blocks = int(round(len(payload) / 10240))
    padded_len = n_blocks * block_len
    print("unpadded vs padded difference is: ", abs(padded_len - len(payload)))
    valid_blocks = n_blocks

    if len(payload) < padded_len:
        payload = np.pad(payload, (0, padded_len - len(payload)))
    else:
        payload = payload[:padded_len]

    last_valid_block_index = valid_blocks

    return payload, start_payload, end_payload, last_valid_block_index

def sync_chopper(payload, start_payload, end_payload, rx, last_valid_block_index, block_length_time = output["ofdm_block_len_with_cp"]):
    time_blocks = []
    x = int(start_payload - block_length_time/2)
    y = int(start_payload + block_length_time*(3/2))
    sync_peak_index = []
    sync_max = []
    time_pilot_blocks_no_cp = np.load(PILOT_TIME_NO_CP_NPY)
    print("the last valid block index is: ", last_valid_block_index)
    for i in range(last_valid_block_index//5):
        window = rx[x:y]
        pilot_correlation = signal.correlate(window, time_pilot_blocks_no_cp[i] )
        plt.plot(pilot_correlation)
        plt.show()
        sync_start = x + np.argmax(pilot_correlation) - FFT_LEN
        sync_max.append(np.max(pilot_correlation))
        sync_peak_index.append(sync_start)
        chopped_start_index = start_payload + i * 5 * block_length_time + CP_LEN
        bit_diff = (sync_start - chopped_start_index)
        if abs(bit_diff) > 5:
            print(f" Desync on pilot block {i}: sync start index {sync_start}, expected {chopped_start_index}, diff = {bit_diff} bits")
            sync_peak_index[-1] = chopped_start_index
        x += 5*block_length_time
        y += 5*block_length_time

    for i in sync_peak_index:
        start = i
        for _ in range(5):
            block = rx[start : start + FFT_LEN]
            print("time block length is: ", len(block))
            time_blocks.append(block)
            print(start)
            start += block_length_time
    return np.array(time_blocks)

def time_OFDM_chopper(payload, block_length_time = output["ofdm_block_len_with_cp"], cp_len=CP_LEN):
    time_blocks = []
    if len(payload) % block_length_time != 0:
        num_blocks = len(payload) // block_length_time
        payload = payload[:num_blocks * block_length_time]
        print(f"Payload trimmed to {len(payload)} samples to fit {num_blocks} blocks.")

    num_blocks = len(payload) // block_length_time
    for i in range(num_blocks):
        block_with_cp = payload[i * block_length_time : (i + 1) * block_length_time]
        block_no_cp = block_with_cp[cp_len:]
        time_blocks.append(block_no_cp)
    return np.array(time_blocks)

def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2]

def channel_estimation(blocks: np.ndarray,
                       pilot_symbols: np.ndarray,
                       method: str = 'zf',
                       noise_var: float = 1e-4) -> np.ndarray:

    eps = 1e-12
    payload_type_list = output["payload_type_list"]
    N = len(payload_type_list)
    estimates = [None] * N

    pilot_counter = 0
    for i, block_type in enumerate(payload_type_list):
        if block_type == 'pilot':
            if pilot_counter < len(pilot_symbols):
                estimates[i] = blocks[i] / (pilot_symbols[pilot_counter] + eps)
                pilot_counter += 1
            else:
                estimates[i] = np.zeros_like(blocks[i])

    for i, block_type in enumerate(payload_type_list):
        if block_type == 'data':
            prev_idx = next((j for j in range(i - 1, -1, -1) if estimates[j] is not None), None)
            next_idx = next((j for j in range(i + 1, N) if estimates[j] is not None), None)

            to_avg = []
            if prev_idx is not None:
                to_avg.append(estimates[prev_idx])
            if next_idx is not None:
                to_avg.append(estimates[next_idx])

            if to_avg:
                estimates[i] = np.mean(np.stack(to_avg, axis=0), axis=0)
            else:
                estimates[i] = np.ones_like(blocks[i])

    H_est = np.stack(estimates, axis=0)
    np.save(CHAN_NPY, H_est)
    return H_est


def reconstruct_data_blocks(useful_frequency_blocks, H_est_array):
    payload_type_list = output["payload_type_list"]
    assert len(useful_frequency_blocks) == len(payload_type_list), "Mismatch between blocks and payload types"
    assert len(H_est_array) == len(payload_type_list), "Mismatch between channel estimates and payload types"
    data_blocks = np.array([useful_frequency_blocks[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    data_H_est_array = np.array([H_est_array[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    decoded_datablocks = data_blocks/data_H_est_array
    return decoded_datablocks

def equalise(rx_fd, H):
    return rx_fd / H

def plot_channel(H:np.ndarray):
    fig, ax = plt.subplots(2, 1, figsize=(9,4), sharex=True)
    ax[0].plot(20*np.log10(np.abs(H)+1e-12))
    ax[0].set_ylabel("|H| [dB]")
    ax[0].set_title("Estimated channel magnitude / phase")
    ax[1].plot(np.angle(H))
    ax[1].set_ylabel("∠H [rad]")
    ax[1].set_xlabel("sub-carrier")
    plt.tight_layout(); plt.show()

def compare_tx_rx(rx:np.ndarray, start_rx_payload:int, end_rx_payload_boundary:int, tx_path:str=WAV_TX):
    tx_sig   = load_wav(tx_path)

    tx_leading_silence = output["leading_silence_samples"]
    tx_chirp_len = output["chirp_samples"]
    tx_start_of_payload = tx_leading_silence + tx_chirp_len
    payload_length_to_compare = output["total_ofdm_length"]

    tx_seg_end = tx_start_of_payload + payload_length_to_compare
    rx_seg_end = start_rx_payload + payload_length_to_compare

    if tx_start_of_payload >= tx_seg_end or tx_seg_end > len(tx_sig):
        print("Warning: TX segment for comparison is invalid or out of bounds.")
        return
    if start_rx_payload >= rx_seg_end or rx_seg_end > len(rx):
        print("Warning: RX segment for comparison is invalid or out of bounds.")
        return

    tx_payload_seg = tx_sig[tx_start_of_payload : tx_seg_end]
    rx_payload_seg = rx[start_rx_payload : rx_seg_end]

    m_peak = np.max(np.abs(rx_payload_seg)) if rx_payload_seg.size > 0 else 0
    n_peak = np.max(np.abs(tx_payload_seg)) if tx_payload_seg.size > 0 else 0
    if n_peak > 0:
        tx_norm = tx_payload_seg / n_peak
    if m_peak > 0:
        rx_norm = rx_payload_seg / m_peak

    plt.figure(figsize=(10,3))
    if n_peak > 0:
        plt.plot(tx_norm, label='TX Payload (norm.)', lw=.8)
    if m_peak > 0:
        plt.plot(rx_norm, label='RX Payload (norm.)', lw=.6, alpha=.7)
    plt.title("TX vs RX OFDM Payload (aligned)")
    plt.xlabel("sample in payload")
    plt.ylabel("normalised amplitude")
    plt.legend(); plt.tight_layout(); plt.show()

def _means_by_colour(z_flat, colours_flat):
    ucols = np.unique(colours_flat)
    means = {c: np.mean(z_flat[colours_flat == c]) for c in ucols}
    return means

def spectrum_plot(sig:np.ndarray, fs:int=FS):
    f, Pxx = signal.welch(sig, fs, nperseg=4096)
    plt.figure(); plt.semilogy(f, Pxx)
    plt.title("Received PSD"); plt.xlabel("Hz"); plt.ylabel("PSD [V²/Hz]")
    plt.tight_layout(); plt.show()

def plot_equalised_blocks(equalised_data_blocks: np.ndarray, sequenced_data_blocks: np.ndarray):
    assert equalised_data_blocks.shape == sequenced_data_blocks.shape, "Shape mismatch between TX and RX blocks"

    eq_flat = equalised_data_blocks.ravel()
    tx_flat = sequenced_data_blocks.ravel()

    sym_to_colour = {1+1j: Q_COL[(0, 0)],
                       1-1j: Q_COL[(0, 1)],
                      -1-1j: Q_COL[(1, 1)],
                      -1+1j: Q_COL[(1, 0)]}

    tx_colours = np.array([sym_to_colour.get(complex(round(s.real), round(s.imag)), 'k')
                           for s in tx_flat])

    plt.figure(); plt.axhline(0, c='k'); plt.axvline(0, c='k')
    plt.scatter(eq_flat.real, eq_flat.imag,
                c=tx_colours, s=12, alpha=.85, edgecolors='none')

    unique_colours = np.unique(tx_colours)
    for c in unique_colours:
        mask = tx_colours == c
        if np.any(mask):
            mean = np.mean(eq_flat[mask])
            plt.plot(mean.real, mean.imag, 'kx')
            plt.text(mean.real, mean.imag, f"{mean.real:+.2f}{mean.imag:+.2f}j",
                     fontsize=7, ha='left', va='bottom')

    bits_to_sym = {'00': '1+1j', '01': '1-1j', '11': '-1-1j', '10': '-1+1j'}
    legend_handles = []
    for bits, colour in Q_COL.items():
        bit_str = ''.join(map(str, bits))
        label = bits_to_sym.get(bit_str, bit_str)
        legend_handles.append(Patch(facecolor=colour, label=label))
    plt.legend(handles=legend_handles, loc='upper right', fontsize='small')

    plt.title("Equalised Constellation (coloured by TX symbols)")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # record_audio(960000)
    SAMPLE_RATE, recording = read('rx_recording.wav')
    # recording = output["waveform"]
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

    payload, start_payload, end_payload, last_valid_block_index  = start_end_synchronise(recording, chirp_up, chirp_down)
    time_blocks = sync_chopper(payload ,start_payload, end_payload, recording,last_valid_block_index )
    # time_blocks = time_OFDM_chopper(payload)
    useful_freq_blocks  = freq_domain(time_blocks)
    h_estimated_array = channel_estimation(useful_freq_blocks, np.load(PILOT_NPY), "zf")
    reconstructed_data = reconstruct_data_blocks(useful_freq_blocks, h_estimated_array)

    plot_equalised_blocks(reconstructed_data[8], output["payload_data_blocks"][8])

    print(len(recording))
    print ("transmitted signal: ", payload,"start bin: ", start_payload, "end bin: ", end_payload)
    compare_tx_rx(recording, start_payload, end_payload)
    spectrum_plot(recording)