import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from scipy.io.wavfile import read
from Tx_LPDC_2 import generate_chirp, WAV_TX, Q_COL
import Transmitter3000 as tx
import ldpc_jossy
import pickle

# look into using GPU for hardware acceleration of convolution, e.g. using CuPy for faster processing

# ------------------------------------------------
#   1.  General parameters (unchanged)
# ------------------------------------------------
FS                   = tx.FS
FFT_LEN              = tx.FFT_LEN
CP_LEN               = tx.CP_LEN
CHIRP_LEN_S          = tx.CHIRP_LEN_S
SILENCE_LEN_S        = tx.SILENCE_LEN_S
F0, F1               = tx.F0, tx.F1
TX_REPS              = tx.TX_REPS
WAV_TX               = WAV_TX
CHIRP_ATTEN          = tx.CHIRP_ATTEN
TARGET_PEAK          = tx.TARGET_PEAK
LENGTH_TOL           = tx.LENGTH_TOL
WAV_RX               = 'rx_recording.wav'
# WAV_RX_1             = 'rx_recording_group1.wav'
# WAV_RX_3             = 'rx_recording_group3.wav'
PILOT_NPY            = 'pilot_symbols.npy'
COLMAP_NPY           = 'colour_map.npy'
CHAN_NPY             = 'channel_estimate.npy'
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"
OUTPUT               = 'output_dict.pkl'
LDPC_Z               = 81
LDPC_N               = 24 * LDPC_Z
LDPC_K               = LDPC_N // 2
#LLR_MAX             = 50.0
LLR_MAX              = 200.0
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"


# Instantiate the same “802.11n” LDPC code that you used in Tx:
my_ldpc = ldpc_jossy.code(standard='802.11n', rate='1/2', z=LDPC_Z)
# Sanity check: each OFDM symbol carries two codewords,
assert 2 * (LDPC_N // 2) <= (FFT_LEN // 2 - 1)
with open(OUTPUT, 'rb') as fp:
    output = pickle.load(fp)

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
    peak_down_locs = np.where(corr_down > 0.8 * corr_down.max())[0]
    peak_down = peak_down_locs[peak_down_locs > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down

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

def sync_chopper(payload, last_valid_block_index, block_length_time = output["ofdm_block_len_with_cp"], cp_len=CP_LEN):
    """
    Corrects for sample clock offset by resampling the payload. It finds the optimal
    resampling factor by maximizing the sum of pilot correlation peaks.
    """
    time_pilot_blocks_no_cp = np.load(PILOT_TIME_NO_CP_NPY)

    # 1. Generate a range of resampling factors to test (e.g., +/- 500 parts-per-million)
    resampling_factors = np.linspace(0.9995, 1.0005, 51)
    correlation_sums = []

    # This nested helper function calculates the correlation score for a given payload
    def get_correlation_sum(current_payload, current_block_len):
        total_max_corr = 0
        num_pilot_groups = last_valid_block_index // 5

        for i in range(num_pilot_groups):
            # Define a search window based on the expected position of the pilot
            expected_pilot_pos = int(i * 5 * current_block_len)
            search_start = max(0, expected_pilot_pos - current_block_len)
            search_end = min(len(current_payload), expected_pilot_pos + current_block_len)
            window = current_payload[search_start:search_end]
            
            # Ensure the window is large enough for correlation
            if len(window) < len(time_pilot_blocks_no_cp[i]):
                continue

            pilot_correlation = signal.correlate(window, time_pilot_blocks_no_cp[i], mode='valid')
            if pilot_correlation.size > 0:
                total_max_corr += np.max(pilot_correlation)
        
        return total_max_corr

    # 2. For each factor, resample the payload and calculate its correlation score
    for factor in resampling_factors:
        resampled_len = int(len(payload) * factor)
        resampled_payload = signal.resample(payload, resampled_len)
        scaled_block_len = int(block_length_time * factor)
        
        current_sum = get_correlation_sum(resampled_payload, scaled_block_len)
        correlation_sums.append(current_sum)

    # 3. Find the best resampling factor and create the final, corrected payload
    if not correlation_sums or max(correlation_sums) == 0:
         print("Warning: Could not find any pilot correlation. Using original payload.")
         best_factor = 1.0
         final_payload = payload
    else:
        best_idx = np.argmax(correlation_sums)
        best_factor = resampling_factors[best_idx]
        final_len = int(len(payload) * best_factor)
        final_payload = signal.resample(payload, final_len)
        print(f"Optimal resampling factor found: {best_factor:.6f}")

    # 4. Plot the graph of correlation sums vs. resampling factors
    plt.figure(figsize=(9, 5))
    plt.plot(resampling_factors, correlation_sums, '.-')
    plt.axvline(best_factor, color='r', linestyle='--', label=f'Best Factor: {best_factor:.6f}')
    plt.title("Resampling Factor vs. Sum of Max Pilot Correlations")
    plt.xlabel("Resampling Factor")
    plt.ylabel("Sum of Max Correlation Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Chop the optimally resampled payload into blocks
    final_block_len = int(block_length_time * best_factor)
    final_cp_len = int(cp_len * best_factor)
    
    time_blocks = []
    num_blocks = len(final_payload) // final_block_len
    for i in range(num_blocks):
        block_with_cp = final_payload[i * final_block_len : (i + 1) * final_block_len]
        
        if len(block_with_cp) > final_cp_len:
            block_no_cp = block_with_cp[final_cp_len:]

            # CRITICAL FIX: Resample the chopped block to the exact FFT length
            block_for_fft = signal.resample(block_no_cp, FFT_LEN)
            time_blocks.append(block_for_fft)
            
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
    # payload_type_list = output["payload_type_list"]
    payload_type_list = ["pilot" if i % 5 == 0 else "data" for i in range(last_valid_block_index)]
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

def phase_error_correction(equalised_blocks, pilot_symbols):
    """
    Corrects phase error and returns ONLY the corrected data blocks.

    This function identifies pilot blocks, calculates the average phase offset
    for each one, and interpolates these phase errors to correct all blocks.
    Finally, it filters out the pilot blocks and returns only the clean,
    corrected data blocks.

    Args:
        equalised_blocks (np.ndarray): All blocks (pilots and data) after
                                     frequency-domain equalization.
        pilot_symbols (np.ndarray): The known, ideal pilot symbols.
        payload_type_list (list): A list of strings ('pilot' or 'data')
                                  indicating the type of each block.

    Returns:
        np.ndarray: A new array containing ONLY the phase-corrected DATA blocks.
    """
    # 0. Create Payload_type_list properly lolll
    payload_type_list = ["pilot" if i % 5 == 0 else "data" for i in range(last_valid_block_index)]

    # 1. Find the indices of pilot blocks to use as a reference
    pilot_indices = [i for i, block_type in enumerate(payload_type_list) if block_type == 'pilot']
    pilot_indices = [i for i in pilot_indices if i < len(equalised_blocks)]

    if not pilot_indices:
        print("Warning: No pilot blocks found. Cannot perform phase correction.")
        return np.array([]) # Return an empty array as no data can be processed

    # 2. Measure the average phase error for each pilot block
    measured_phase_errors = []
    tx_pilot_counter = 0
    for rx_pilot_index in pilot_indices:
        if tx_pilot_counter < len(pilot_symbols):
            rx_pilot_block = equalised_blocks[rx_pilot_index]
            tx_pilot_block = pilot_symbols[tx_pilot_counter]
            
            error_vector_sum = np.sum(tx_pilot_block * np.conj(rx_pilot_block))
            avg_phase_error = np.angle(error_vector_sum)
            measured_phase_errors.append(avg_phase_error)
            tx_pilot_counter += 1

    if not measured_phase_errors:
        print("Warning: Could not measure phase error. Cannot perform phase correction.")
        return np.array([])

    # 3. Interpolate the measured errors across all block positions
    all_block_indices = np.arange(len(equalised_blocks))
    interpolated_phase_errors = np.interp(
        x=all_block_indices,
        xp=pilot_indices,
        fp=measured_phase_errors
    )
    
    # 4. Apply the phase correction to ALL blocks (pilots and data)
    phase_corrected_all_blocks = np.array([
        block * np.exp(1j * error) for block, error in zip(equalised_blocks, interpolated_phase_errors)
    ])

    # 5. Filter and return ONLY the data blocks, discarding the pilots
    corrected_data_blocks = np.array([
        phase_corrected_all_blocks[i] for i, btype in enumerate(payload_type_list)
        if btype == 'data' and i < len(phase_corrected_all_blocks)
    ])
    
    return corrected_data_blocks

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

def calculate_and_plot_ber(received_symbols, transmitted_symbols):
    """
    Calculates and plots the Bit Error Rate for each subcarrier bin,
    averaged over all data blocks.
    
    Args:
        received_symbols (np.ndarray): The final, corrected data symbols.
                                       Shape: (num_blocks, num_subcarriers)
        transmitted_symbols (np.ndarray): The ideal transmitted data symbols.
                                          Shape: (num_blocks, num_subcarriers)
    """
    # --- Input Validation ---
    if received_symbols.shape != transmitted_symbols.shape:
        print("Error: Shape mismatch between received and transmitted symbols. Cannot calculate BER.")
        return
    
    if received_symbols.ndim != 2 or received_symbols.size == 0:
        print("Error: Input symbols must be a 2D array of (blocks, subcarriers).")
        return

    num_blocks, num_subcarriers = received_symbols.shape

    # --- Demodulation (Symbols to Bits) for each symbol, preserving structure ---
    sym_to_bits_map = {(1, 1): (0, 0), (1, -1): (0, 1), (-1, -1): (1, 1), (-1, 1): (1, 0)}

    # Convert transmitted symbols to a 3D bit array: (blocks, subcarriers, 2 bits)
    tx_bits = np.array(
        [[sym_to_bits_map.get((s.real, s.imag)) for s in block] for block in transmitted_symbols]
    )

    # Perform hard-decision demodulation on received symbols
    detected_real = np.sign(received_symbols.real)
    detected_imag = np.sign(received_symbols.imag)
    # Handle cases where a symbol is exactly on an axis
    detected_real[detected_real == 0] = 1
    detected_imag[detected_imag == 0] = 1
    
    # Convert received symbols to a 3D bit array
    rx_bits = np.zeros_like(tx_bits)
    for i in range(num_blocks):
        for j in range(num_subcarriers):
            coords = (detected_real[i, j], detected_imag[i, j])
            rx_bits[i, j, :] = sym_to_bits_map.get(coords, (0, 0))

    # --- BER Calculation per Bin ---
    # Find all bit errors, resulting in a 3D boolean array
    bit_errors = (rx_bits != tx_bits)
    
    # Sum errors over all blocks and the two bits per symbol for each subcarrier bin
    errors_per_bin = np.sum(bit_errors, axis=(0, 2))
    
    # Calculate BER for each bin
    total_bits_per_bin = num_blocks * 2  # Each symbol carries 2 bits
    ber_per_bin = errors_per_bin / total_bits_per_bin
    
    # Calculate the overall average BER for a summary statistic
    total_errors = np.sum(errors_per_bin)
    total_bits = num_blocks * num_subcarriers * 2
    overall_ber = total_errors / total_bits
    
    print(f"\n--- Bit Error Rate (BER) ---")
    print(f"Total Bits Compared: {total_bits}")
    print(f"Total Bit Errors: {total_errors}")
    print(f"Overall Average BER: {overall_ber:.2e}")

    # --- ADDED: Calculate and print BER for the specific sub-range ---
    start_bin = 200
    end_bin = 2143  # Inclusive
    # Check if the requested range is valid for the number of subcarriers
    if num_subcarriers > end_bin:
        # Slice the ber_per_bin array. Add 1 to end_bin for Python's exclusive slicing.
        range_slice = ber_per_bin[start_bin : end_bin + 1]
        avg_ber_in_range = np.mean(range_slice)
        print(f"Average BER for bins {start_bin}-{end_bin} (inclusive): {avg_ber_in_range:.2e}")
    else:
        print(f"Warning: Cannot calculate BER for range {start_bin}-{end_bin} as it exceeds the number of subcarriers ({num_subcarriers}).")

    # --- Plotting BER per Bin ---
    plt.figure(figsize=(12, 5))
    subcarrier_indices = np.arange(num_subcarriers)
    
    plt.bar(subcarrier_indices, ber_per_bin, width=1.0, label=f'BER (Avg over {num_blocks} blocks)')
    
    plt.title('Average Bit Error Rate per Subcarrier Bin')
    plt.xlabel('Subcarrier Index (Bin)')
    plt.ylabel('Average Bit Error Rate')
    plt.grid(True, which='both', linestyle=':')
    plt.yscale('log') # BER is best viewed on a logarithmic scale
    plt.ylim(bottom=1e-5, top=1.0) # Set sensible Y-axis limits
    plt.xlim(0, num_subcarriers - 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_noise_variance_robust(received_symbols_slice):
    """
    Calculates noise variance from a slice of QPSK symbols by first
    normalizing them to have the correct ideal average power.
    """
    if received_symbols_slice.size == 0:
        return 1.0  # Return a default high noise value if slice is empty

    P_ideal = 2.0  # Ideal average power of a (+-1, +-1j) QPSK constellation
    p_received = np.mean(np.abs(received_symbols_slice)**2)
    scaling_factor = np.sqrt(P_ideal / (p_received + 1e-12))
    normalized_symbols = received_symbols_slice * scaling_factor
    
    hard_decisions = np.sign(normalized_symbols.real) + 1j * np.sign(normalized_symbols.imag)
    residuals = normalized_symbols - hard_decisions
    sigma2_est = np.mean(np.abs(residuals)**2)
    
    return sigma2_est

def qpsk_to_bits(sym_array):
    """ Converts an array of QPSK symbols to an interleaved bit stream. """
    bI = (sym_array.real < 0).astype(int)
    bQ = (sym_array.imag < 0).astype(int)
    bits = np.zeros(2 * len(sym_array), dtype=int)
    bits[0::2] = bI
    bits[1::2] = bQ
    return bits

def ldpc_decode_cw(llr_vec: np.ndarray) -> np.ndarray:
    soft, _ = my_ldpc.decode(llr_vec)
    return (soft < 0).astype(np.uint8)[:LDPC_K] 

if __name__ == "__main__":
#------------------------------initialization-------------------------------------------
    # record_audio(20*FS)
    # SAMPLE_RATE, recording = read('rx_recording.wav')
    recording = output["waveform"]
    chirp_up   = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

#--------------------------------main sequence------------------------------------------
    payload, start_payload, end_payload, last_valid_block_index = start_end_synchronise(recording, chirp_up, chirp_down)
    time_blocks = sync_chopper(payload, last_valid_block_index)
    useful_freq_blocks  = freq_domain(time_blocks)
    h_estimated_array = channel_estimation(useful_freq_blocks, np.load(PILOT_NPY), "zf")
    equalised_all_blocks = equalise(useful_freq_blocks, h_estimated_array)
    corrected_data_blocks = phase_error_correction(equalised_all_blocks, np.load(PILOT_NPY))
    print("Shape of equalised data blocks: ", corrected_data_blocks.shape)

#-------------------------------LDPC shizzle---------------------------------------------
    decoded_info_bits = []
    for blk_idx, blk_syms in enumerate(corrected_data_blocks):
        print("Enumerate shape: ", blk_idx, blk_syms.shape)
        # ----- build one 3888-LLR vector ---------------------------
        llr = np.empty(2 * LDPC_N, np.float64)
        sigma2_est = calculate_noise_variance_robust(blk_syms)
        gain = min(2.0 / sigma2_est, 10.0)
        for k, s in enumerate(blk_syms[200:200 + LDPC_N]):  # <- blk_syms, not eq_syms
            llr[2 * k] = gain * s.real
            llr[2 * k + 1] = gain * s.imag
        print("Shape of LLR pre-clipped (per 4095 freq sym): ", np.array(llr).shape)
        np.clip(llr, -LLR_MAX, LLR_MAX, out=llr)
        print("Shape of LLR post-clipped (per 4095 freq sym): ", llr.shape)

        # ----- de-interleave exactly like the mapper ---------------
        #pairs = llr.reshape(-1, 2)  # 1944×2
        #llr_cw1 = pairs[:, 0].ravel()  # I bits
        #llr_cw2 = pairs[:, 1].ravel()  # Q bits
        pairs = llr.reshape(-1, 2)  # shape (1 944, 2)
        print("LLR bit pairs shape (per 4095 freq sym): ", pairs.shape)
        SYMS_PER_CW = LDPC_N // 2  # 972

        llr_cw1 = np.ascontiguousarray(pairs[:SYMS_PER_CW].ravel())
        llr_cw2 = np.ascontiguousarray(pairs[SYMS_PER_CW:].ravel())
        print("Shape of LDPC block to be decoded: ", llr_cw1.shape, llr_cw2.shape)

        """
        # ----- LDPC decode (ldpc_jossy returns hard bits) ----------
        #cw1_hat, _ = my_ldpc.decode(llr_cw1)  # 0/1 ints, length 1944
        #cw2_hat, _ = my_ldpc.decode(llr_cw2)

        #u1_hat = cw1_hat[:LDPC_K].astype(np.int8)  # 972 information bits
        #u2_hat = cw2_hat[:LDPC_K].astype(np.int8)
        """

        u1_hat = ldpc_decode_cw(llr_cw1)
        u2_hat = ldpc_decode_cw(llr_cw2)
        print("Shape of LDPC block decoded: ", u1_hat.shape, u2_hat.shape)

        # ----- reference TX info bits (for testing only) ------------------------------
        """
        print("Length of payload info bits: ", len(output["payload_info_bits"])) # Test point
        print("This should be 1944 (LDPC_N): ", LDPC_N)
        print("Two truncate points: ", blk_idx * LDPC_N, (blk_idx + 1) * LDPC_N)
        tx_pair = output["payload_info_bits"][blk_idx * LDPC_N: (blk_idx + 1) * LDPC_N].astype(np.int8)
        print("Length of payload info bits selected: ", len(tx_pair))
        if (blk_idx + 1) * LDPC_N >= len(output["payload_info_bits"]):
            if len(tx_pair) == 1944:
                tx_u1, tx_u2 = np.split(tx_pair, 2)
            elif len(tx_pair) == 972:
                print("Padding required")
                tx_u1 = tx_pair
                tx_u2 = u2_hat
                print("Last LDPC block does not matter, padded perfectly")
            else:
                pass
                raise Exception("The decoded LDPC block has the wrong length")
        else:
            tx_u1, tx_u2 = np.split(tx_pair, 2)

        #err1 = np.count_nonzero(u1_hat ^ tx_u1)
        err1 = np.count_nonzero(u1_hat.astype(np.uint8) ^ tx_u1.astype(np.uint8))
        #err2 = np.count_nonzero(u2_hat ^ tx_u2)
        err2 = np.count_nonzero(u2_hat.astype(np.uint8) ^ tx_u2.astype(np.uint8))

        print(f"[blk {blk_idx}] POST-LDPC CW1: {err1}/{LDPC_K}  BER={err1 / LDPC_K:.2e}")
        print(f"[blk {blk_idx}] POST-LDPC CW2: {err2}/{LDPC_K}  BER={err2 / LDPC_K:.2e}")
        """

        decoded_info_bits.append(u1_hat)
        decoded_info_bits.append(u2_hat)


    # 11) Print out first few recovered bits of the first block for verification
    #first_rec1, first_rec2 = decoded_info_bits[0]
    #print("Decoded info bits (first OFDM symbol):")
    #print(" Codeword 1 (first 16 bits):", first_rec1[:16], "…")
    #print(" Codeword 2 (first 16 bits):", first_rec2[:16], "…")

    print("Decoder output shape: ", np.array(decoded_info_bits).shape)
    print(len(decoded_info_bits))

    received_binary = np.array(decoded_info_bits).flatten()
    print("Shape of binary sequence: ", received_binary.shape)

    """
    info1_0, info2_0 = decoded_info_bits[0]
    print("Decoded info bits (first OFDM symbol):")
    print(" Codeword 1 (first 16 bits):", info1_0[:16], "…")
    print(" Codeword 2 (first 16 bits):", info2_0[:16], "…")
    """

    # Save the decoded bits / ASCII
    np.save("received_bin", received_binary)

    print(f'Received {received_binary.size} bits, first 64 bits:\n', received_binary[:64])

    byte_array = np.packbits(received_binary)

    file_name_terminate = np.where(byte_array == 0)[0][0]
    file_name = byte_array[:file_name_terminate].tobytes().decode("utf-8")

    file_size_terminate = np.where(byte_array == 0)[0][1]
    file_size = int(byte_array[file_name_terminate + 1:file_size_terminate].tobytes().decode("utf-8"))

    file_content = byte_array[(file_size_terminate + 1):(file_size_terminate + int(file_size) + 1)]

    print("File name: ", file_name, ", File size: ", file_size, ", File content: ", file_content.tobytes())

    with open("received.txt", "wb") as f:
        f.write(file_content.tobytes())

    # 12) Optionally compare TX vs RX in time‐domain & show spectrum
    compare_tx_rx(recording, start_payload, end_payload)
    spectrum_plot(recording)