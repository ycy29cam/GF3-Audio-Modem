import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from scipy.io.wavfile import read
from Tx_LPDC import generate_chirp, WAV_TX, output, Q_COL
import Tx_LPDC as tx
import ldpc_jossy

# look into using GPU for hardware acceleration of convolution, e.g. using CuPy for faster processing

# ------------------------------------------------
#   1.  General parameters (unchanged)
# ------------------------------------------------
FS = tx.FS
FFT_LEN = tx.FFT_LEN
CP_LEN = tx.CP_LEN
CHIRP_LEN_S = tx.CHIRP_LEN_S
SILENCE_LEN_S = tx.SILENCE_LEN_S
F0, F1 = tx.F0, tx.F1
TX_REPS = tx.TX_REPS
WAV_TX = WAV_TX
WAV_RX = 'rx_recording.wav'
PILOT_NPY = 'pilot_symbols.npy'
COLMAP_NPY = 'colour_map.npy'
CHAN_NPY = 'channel_estimate.npy'

CHIRP_ATTEN = tx.CHIRP_ATTEN
TARGET_PEAK = tx.TARGET_PEAK
LENGTH_TOL = tx.LENGTH_TOL

# LDPC Param
LDPC_Z = 81
LDPC_N = 24 * LDPC_Z
LDPC_K = LDPC_N // 2

# Instantiate the same “802.11n” LDPC code that you used in Tx:
my_ldpc = ldpc_jossy.code(standard='802.11n', rate='1/2', z=LDPC_Z)

# Sanity check: each OFDM symbol carries two codewords,
# each codeword→972 QPSK symbols, so total=1944 symbols ≤ 4095-available
assert 2 * (LDPC_N // 2) <= (FFT_LEN // 2 - 1)


# ------------------------------------------------
#   2.  Input/Output
# ------------------------------------------------
def record_audio(expected_len: int, fs: int = FS) -> np.ndarray:
    print(f"Recording ≈{expected_len / fs:.2f} s …")
    rec = sd.rec(expected_len, samplerate=fs, channels=1,
                 dtype='float32').squeeze()  # removes extra unused dimension
    sd.wait()
    sf.write(WAV_RX, rec, fs)
    return rec


def load_wav(path):
    data, sr = sf.read(path, always_2d=False)  # sr is sample-rate of recording
    assert sr == FS, "sample-rate mismatch"
    return data.astype(np.float32)


# ------------------------------------------------
#   3.  Synchronisation
# ------------------------------------------------
def start_end_synchronise(rx: np.ndarray,
                          chirp_up: np.ndarray,
                          chirp_down: np.ndarray) -> tuple[
    np.ndarray, int, int]:  # colon tells you what type the function takes, and arrow tells you what the function returns
    corr_up = signal.correlate(rx, chirp_up, mode='valid')
    peak_up = np.argmax(corr_up)
    corr_down = signal.correlate(rx, chirp_down, mode='valid')
    plt.plot(corr_up, label='up-chirp correlation')
    plt.plot(corr_down, label='down-chirp correlation')
    plt.plot(rx * 10000, label='received signal', alpha=0.5, )
    search_from = peak_up + len(chirp_up)  # search for down-chirp after up-chirp
    peak_down = np.where(corr_down > 0.5 * corr_down.max())[0]  # formatting, 2D - 1D but no information loss
    peak_down = peak_down[peak_down > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload = peak_down
    print("start_payload:", start_payload, "end_payload:", end_payload)
    payload = rx[start_payload:end_payload]
    exp = output["total_ofdm_length"]

# --------------------------------------        
    if len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp - len(payload)))
    else:
        payload = payload[:exp]
    # sf.write("chopped_payload_sound.wav", payload, FS)
    return payload, start_payload, end_payload
# --------------------------------------        
        
    # If RuntimeError is occuring (because LENGTH_TOL is being exceeded) then this option is more reliable, as
    # it simply zero-pads never throws an exception if len(payload) is “too short.” Instead, it just pads.
    # if len(payload) < exp:
    #     payload = np.pad(payload, (0, exp - len(payload)))
    # elif len(payload) > exp:
    #     payload = payload[:exp]


# ------------------------------------------------
#   4.  OFDM helpers
# ------------------------------------------------

def time_OFDM_chopper(payload, block_length_time=output["ofdm_block_len_with_cp"]):
    time_blocks = []
    print(len(payload))
    print((output["no_of_payload_blocks"]))
    if len(payload) % block_length_time != 0:
        raise ValueError("Payload length is not a multiple of block length.")
    for i in range(len(payload) // block_length_time - 0):
        i = i  # allows for future non prefixed code at the beginning of the payload
        time_blocks.append(payload[i * block_length_time:(i + 1) * block_length_time])
        time_blocks[-1] = time_blocks[-1][CP_LEN:]
    return np.array(time_blocks)


def freq_domain(blocks_td: np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN // 2]


# ------------------------------------------------
#   5.  Channel estimation
# ------------------------------------------------

def channel_estimation(blocks: np.ndarray,
                       pilot_symbol: np.ndarray,
                       method: str = 'zf',
                       noise_var: float = 1e-4) -> np.ndarray:
    eps = 1e-12
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block
    N = len(payload_type_list)
    estimates = [None] * N

    def estimate_pilot_channel(freq_block: np.ndarray) -> np.ndarray:
        return freq_block / (pilot_symbol + eps)

    # 1. fill in pilot estimates
    for i, t in enumerate(payload_type_list):
        if t == 'pilot':
            estimates[i] = estimate_pilot_channel(blocks[i])

    # 2. for each data block, average nearest pilot estimates
    for i, t in enumerate(payload_type_list):
        if t == 'data':
            prev_idx = next((j for j in range(i - 1, -1, -1) if payload_type_list[j] == 'pilot'), None)
            next_idx = next((j for j in range(i + 1, N) if payload_type_list[j] == 'pilot'), None)

            to_avg = []
            if prev_idx is not None:
                to_avg.append(estimates[prev_idx])
            if next_idx is not None:
                to_avg.append(estimates[next_idx])

            estimates[i] = np.mean(np.stack(to_avg, axis=0), axis=0) if to_avg else np.zeros_like(blocks[i])

    H_est = np.stack(estimates, axis=0)
    np.save(CHAN_NPY, H_est)
    return H_est


def reconstruct_data_blocks(useful_frequency_blocks, H_est_array):
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block
    assert len(useful_frequency_blocks) == len(payload_type_list), "Mismatch between blocks and payload types"
    assert len(H_est_array) == len(payload_type_list), "Mismatch between channel estimates and payload types"
    data_blocks = np.array(
        [useful_frequency_blocks[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    data_H_est_array = np.array([H_est_array[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    decoded_datablocks = data_blocks / data_H_est_array  # element-wise division
    return decoded_datablocks


def equalise(rx_fd, H):
    return rx_fd / H


# ------------------------------------------------
#   6. Visualisation helpers
# ------------------------------------------------
def plot_channel(H: np.ndarray):
    """Visualise magnitude & phase of the estimated channel."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    ax[0].plot(20 * np.log10(np.abs(H) + 1e-12))
    ax[0].set_ylabel("|H| [dB]")
    ax[0].set_title("Estimated channel magnitude / phase")
    ax[1].plot(np.angle(H))
    ax[1].set_ylabel("∠H [rad]")
    ax[1].set_xlabel("sub-carrier")
    plt.tight_layout();
    plt.show()


def compare_tx_rx(rx: np.ndarray, start_rx_payload: int, end_rx_payload_boundary: int, tx_path: str = WAV_TX):
    """
    Compares the extracted RX payload against the corresponding TX payload.
    start_rx_payload: Index in rx where the payload begins (after up-chirp).
    end_rx_payload_boundary: Index in rx where the down-chirp begins (payload ends before this).
    """
    tx_sig = load_wav(tx_path)

    tx_leading_silence = output["leading_silence_samples"]
    tx_chirp_len = output["chirp_samples"]  # Length of the core chirp signal
    tx_start_of_payload = tx_leading_silence + tx_chirp_len
    payload_length_to_compare = output["total_ofdm_length"]  # This is 'exp'

    # Define segments for comparison (both should be the OFDM payload part)
    tx_seg_end = tx_start_of_payload + payload_length_to_compare
    rx_seg_end = start_rx_payload + payload_length_to_compare

    # Boundary checks
    if tx_start_of_payload >= tx_seg_end or tx_seg_end > len(tx_sig):
        print("Warning: TX segment for comparison is invalid or out of bounds.")
        return
    if start_rx_payload >= rx_seg_end or rx_seg_end > len(rx):
        print("Warning: RX segment for comparison is invalid or out of bounds.")
        return

    tx_payload_seg = tx_sig[tx_start_of_payload: tx_seg_end]
    rx_payload_seg = rx[start_rx_payload: rx_seg_end]

    # Normalize for plotting
    m_peak = np.max(np.abs(rx_payload_seg)) if rx_payload_seg.size > 0 else 0
    n_peak = np.max(np.abs(tx_payload_seg)) if tx_payload_seg.size > 0 else 0

    tx_norm = tx_payload_seg / n_peak
    rx_norm = rx_payload_seg / m_peak

    plt.figure(figsize=(10, 3))
    plt.plot(tx_norm, label='TX Payload (norm.)', lw=.8)
    plt.plot(rx_norm, label='RX Payload (norm.)', lw=.6, alpha=.7)
    plt.title("TX vs RX OFDM Payload (aligned)")
    plt.xlabel("sample in payload")
    plt.ylabel("normalised amplitude")
    plt.legend();
    plt.tight_layout();
    plt.show()


# ------------------------------------------------
#   7.  Spectrum & constellation
# ------------------------------------------------
def _means_by_colour(z_flat, colours_flat):
    ucols = np.unique(colours_flat)  # pulls out an array of the unique(in this case 4 colours) colours used
    means = {c: np.mean(z_flat[colours_flat == c]) for c in ucols}  # finds the mean of each colour in the constellation
    return means  # a dictionary of colour:mean pairs


def spectrum_plot(sig: np.ndarray, fs: int = FS):
    f, Pxx = signal.welch(sig, fs, nperseg=4096)
    plt.figure();
    plt.semilogy(f, Pxx)
    plt.title("Received PSD");
    plt.xlabel("Hz");
    plt.ylabel("PSD [V²/Hz]")
    plt.tight_layout();
    plt.show()


def plot_equalised_blocks(equalised_data_blocks: np.ndarray, sequenced_data_blocks: np.ndarray):
    """
    Plot equalised constellation blocks with correct colouring and legend.

    Args:
        equalised_data_blocks (np.ndarray): Equalised data blocks (N_blocks, N_subcarriers)
        tx_blocks (np.ndarray): Corresponding TX data symbols (ideal, same shape)
    """
    assert equalised_data_blocks.shape == sequenced_data_blocks.shape, "Shape mismatch between TX and RX blocks"

    # Flatten both
    eq_flat = equalised_data_blocks.ravel()
    tx_flat = sequenced_data_blocks.ravel()

    # Rebuild colour map based on TX ideal symbols
    # Reverse map: symbol -> colour
    sym_to_colour = {1 + 1j: Q_COL[(0, 0)],
                     1 - 1j: Q_COL[(0, 1)],
                     -1 - 1j: Q_COL[(1, 1)],
                     -1 + 1j: Q_COL[(1, 0)]}

    tx_colours = np.array([sym_to_colour.get(complex(round(s.real), round(s.imag)), 'k')
                           for s in tx_flat])

    # Normalise constellation energy (unit average power)
    eq_flat /= np.sqrt(np.mean(np.abs(eq_flat) ** 2) + 1e-12)

    # --- Plotting ---
    plt.figure();
    plt.axhline(0, c='k');
    plt.axvline(0, c='k')
    plt.scatter(eq_flat.real, eq_flat.imag,
                c=tx_colours, s=12, alpha=.85, edgecolors='none')

    # --- Means ---
    unique_colours = np.unique(tx_colours)
    for c in unique_colours:
        mask = tx_colours == c
        mean = np.mean(eq_flat[mask])
        plt.plot(mean.real, mean.imag, 'kx')
        plt.text(mean.real, mean.imag, f"{mean.real:+.2f}+{mean.imag:+.2f}j",
                 fontsize=7, ha='left', va='bottom')

    # --- Legend ---
    bits_to_sym = {'00': '1+1j', '01': '1-1j', '11': '-1-1j', '10': '-1+1j'}
    legend_handles = []
    for bits, colour in Q_COL.items():
        bit_str = ''.join(map(str, bits))
        label = bits_to_sym.get(bit_str, bit_str)
        legend_handles.append(Patch(facecolor=colour, label=label))
    plt.legend(handles=legend_handles, loc='upper right', fontsize='small')

    plt.title("Equalised Constellation (coloured by TX symbols)")
    plt.xlabel("I");
    plt.ylabel("Q")
    plt.gca().set_aspect('equal');
    plt.tight_layout();
    plt.show()


if __name__ == "__main__":
    # 1) Optionally record in real time, or load a pre-existing file
    # record_audio(480000)
    SAMPLE_RATE, recording = read('rx_recording.wav')

    # 2) Load TX waveform (for synchronisation)
    SAMPLE_RATE, transmission = read("tx_sequence.wav")

    #-----------------------------------------#
    ## NO MIC-SPEAKER TEST:
    #recording = output["waveform"]
    #SAMPLE_RATE = FS
    #_, transmission = read("tx_sequence.wav")
    #-----------------------------------------#


    chirp_up = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

    # 3) Synchronise and extract the OFDM payload
    payload, start_payload, end_payload = start_end_synchronise(recording, chirp_up, chirp_down)

    # 4) Chop into time-domain OFDM blocks (remove CP)
    time_blocks = time_OFDM_chopper(payload)

    # 5) FFT → get frequency-domain data for each block
    useful_freq_blocks = freq_domain(time_blocks)

    # 6) Channel estimation (pilot → H_est for each block)
    pilot_symbols = np.load(PILOT_NPY)  # shape = (4095,)
    h_estimated_array = channel_estimation(useful_freq_blocks, pilot_symbols, method="zf")

    # 7) Equalise & reconstruct data subcarriers
    reconstructed_data = reconstruct_data_blocks(useful_freq_blocks, h_estimated_array)
    # `reconstructed_data` has shape (N_data_blocks=TX_REPS, 4095)

    # ── Perfect I/Q normalization by dividing axes separately ──
    raw0 = reconstructed_data[0, :LDPC_N]  # first 1944 “data” symbols of OFDM block 0
    tx0 = output["payload_data_blocks"][0]  # Tx’s ideal QPSK symbols (each exactly ±1±1j)

    # Identify which indices were “+1 + 1j” at TX:
    is_11 = np.logical_and(np.real(tx0) > 0, np.imag(tx0) > 0)
    # If no “+1 + 1j” symbols in that block, pick any other nonempty corner:
    is_1m1 = np.logical_and(np.real(tx0) > 0, np.imag(tx0) < 0)
    is_m1m1 = np.logical_and(np.real(tx0) < 0, np.imag(tx0) < 0)
    is_m11 = np.logical_and(np.real(tx0) < 0, np.imag(tx0) > 0)

    if np.any(is_11):
        ref_mask = is_11
    elif np.any(is_1m1):
        ref_mask = is_1m1
    elif np.any(is_m1m1):
        ref_mask = is_m1m1
    else:
        ref_mask = is_m11

    # Compute mean real/I and mean imag/Q for those positions:
    mean_rx_I = np.mean(np.real(raw0[ref_mask]))  # e.g. ≈0.71 in loopback
    mean_rx_Q = np.mean(np.imag(raw0[ref_mask]))  # e.g. ≈0.71 in loopback

    # Divide EACH symbol’s real‐component by mean_rx_I,
    # and each symbol’s imag‐component by mean_rx_Q:
    reconstructed_data = (
            (reconstructed_data.real / (mean_rx_I + 1e-12))
            + 1j * (reconstructed_data.imag / (mean_rx_Q + 1e-12))
    )

    print("First‐block EQ symbols (first 8 of raw0, after normalization):")
    print(np.round(reconstructed_data[0, :8], 3))

    # 8) Plot one constellation for sanity (optional)
    plot_equalised_blocks(
        reconstructed_data[0][:LDPC_N],  # take only the first 1944 symbols
        output["payload_data_blocks"][4]  # this is already length=1944
    )
        # after LDPC‐packing, each OFDM “data” block now has 4095 equalized QPSK symbols (including
        # the zero‐padding), whereas output["payload_data_blocks"][i] is only the first 1944 symbols
        # (the actual LDPC‐coded payload). Ie. Only consider the first 1944 symbols and ignore
        # the remaining 2151 that symbols were zero‐padded at Tx and don’t correspond to any actual data.


    # 9) Noise‐variance estimation via “residuals” on the first 1944 symbols of each block
    def _qpsk_hard_decision(sym):
        b_i = 0 if sym.real > 0 else 1
        b_q = 0 if sym.imag > 0 else 1
        return (1 - 2 * b_i) + 1j * (1 - 2 * b_q)


    residuals = []
    for blk in reconstructed_data:
        eq_symbols = blk[:LDPC_N]  # only first 1944 were actual data
        hard_pts = np.array([_qpsk_hard_decision(s) for s in eq_symbols])
        residuals.append(eq_symbols - hard_pts)
    residuals = np.concatenate(residuals)  # shape = (TX_REPS * 1944,)
    sigma2_est = np.mean(np.abs(residuals) ** 2)
    print(f"Estimated noise variance σ² = {sigma2_est:.3e}")

    # 10) LDPC decode: form LLRs, split into two codewords, decode each
    decoded_info_bits = []  # will hold (rec1, rec2) for each OFDM block
    for blk_idx, blk_symbols in enumerate(reconstructed_data):
        eq_payload = blk_symbols[:LDPC_N]  # length = 1944

        # Build LLR vector length=2×1944 = 3888
        llr = np.zeros(2 * LDPC_N, dtype=np.float32)
        for j, sym in enumerate(eq_payload):
            llr[2 * j] = (2.0 / sigma2_est) * sym.real
            llr[2 * j + 1] = (2.0 / sigma2_est) * sym.imag

        # Split into two chunks of length=1944 each
        llr_cw1 = llr[:LDPC_N]
        llr_cw2 = llr[LDPC_N: 2 * LDPC_N]

        # Decode both codewords
        soft1, iters1 = my_ldpc.decode(llr_cw1)  # soft1 is length=972
        hard1 = (soft1 > 0).astype(np.int8)  # threshold at 0 → 0/1 bits

        print("llr_cw2.shape  =", llr_cw2.shape)  # Should be exactly (1944,)
        print("LDPC_N        =", LDPC_N)  # Should print 1944
        print("any NaN in llr_cw2? ", np.isnan(llr_cw2).any())
        print("any Inf in llr_cw2? ", np.isinf(llr_cw2).any())
        print("llr_cw2 min/max:", np.nanmin(llr_cw2), np.nanmax(llr_cw2))

        soft2, iters2 = my_ldpc.decode(llr_cw2)
        hard2 = (soft2 > 0).astype(np.int8)
        decoded_info_bits.append((hard1, hard2))

        #rec1 = my_ldpc.decode(llr_cw1)  # returns array of length=972 (info bits)
        #rec2 = my_ldpc.decode(llr_cw2)  # same
        #decoded_info_bits.append((rec1, rec2))

    # 11) Print out first few recovered bits of the first block for verification
    first_rec1, first_rec2 = decoded_info_bits[0]
    print("Decoded info bits (first OFDM symbol):")
    print(" Codeword 1 (first 16 bits):", first_rec1[:16], "…")
    print(" Codeword 2 (first 16 bits):", first_rec2[:16], "…")

    # 12) Optionally compare TX vs RX in time‐domain & show spectrum
    compare_tx_rx(recording, start_payload, end_payload)
    spectrum_plot(recording)
