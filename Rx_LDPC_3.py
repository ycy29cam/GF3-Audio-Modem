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
#LLR_MAX = 50.0
LLR_MAX = 200.0
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"


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
                chirp_down: np.ndarray) -> tuple[np.ndarray, int, int, int]:

    corr_up   = signal.correlate(rx, chirp_up, mode='valid')
    peak_up   = np.argmax(corr_up)
    corr_down = signal.correlate(rx, chirp_down, mode='valid')

    plt.plot(corr_up, label='up-chirp correlation')
    plt.plot(corr_down, label='down-chirp correlation')
    plt.plot(rx * 5000, label='received signal', alpha=0.5)

    search_from = peak_up + len(chirp_up)
    peak_down_locs = np.where(corr_down > 0.5 * corr_down.max())[0]
    peak_down = peak_down_locs[peak_down_locs > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down
    print("start_payload:", start_payload, "end_payload:", end_payload)

    payload = rx[start_payload:end_payload]

    block_len = 10240
    n_blocks = int(round(len(payload) / 10240))
    padded_len = n_blocks * block_len
    print("unpadded vs padded difference is: ", abs(padded_len - len(payload)))
    valid_blocks = n_blocks

    if len(payload) < padded_len:
        payload = np.pad(payload, (0, padded_len - len(payload)))
    else:
        payload = payload[:padded_len]

    if n_blocks % 5 != 0:
        pad_blocks = 5 - (n_blocks % 5)
        payload = np.pad(payload, (0, pad_blocks * block_len))
        n_blocks += pad_blocks

    last_valid_block_index = valid_blocks - 1

    return payload, start_payload, end_payload, last_valid_block_index


# ------------------------------------------------
#   4.  OFDM helpers
# ------------------------------------------------
"""
def sync_chopper(payload, start_payload, end_payload, rx,
                 block_length_time=output["ofdm_block_len_with_cp"]):
#    
#    Returns:
#      pilot_td_blocks: array shape (TX_REPS, FFT_LEN) — CP-stripped pilot symbols
#      data_td_blocks:  array shape (TX_REPS*4, FFT_LEN) — CP-stripped data symbols
#    
    # Preallocate lists
    pilot_blocks = []
    data_blocks  = []

    # Load the time-domain pilot waveforms (no CP) you saved in Tx
    time_pilot_blocks_no_cp = np.load(PILOT_TIME_NO_CP_NPY, allow_pickle=True)

    # Window initial bounds: half a block before payload start, 1.5 blocks after
    x = int(start_payload - block_length_time/2)
    y = int(start_payload + block_length_time * 1.5)

    for i in range(TX_REPS):
        # Correlate to find exact pilot start
        window     = rx[x:y]
        pilot_td   = time_pilot_blocks_no_cp[i]
        corr       = signal.correlate(window, pilot_td, mode="valid")
        sync_start = x + np.argmax(corr)

        #expected = start_payload + i * 5 * block_length_time
        expected = start_payload + i * (FFT_LEN + CP_LEN) + CP_LEN

        if abs(sync_start - expected) > LENGTH_TOL:
            print(f" Desync on pilot {i}: got {sync_start}, expected {expected}")
            sync_start = expected

        # 1) Extract CP-stripped pilot
        #p0 = sync_start + CP_LEN
        #pilot_blocks.append(rx[p0 : p0 + FFT_LEN])
        p0 = sync_start
        pilot_blocks.append(rx[p0: p0 + FFT_LEN + CP_LEN])

        # 2) Extract the 4 full (CP+FFT) data blocks
        d0 = p0 + CP_LEN + FFT_LEN  # jump past pilot's CP+FFTs
        for _ in range(4):
            data_blocks.append(rx[d0: d0 + FFT_LEN + CP_LEN])
            d0 += FFT_LEN + CP_LEN  # stride past each full block


        # Advance the search window for next rep
        x += 5 * block_length_time
        y += 5 * block_length_time

    # Convert to numpy arrays
    pilot_td_blocks = np.stack(pilot_blocks)  # shape = (TX_REPS, FFT_LEN)
    data_td_blocks  = np.stack(data_blocks)   # shape = (TX_REPS*4, FFT_LEN)
    return pilot_td_blocks, data_td_blocks
"""


def sync_chopper(rx,
                 start_payload,
                 num_pilots,
                 fft_len,
                 cp_len,
                 length_tol=LENGTH_TOL):
    
#    Args:
#      rx            : 1D received waveform
#      start_payload : index where payload begins
#      num_pilots    : TX_REPS
#      fft_len       : FFT_LEN
#      cp_len        : CP_LEN
#    Returns:
#      pilot_td_blocks: np.ndarray (num_pilots, fft_len)
#      data_td_blocks : np.ndarray (num_pilots*4, fft_len)
 
    pilot_blocks = []
    data_blocks  = []
    time_pilots  = np.load(PILOT_TIME_NO_CP_NPY, allow_pickle=True)

    # initial search window
    blk_time = fft_len + cp_len
    x = int(start_payload - blk_time/2)
    y = int(start_payload + blk_time * 1.5)

    for i in range(num_pilots):
        window   = rx[x:y]
        pilot_td = time_pilots[i]
        corr     = signal.correlate(window, pilot_td, mode="valid")
        sync0    = x + np.argmax(corr)
        #expected = start_payload + i * 5 * blk_time
        expected = start_payload + i * 5 * blk_time + cp_len

        #if abs(sync0 - expected) > length_tol:
        #    print(f" Desync on pilot {i}: got {sync0}, expected {expected}")
        #    sync0 = expected

        if abs(sync0 - expected) > length_tol:
            print(f"Desync {i}: diff={sync0 - expected}")

        """
        # strip CP and collect
        #p0 = sync0 + cp_len
        p0 = sync0
        pilot_blocks.append(rx[p0 : p0 + fft_len])

        # then grab the four data blocks
        d0 = p0 + fft_len
        for _ in range(4):
            data_blocks.append(rx[d0 : d0 + fft_len])
            d0 += fft_len
        """
        # --- pilots ---
        p0 = sync0 - cp_len  # <-- start at the CP
        pilot_blocks.append(rx[p0: p0 + blk_time])  # blk_time = cp_len + fft_len

        # --- data blocks ---
        d0 = p0 + blk_time
        for _ in range(4):
            data_blocks.append(rx[d0: d0 + blk_time])
            d0 += blk_time

        x += 5 * blk_time
        y += 5 * blk_time

    return np.stack(pilot_blocks), np.stack(data_blocks)


#def freq_domain(blocks_td: np.ndarray) -> np.ndarray:
#    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN // 2]

def freq_domain(blocks_td: np.ndarray) -> np.ndarray:
    """
    blocks_td: shape (n_blocks, FFT_LEN+CP_LEN)
    Returns the positive half‐spectrum (bins 1 … FFT_LEN/2−1)
    after stripping off the cyclic prefix.
    """
    # 1) drop the CP
    no_cp = blocks_td[:, CP_LEN:]               # shape = (n_blocks, FFT_LEN)

    # 2) FFT and take bins 1 … FFT_LEN/2−1
    F = fft.fft(no_cp, axis=1)                  # shape = (n_blocks, FFT_LEN)

    return F[:, 1 : FFT_LEN // 2]               # shape = (n_blocks, FFT_LEN/2−1)


# ------------------------------------------------
#   5.  Channel estimation
# ------------------------------------------------

"""
def channel_estimation(blocks: np.ndarray,
                       pilot_symbols: np.ndarray,
                       method: str = 'zf',
                       noise_var: float = 1e-4) -> np.ndarray:
    eps = 1e-12
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block
    N = len(payload_type_list)
    estimates = [None] * N

    def estimate_pilot_channel(freq_block: np.ndarray) -> np.ndarray:
        return freq_block / (pilot_symbols[i] + eps)

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
"""

def channel_estimation(pilot_blocks: np.ndarray,
                       pilot_symbols: np.ndarray,
                       method: str = 'zf') -> np.ndarray:
    """
    Zero-forcing channel estimate on exactly the pilot OFDM symbols.
    pilot_blocks   : (TX_REPS, N_subcarriers) freq-domain received pilots
    pilot_symbols  : (TX_REPS, N_subcarriers) known TX pilot symbols
    returns H_pilots: same shape, each H_pilots[i] = pilot_blocks[i]/pilot_symbols[i]
    """
    eps = 1e-12
    assert pilot_blocks.shape == pilot_symbols.shape, \
        f"pilot_blocks {pilot_blocks.shape} vs pilot_symbols {pilot_symbols.shape}"

    # Elementwise division → one H-estimate per pilot
    H_pilots = pilot_blocks / (pilot_symbols + eps)
    np.save(CHAN_NPY, H_pilots)
    return H_pilots


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

    equalised_data_blocks = np.array(equalised_data_blocks, dtype=np.complex64)
    sequenced_data_blocks = np.array(sequenced_data_blocks,  dtype=np.complex64)

    assert equalised_data_blocks.shape == sequenced_data_blocks.shape, "Shape mismatch between TX and RX blocks"

    #eq = np.array(equalised_data_blocks, dtype=np.complex64)

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
    #record_audio(480000)
    #SAMPLE_RATE, recording = read('rx_recording.wav')

    recording, SAMPLE_RATE = sf.read("tx_sequence.wav")
    if recording.ndim > 1:
        recording = recording.mean(axis=1)

    # 2) Load TX waveform (for synchronisation)
    #SAMPLE_RATE, transmission = sf.read("tx_sequence.wav")

    #-----------------------------------------#
    ## NO MIC-SPEAKER TEST:
    #recording = output["waveform"]
    #SAMPLE_RATE = FS
    #_, transmission = read("tx_sequence.wav")
    #-----------------------------------------#


    chirp_up = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

    # 3) Synchronise and extract the OFDM payload
    #payload, start_payload, end_payload = start_end_synchronise(recording, chirp_up, chirp_down)
    payload, start_payload, end_payload, last_valid_block_index = start_end_synchronise(recording, chirp_up, chirp_down)
    print("SYNC PAYLOAD START at", start_payload)

    # 4) Chop into time-domain OFDM blocks (remove CP)
    #pilot_td_blocks, data_td_blocks = sync_chopper(payload, start_payload, end_payload, recording)
    pilot_td_blocks, data_td_blocks = sync_chopper(
        recording,  # rx waveform
        start_payload,  # payload start index
        TX_REPS,  # number of pilots (e.g. 5)
        FFT_LEN,  # your FFT length
        CP_LEN  # your cyclic‐prefix length
    )
    assert pilot_td_blocks.shape[1] == FFT_LEN + CP_LEN, \
        "pilot length mismatch – CP missing?"

    # 5) FFT → get frequency-domain data for each block
    #useful_freq_blocks = freq_domain(time_blocks)
    pilot_freq_blocks = freq_domain(pilot_td_blocks)
    data_freq_blocks = freq_domain(data_td_blocks)

    pilot_symbols = np.load(PILOT_NPY, allow_pickle=True)  # shape (TX_REPS, 4095)
    print("pilot error:",
          np.mean(np.abs(pilot_freq_blocks - pilot_symbols) ** 2))

    # 6) Channel estimation (pilot → H_est for each block)
    #pilot_blocks = useful_freq_blocks[0:: 5]
    #data_blocks = useful_freq_blocks[1:: 5]  # then feed data into your data path

    pilot_symbols = np.load(PILOT_NPY, allow_pickle=True)
    H_pilots = channel_estimation(pilot_freq_blocks, pilot_symbols, method='zf')

    post_eq = pilot_freq_blocks / H_pilots  # now ≈ ideal QPSK
    print("pilot error (post-EQ):",
          np.mean(np.abs(post_eq - pilot_symbols) ** 2))  # expect ≲ 1e-8

    # 7) Equalise & reconstruct data subcarriers
    H_data = np.repeat(H_pilots, 4, axis=0)
    reconstructed_data = data_freq_blocks / H_data

    tx_data_symbols = output["payload_data_blocks"]  # shape (TX_REPS*4, 1944)

    """
    # ── 7.1) Perfect I/Q normalisation ──
    for blk_idx in range(reconstructed_data.shape[0]):
        # take just the real QPSK part (first LDPC_N symbols)
        eq = reconstructed_data[blk_idx, :LDPC_N]
        # transmitted symbols for that block
        tx0 = tx_data_symbols[blk_idx]
        # mask of those that were +1+1j
        mask = (tx0.real > 0) & (tx0.imag > 0)
        # if we have at least two references
        ref_syms = eq[mask]
        if ref_syms.size > 1:
            # compute angle‐error against ideal (+1+1j)
            theta = np.angle(np.sum(np.conj(ref_syms) * (1 + 1j)))
            eq *= np.exp(-1j * theta)
        # now rescale each axis so that +1+1j lands exactly at (1,1)
        if mask.any():
            mu_i = np.mean(eq.real[mask])
            mu_q = np.mean(eq.imag[mask])
        else:
            mu_i = mu_q = 1.0
        eq = (eq.real / (mu_i + 1e-12)) + 1j * (eq.imag / (mu_q + 1e-12))

        # write it back
        reconstructed_data[blk_idx, :LDPC_N] = eq
        """


    for blk_idx in range(reconstructed_data.shape[0]):
        # take just the “real” QPSK part (first LDPC_N symbols) of block blk_idx
        eq_payload = reconstructed_data[blk_idx, :LDPC_N]

        # first derotate by using all the +1+1j pilots as a reference -
        # build mask of transmitted +1+1j symbols
        tx0 = tx_data_symbols[blk_idx]
        mask = (tx0.real > 0) & (tx0.imag > 0)
        """
        # collect your +1+1j refs
        ref_syms = eq_payload[mask]  # length = M
        if ref_syms.size > 1:
            # compute the average conj·prod against (1+1j):
            #   sum_i conj(ref_i) * (1+1j)
            # then take its angle
            theta = np.angle(np.sum(np.conj(ref_syms) * (1 + 1j)))
            eq_payload *= np.exp(-1j * theta)
        """
        # — now rescale each axis so that +1+1j lands exactly at (1,1) —
        mu_i = np.mean(eq_payload.real[mask]) if np.any(mask) else 1.0
        mu_q = np.mean(eq_payload.imag[mask]) if np.any(mask) else 1.0
        eq_payload = (eq_payload.real / (mu_i + 1e-12)) + 1j * (eq_payload.imag / (mu_q + 1e-12))

        # write it back into your array
        reconstructed_data[blk_idx, :LDPC_N] = eq_payload


    print("First‐block EQ symbols (first 8 of raw0, after normalization):")
    #print(np.round(reconstructed_data[0, :8], 3))
    first8 = reconstructed_data[0, :8]
    rounded = [complex(round(x.real, 3), round(x.imag, 3)) for x in first8]
    print(rounded)

    # 8) Plot one constellation for sanity (optional)
    plot_equalised_blocks(
        reconstructed_data[0][:LDPC_N],  # take only the first 1944 symbols
        output["payload_data_blocks"][0]  # this is already length=1944
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

    def qpsk_to_bits(sym_array):
        # sym_array: 1D array of complex QPSK symbols
        bI = (sym_array.real < 0).astype(int)  # 0 if ≥0, 1 if <0
        bQ = (sym_array.imag < 0).astype(int)
        # Now interleave I- and Q-bits into a single 1D array of length 2*N:
        bits = np.zeros(2 * len(sym_array), dtype=int)
        bits[0::2] = bI
        bits[1::2] = bQ
        return bits


    for blk_idx, blk_symbols in enumerate(reconstructed_data):
        eq_payload = blk_symbols[:LDPC_N]  # length = 1944

        """
        # Phase & amplitude normalisation
        # a) Phase-derotate by aligning to the nearest QPSK corner
        hard_ref = np.array([_qpsk_hard_decision(s) for s in eq_payload])
        # Compute average phase offset between received and ideal symbols
        phase_offset = np.angle(np.vdot(hard_ref, eq_payload))
        eq_payload *= np.exp(-1j * phase_offset)

        # b) Amplitude-normalise to unit average power
        avg_power = np.mean(np.abs(eq_payload) ** 2)
        eq_payload /= np.sqrt(avg_power + 1e-12)
        """

        # ——— PRE-LDPC BER ———
        # hard-decide each QPSK symbol back to {0,1} bits
        hard_pts = np.array([_qpsk_hard_decision(s) for s in eq_payload])
        hard_bits = np.zeros(2 * LDPC_N, dtype=int)
        hard_bits[0::2] = (hard_pts.real < 0).astype(int)  # I-bit
        hard_bits[1::2] = (hard_pts.imag < 0).astype(int)  # Q-bit

        # Recover the transmitted codeword bits by inverting the QPSK map:
        tx_bits = qpsk_to_bits(output["payload_data_blocks"][blk_idx])  # your transmitted coded-bit array
        pre_err = np.sum(hard_bits != tx_bits)
        pre_ber = pre_err / float(len(tx_bits))
        print(f"[blk {blk_idx}] PRE-LDPC: {pre_err}/{len(tx_bits)} errors → BER={pre_ber:.2e}")

        # Build LLR vector length=2×1944 = 3888
        llr = np.zeros(2 * LDPC_N, dtype=np.float32)
#        for j, sym in enumerate(eq_payload):
#            llr[2 * j] = (2.0 / sigma2_est) * sym.real
#            llr[2 * j + 1] = (2.0 / sigma2_est) * sym.imag

        SNR_cap = 15.0  # dB, about Eb/N0 = 15 → |LLR| ≈ 10
        gain = min(2.0 / sigma2_est, 10.0)
        for j, sym in enumerate(eq_payload):
            llr[2 * j] = gain * sym.real
            llr[2 * j + 1] = gain * sym.imag

        # FIX: flip all LLR signs because we saw they’re inverted
        #llr = -llr

        # DIAGNOSTIC: check first 10 LLR entries against the true coded bits
        tx_coded = qpsk_to_bits(output["payload_data_blocks"][blk_idx])
        print(f"LLR sign check for block {blk_idx}, first 10 bits:")

        for bit_idx in range(10):
            true_b = tx_coded[bit_idx]
            print(f"  idx {bit_idx:2d}: true={true_b}  LLR={llr[bit_idx]:+.3f}")

        # Clamp every LLR into [–LLR_MAX, +LLR_MAX]
        llr = np.clip(llr, -LLR_MAX, +LLR_MAX)

        # Split into two chunks of length=1944 each
        #llr_cw1 = llr[:LDPC_N]
        #llr_cw2 = llr[LDPC_N: 2 * LDPC_N]

        # LLRs for one OFDM block – length 3888  (1944 symbols × 2 bits)
        llr_pairs = llr.reshape(-1, 2)  # shape (1944 , 2)
        llr_cw1 = llr_pairs[:, 0].ravel()  # 1944 floats, one per bit of CW1
        llr_cw2 = llr_pairs[:, 1].ravel()  # 1944 floats, one per bit of CW2

        # Decode both codewords
        soft1, iters1 = my_ldpc.decode(llr_cw1)  # soft1 is length=972
        hard1 = (soft1 > 0).astype(np.int8)  # threshold at 0 → 0/1 bits

        #print("llr_cw2.shape  =", llr_cw2.shape)  # Should be exactly (1944,)
        #print("LDPC_N        =", LDPC_N)  # Should print 1944
        #print("any NaN in llr_cw2? ", np.isnan(llr_cw2).any())
        #print("any Inf in llr_cw2? ", np.isinf(llr_cw2).any())
        #print("llr_cw2 min/max:", np.nanmin(llr_cw2), np.nanmax(llr_cw2))

        soft2, iters2 = my_ldpc.decode(llr_cw2)
        hard2 = (soft2 > 0).astype(np.int8)
        decoded_info_bits.append((hard1, hard2))

        # ——— POST-LDPC BER ———

        # 1) Extract info bits from each decoded codeword (systematic code)
        rec_info1 = hard1[:LDPC_K]
        rec_info2 = hard2[:LDPC_K]

        # 2) Recover the original info bits for this block
        # (either from output["payload_info_bits"] or by slicing the same PRNG stream)
        tx_two_info = output["payload_info_bits"][blk_idx]  # should be shape (2*LDPC_K,)
        tx_info1 = tx_two_info[0:LDPC_K]
        tx_info2 = tx_two_info[LDPC_K:2 * LDPC_K]

        # 3) Compute per-codeword BER
        err1 = np.sum(rec_info1 != tx_info1)
        err2 = np.sum(rec_info2 != tx_info2)
        ber1 = err1 / float(LDPC_K)
        ber2 = err2 / float(LDPC_K)
        print(f"[blk {blk_idx}] POST-LDPC CW1: {err1}/{LDPC_K} errors → BER={ber1:.2e}")
        print(f"[blk {blk_idx}] POST-LDPC CW2: {err2}/{LDPC_K} errors → BER={ber2:.2e}")


    # 11) Print out first few recovered bits of the first block for verification
    first_rec1, first_rec2 = decoded_info_bits[0]
    print("Decoded info bits (first OFDM symbol):")
    print(" Codeword 1 (first 16 bits):", first_rec1[:16], "…")
    print(" Codeword 2 (first 16 bits):", first_rec2[:16], "…")

    # 12) Optionally compare TX vs RX in time‐domain & show spectrum
    compare_tx_rx(recording, start_payload, end_payload)
    spectrum_plot(recording)