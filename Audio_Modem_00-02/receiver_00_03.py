import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
from scipy import signal, fft
from scipy.io.wavfile import read
from transmitter_00_02 import generate_chirp, WAV_TX, output        
import transmitter_00_02 as tx 
# look into using GPU for hardware acceleration of convolution, e.g. using CuPy for faster processing

# ------------------------------------------------
#   1.  General parameters (unchanged)
# ------------------------------------------------
FS              = tx.FS
FFT_LEN         = tx.FFT_LEN
CP_LEN          = tx.CP_LEN
CHIRP_LEN_S     = tx.CHIRP_LEN_S
SILENCE_LEN_S   = tx.SILENCE_LEN_S
F0, F1          = tx.F0, tx.F1
TX_REPS         = tx.TX_REPS
WAV_TX          = WAV_TX              
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

CHIRP_ATTEN     = tx.CHIRP_ATTEN
TARGET_PEAK     = tx.TARGET_PEAK
LENGTH_TOL      = tx.LENGTH_TOL

# ------------------------------------------------
#   2.  Input/Output
# ------------------------------------------------
def record_audio(expected_len:int, fs:int=FS) -> np.ndarray:
    print(f"Recording ≈{expected_len/fs:.2f} s …")
    rec = sd.rec(expected_len, samplerate=fs, channels=1,
                 dtype='float32').squeeze() #removes extra unused dimension
    sd.wait()
    sf.write(WAV_RX, rec, fs)
    return rec

def load_wav(path):
    data, sr = sf.read(path, always_2d=False) #sr is sample-rate of recording
    assert sr == FS, "sample-rate mismatch"
    return data.astype(np.float32)

# ------------------------------------------------
#   3.  Synchronisation
# ------------------------------------------------
def start_end_synchronise(rx:np.ndarray,
                chirp_up:np.ndarray,
                chirp_down:np.ndarray) -> tuple[np.ndarray,int,int]: # colon tells you what type the function takes, and arrow tells you what the function returns
    corr_up   = signal.correlate(rx, chirp_up,   mode='valid')
    peak_up   = np.argmax(corr_up)
 
    corr_down = signal.correlate(rx, chirp_down, mode='valid')
    search_from = peak_up + len(chirp_up) # search for down-chirp after up-chirp
    peak_down = np.where(corr_down > 0.8*corr_down.max())[0] #formatting, 2D - 1D but no information loss
    peak_down = peak_down[peak_down > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down - CP_LEN
    payload = rx[start_payload:end_payload]
    exp = output["total_ofdm_length"]
    if len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))
    else:
        payload = payload[:exp]
    return payload, start_payload, end_payload   
# ------------------------------------------------
#   4.  OFDM helpers 
# ------------------------------------------------
# def ofdm_blocks(payload):
#     blocks, idx = [], CP_LEN # idx = index 
#     for _ in range(TX_REPS):
#         blocks.append(payload[idx:idx+FFT_LEN]); idx += FFT_LEN # !make a more robust function that can handle blocks with cyclic prefixes throughout
#     return np.array(blocks)

# def ofdm_blocks(payload):
#     blocks, idx = [], 0  # idx = index
#     for _ in range(5):
#         blocks.append(payload[idx:idx+FFT_LEN+CP_LEN]); idx += FFT_LEN + CP_LEN  # !make a more robust function that can handle blocks with cyclic prefixes throughout
#         blocks[-1] = blocks[-1][CP_LEN:]  # remove cyclic prefix
#     return np.array(blocks)

def time_OFDM_blocks(payload, block_length_time = output["ofdm_block_len_with_cp"]):
    time_blocks = []
    if len(payload)%block_length_time != 0:
        raise ValueError("Payload length is not a multiple of block length.")
    for i in range(len(payload)//block_length_time - 0):
        i = i # allows for future non prefixed code at the beginning of the payload
        time_blocks.append(payload[i*block_length_time:(i+1)*block_length_time])
        time_blocks[-1] = time_blocks[-1][CP_LEN:]
    return np.array(time_blocks)


def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2] 

# ------------------------------------------------
#   5.  Channel estimation  
# ------------------------------------------------
def channel_estimate(useful_frequency_blocks: np.ndarray, pilot_symbols: np.ndarray, method: str = 'zf',noise_var: float = 1e-4) -> np.ndarray:

    eps = 1e-12
    averaged_channel_estimates = []
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block

    def estimate_block(freq_block):
        h_zf = freq_block / (pilot_symbols + eps)
        # if method.lower() == 'mmse':
        #     Rhh = np.mean(np.abs(h_zf)**2)
        #     return (Rhh / (Rhh + noise_var)) * h_zf
        return h_zf

    # Pre-compute estimates for all pilot blocks
    pilot_estimates = [estimate_block(freq_block) for freq_block, btype in zip(useful_frequency_blocks, payload_type_list) if btype == 'pilot']

    # Iterate and estimate for data blocks based on nearby pilots
    for idx, btype in enumerate(payload_type_list):
        if btype == 'data':
            # Find surrounding pilot indices
            prev_pilot = next((j for j in range(idx - 1, -1, -1) if payload_type_list[j] == 'pilot'), None)
            next_pilot = next((j for j in range(idx + 1, len(payload_type_list)) if payload_type_list[j] == 'pilot'), None)

            # Get channel estimates from pilot blocks
            averaged_channel_estimates = []
            if prev_pilot is not None:
                averaged_channel_estimates.append(estimate_block(useful_frequency_blocks[prev_pilot]))
            if next_pilot is not None:
                averaged_channel_estimates.append(estimate_block(useful_frequency_blocks[next_pilot]))

            avg_est = np.mean(averaged_channel_estimates, axis=0)
            averaged_channel_estimates.append(avg_est)

    H_est_array = np.array(averaged_channel_estimates)
    np.save(CHAN_NPY, H_est_array)
    return H_est_array

def reconstruct_data_blocks(useful_frequency_blocks, H_est_array):
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block
    assert len(useful_frequency_blocks) == len(payload_type_list), "Mismatch between blocks and payload types"
    assert len(H_est_array) == payload_type_list.count("data"), "Mismatch between channel estimates and payload types"
    data_blocks = [useful_frequency_blocks[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data']
    data_blocks = np.array(data_blocks) 
    decoded_datablocks = H_est_array // data_blocks  # element-wise division
    return decoded_datablocks

def equalise(rx_fd, H):
    return rx_fd / H

# def _normalise_symbols(z):
#     """unit-power normalisation after equalisation"""
#     return z / (np.sqrt(np.mean(np.abs(z)**2)) + 1e-12)

# ------------------------------------------------
#   6.  New visualisation helpers                
# ------------------------------------------------
def plot_channel(H:np.ndarray):
    """Visualise magnitude & phase of the estimated channel."""
    fig, ax = plt.subplots(2, 1, figsize=(9,4), sharex=True)
    ax[0].plot(20*np.log10(np.abs(H)+1e-12))
    ax[0].set_ylabel("|H| [dB]")
    ax[0].set_title("Estimated channel magnitude / phase")
    ax[1].plot(np.angle(H))
    ax[1].set_ylabel("∠H [rad]")
    ax[1].set_xlabel("sub-carrier")
    plt.tight_layout(); plt.show()

def compare_tx_rx(rx:np.ndarray, start:int, end:int, tx_path:str=WAV_TX):
    """
    Trim off leading silence from TX, align lengths exactly to chirp-OFDM-chirp
    span, normalise both segments by their own peaks, then overlay.
    """
    tx_sig  = load_wav(tx_path)


    tx_start = int(SILENCE_LEN_S * FS)              # first chirp sample
    seg_len  = end - start                          # length of interest
    tx_seg   = tx_sig[tx_start : tx_start + seg_len]
    rx_seg   = rx[start       : start + seg_len]    # trimmed RX

    m = max(np.max(np.max(np.abs(rx_seg))), 1e-3) 
    n = max(np.max(np.max(np.abs(tx_seg))), 1e-3) 
    tx_seg, rx_seg = tx_seg/n, rx_seg/m #-#-# normalised amplitudes by respective max values  

    plt.figure(figsize=(10,3))
    plt.plot(tx_seg, label='TX (norm.)', lw=.8)
    plt.plot(rx_seg, label='RX (norm.)', lw=.6, alpha=.7)
    plt.title("TX vs RX waveform (aligned, silence removed)")
    plt.xlabel("sample"); plt.ylabel("normalised amplitude")
    plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------------------------
#   7.  Spectrum & constellation 
# ------------------------------------------------
def _means_by_colour(z_flat, colours_flat):
    ucols = np.unique(colours_flat) # pulls out an array of the unique(in this case 4 colours) colours used
    means = {c: np.mean(z_flat[colours_flat == c]) for c in ucols} # finds the mean of each colour in the constellation
    return means # a dictionary of colour:mean pairs

def spectrum_plot(sig:np.ndarray, fs:int=FS):
    f, Pxx = signal.welch(sig, fs, nperseg=4096)
    plt.figure(); plt.semilogy(f, Pxx)
    plt.title("Received PSD"); plt.xlabel("Hz"); plt.ylabel("PSD [V²/Hz]")
    plt.tight_layout(); plt.show()
 
# def constellation_plot(eq_fd: np.ndarray): # essentially takes in already equalised( i.e channel effects removed) frequency domain symbols and plots them using the transmitter symbol colours - the colours will loop round after base colur length is exceeded so you can plot multiple symbols at once - it also adds a legend to the plot
#     # ------------------------------------------------------------
#     # 1.  Build a colour array that is **exactly** len(eq_fd)
#     # ------------------------------------------------------------
#     base_col = np.load(COLMAP_NPY, allow_pickle=True)

#     # --- unwrap “array([list([...])], dtype=object)” -------------
#     if (base_col.ndim == 1 and len(base_col) == 1
#             and isinstance(base_col[0], (list, np.ndarray))):
#         base_col = np.asarray(base_col[0], dtype=str)

#     base_col = np.asarray(base_col, dtype=str)         # make flat 1-D

#     if base_col.size == 0:
#         base_col = np.array(['k'])                     # fallback colour

#     reps     = int(np.ceil(eq_fd.size / base_col.size))
#     colours  = np.tile(base_col, reps)[:eq_fd.size]

#     # ------------------------------------------------------------
#     # 2.  Normalise constellation energy
#     # ------------------------------------------------------------
#     eq_fd_n = eq_fd / (np.sqrt(np.mean(np.abs(eq_fd)**2)) + 1e-12)

#     eq_fd_n = eq_fd_n.ravel()          # <<<  NEW  (make it 1-D)
#     colours = colours.ravel()          # <<<  NEW  (defensive; already 1-D)

#     # ------------------------------------------------------------
#     # 3.  Scatter plot
#     # ------------------------------------------------------------
#     plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
#     plt.scatter(eq_fd_n.real, eq_fd_n.imag,
#                 c=colours, s=12, edgecolors='none', alpha=.82)

#     # ------------------------------------------------------------
#     # 4.  Legend – map each colour to nominal point
#     # ------------------------------------------------------------
#     # Use the transmitter’s colour dictionary if available
#     try:
#         from transmitter_00_02 import Q_COL
#         colour_map = {v:k for k,v in Q_COL.items()}  # colour→bits
#         label_map  = {'00':'1+1j','01':'1-1j','11':'-1-1j','10':'-1+1j'}
#         legend_elems = []
#         for c in np.unique(colours):
#             bits = ''.join(map(str, colour_map.get(c, ('?','?'))))
#             legend_elems.append(
#                 Patch(facecolor=c, label=label_map.get(bits, bits)))
#         plt.legend(handles=legend_elems, loc='upper right', fontsize='small')
#     except ImportError:
#         pass  # transmitter not available – skip legend

#     # ------------------------------------------------------------
#     # 5.  Per-quadrant means (computed on **normalised** points)
#     # ------------------------------------------------------------
#     means = {c: np.mean(eq_fd_n[colours == c]) for c in np.unique(colours)}
#     for c, m in means.items():
#         plt.plot(m.real, m.imag, 'kx')
#         plt.text(m.real, m.imag,
#                  f"{m.real:+.2f}+{m.imag:+.2f}j",
#                  fontsize=7, ha='left', va='bottom')

#     plt.title("Equalised constellation (unit power)")
#     plt.xlabel("I"); plt.ylabel("Q")
#     plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

#     # ---- console read-out --------------------------------------
#     print("\nConstellation means by colour:")
#     for c, m in means.items():
#         print(f"{c}: {m.real:+.3f}{m.imag:+.3f}j")

# def simple_constellation_plot(eq_fd:np.ndarray):
#     col = np.load(COLMAP_NPY)
#     colours = np.tile(col,1)
#     plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
#     plt.scatter(eq_fd[1].real, eq_fd[1].imag, c=colours,
#                 s=10, alpha=.85, edgecolors='none')
#     plt.title("Equalised constellation"); plt.xlabel("I"); plt.ylabel("Q")
#     plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()



def plot_constellation_blocks(decoded_blocks, indices=None):
    """
    Plot constellation points for one or multiple decoded OFDM data blocks.

    Parameters:
    - decoded_blocks: np.ndarray or list/array of np.ndarrays
        Equalised frequency-domain decoded blocks to plot.
    - indices: list[int] or None
        Optional indices specifying which blocks to plot (if decoded_blocks is a list).
        If None, plots all blocks if input is a list, or the single block if input is an array.

    Behavior:
    - Normalizes each block to unit power.
    - Cycles through a base color map loaded from COLMAP_NPY for points.
    - Adds a legend showing the symbol mapping if available.
    """
    # Make sure decoded_blocks is a list of arrays
    if isinstance(decoded_blocks, np.ndarray) and decoded_blocks.ndim > 1:
        # Already a 2D array, treat as one block
        blocks_to_plot = [decoded_blocks]
    else:
        # Assume list-like or 1D array of blocks
        blocks_to_plot = list(decoded_blocks)

    if indices is not None:
        blocks_to_plot = [blocks_to_plot[i] for i in indices]

    # Load base colors
    base_colors = np.load(COLMAP_NPY, allow_pickle=True)
    if base_colors.ndim == 1 and len(base_colors) == 1 and isinstance(base_colors[0], (list, np.ndarray)):
        base_colors = np.asarray(base_colors[0], dtype=str)
    base_colors = np.asarray(base_colors, dtype=str)
    if base_colors.size == 0:
        base_colors = np.array(['k'])

    plt.figure()
    plt.axhline(0, c='k'); plt.axvline(0, c='k')

    # Plot each block with cyclic colors
    for block_idx, block in enumerate(blocks_to_plot):
        block = np.asarray(block).ravel()
        # Normalize block power to unit power
        norm_block = block / (np.sqrt(np.mean(np.abs(block)**2)) + 1e-12)
        reps = int(np.ceil(norm_block.size / base_colors.size))
        colors = np.tile(base_colors, reps)[:norm_block.size]

        plt.scatter(norm_block.real, norm_block.imag, c=colors, s=12, alpha=0.8, label=f"Block {indices[block_idx] if indices else block_idx}")

    # Legend setup, if transmitter color map is available
    try:
        from transmitter_00_02 import Q_COL
        colour_map = {v: k for k, v in Q_COL.items()}  # color to bits mapping
        label_map = {'00': '1+1j', '01': '1-1j', '11': '-1-1j', '10': '-1+1j'}
        unique_colors = np.unique(base_colors)
        legend_elements = []
        for c in unique_colors:
            bits = ''.join(map(str, colour_map.get(c, ('?','?'))))
            legend_elements.append(Patch(facecolor=c, label=label_map.get(bits, bits)))
        plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
    except ImportError:
        pass  # no transmitter module, skip legend

    plt.title("Equalised constellation (unit power)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # record_audio(480000)

    SAMPLE_RATE, recording = read('rx_recording.wav')
    SAMPLE_RATE, transmission = read("tx_sequence.wav")

    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

    print(len(recording))
    sync = start_end_synchronise(recording, chirp_up, chirp_down)
    print(sync)
    print(sync[0], sync[1], sync[2]) # payload, start payload, end payload

    compare_tx_rx(recording, sync[1], sync[2])

    split_block = ofdm_blocks(sync[0])
    freq_block = freq_domain(split_block)
    channel = channel_estimate(freq_block, np.load("pilot_symbols.npy"), "zf")
    plot_channel(channel)
    eq_block = equalise(freq_block, channel)
    print(eq_block.shape)
    spectrum_plot(recording)
    simple_constellation_plot(eq_block)
    constellation_plot(eq_block)