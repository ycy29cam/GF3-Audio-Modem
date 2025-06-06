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
WAV_RX_1        = 'rx_recording_group1.wav'
WAV_RX_3        = 'rx_recording_group3.wav'
PILOT_NPY       = 'pilot_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'
PILOT_TIME_NO_CP_NPY = "time_pilot_blocks_no_cp.npy"

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
    plt.plot(corr_up, label='up-chirp correlation')
    plt.plot(corr_down, label='down-chirp correlation')
    plt.plot(rx*5000, label='received signal', alpha=0.5,)
    search_from = peak_up + len(chirp_up) # search for down-chirp after up-chirp
    peak_down = np.where(corr_down > 0.5*corr_down.max())[0] #formatting, 2D - 1D but no information loss
    peak_down = peak_down[peak_down > search_from][0]
    

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down 
    print("start_payload:", start_payload, "end_payload:", end_payload)
    payload = rx[start_payload:end_payload]
    exp = output["total_ofdm_length"]
    if len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))
    else:
        payload = payload[:exp]
    # sf.write("chopped_payload_sound.wav", payload, FS)
    return payload, start_payload, end_payload   

# def time_OFDM_chopper(payload, block_length_time = output["ofdm_block_len_with_cp"]):
#     time_blocks = []
#     print(len(payload)) 
#     print((output["no_of_payload_blocks"]))
#     if len(payload)%block_length_time != 0:
#         raise ValueError("Payload length is not a multiple of block length.")
#     for i in range(len(payload)//block_length_time - 0):
#         i = i # allows for future non prefixed code at the beginning of the payload
#         time_blocks.append(payload[i*block_length_time:(i+1)*block_length_time])
#         time_blocks[-1] = time_blocks[-1][CP_LEN:]
#     return np.array(time_blocks)

def sync_chopper(payload, start_payload, end_payload, rx, block_length_time = output["ofdm_block_len_with_cp"]):
    time_blocks = []
    x = start_payload - block_length_time/2
    y = start_payload + block_length_time*(3/2)
    #replace with correct number of windows in a sec: 
    sync_peak_index = []
    sync_max = []
    time_pilot_blocks_no_cp = np.load(PILOT_TIME_NO_CP_NPY)
    for i in range(5):
        window = rx[x:y]
        pilot_correlation = signal.correlate(window, time_pilot_blocks_no_cp[i] )
        plt.plot(pilot_correlation)
        sync_start = x + np.argmax(pilot_correlation)
        sync_max.append(np.max(pilot_correlation))
        sync_peak_index.append(sync_start)
        chopped_start_index = start_payload + i * 5 * block_length_time
        bit_diff = (sync_start - chopped_start_index) * (output["modulation_bits_per_sample"])
        if abs(bit_diff) > 5:
            print(f" Desync on pilot block {i}: sync start index {sync_start}, expected {chopped_start_index}, diff = {bit_diff} bits")
            sync_peak_index[-1] = chopped_start_index
        x += 4*block_length_time
        y += 4*block_length_time

    for i in sync_peak_index:
        start = i + block_length_time
        for _ in range(4):
            time_blocks.append(payload[start:start+ block_length_time])
            start += block_length_time
    return np.array(time_blocks)
    
    

    





def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2] 

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


def reconstruct_data_blocks(useful_frequency_blocks, H_est_array):
    payload_type_list = output["payload_type_list"]  # list of 'pilot' and 'data' for each block
    assert len(useful_frequency_blocks) == len(payload_type_list), "Mismatch between blocks and payload types"
    assert len(H_est_array) == len(payload_type_list), "Mismatch between channel estimates and payload types"
    data_blocks = np.array([useful_frequency_blocks[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    data_H_est_array = np.array([H_est_array[idx] for idx, btype in enumerate(payload_type_list) if btype == 'data'])
    decoded_datablocks = data_blocks/data_H_est_array  # element-wise division
    return decoded_datablocks

def equalise(rx_fd, H):
    return rx_fd / H
 
# ------------------------------------------------
#   6. Visualisation helpers                
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

def compare_tx_rx(rx:np.ndarray, start_rx_payload:int, end_rx_payload_boundary:int, tx_path:str=WAV_TX):
    """
    Compares the extracted RX payload against the corresponding TX payload.
    start_rx_payload: Index in rx where the payload begins (after up-chirp).
    end_rx_payload_boundary: Index in rx where the down-chirp begins (payload ends before this).
    """
    tx_sig   = load_wav(tx_path)

    tx_leading_silence = output["leading_silence_samples"]
    tx_chirp_len = output["chirp_samples"] # Length of the core chirp signal
    tx_start_of_payload = tx_leading_silence + tx_chirp_len
    payload_length_to_compare = output["total_ofdm_length"] # This is 'exp'

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

    tx_payload_seg = tx_sig[tx_start_of_payload : tx_seg_end]
    rx_payload_seg = rx[start_rx_payload : rx_seg_end]
    
    # Normalize for plotting
    m_peak = np.max(np.abs(rx_payload_seg)) if rx_payload_seg.size > 0 else 0
    n_peak = np.max(np.abs(tx_payload_seg)) if tx_payload_seg.size > 0 else 0
    
    tx_norm = tx_payload_seg / n_peak
    rx_norm = rx_payload_seg / m_peak

    plt.figure(figsize=(10,3))
    plt.plot(tx_norm, label='TX Payload (norm.)', lw=.8)
    plt.plot(rx_payload_seg, label='RX Payload (norm.)', lw=.6, alpha=.7)
    plt.title("TX vs RX OFDM Payload (aligned)")
    plt.xlabel("sample in payload")
    plt.ylabel("normalised amplitude")
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
    sym_to_colour = {1+1j: Q_COL[(0, 0)],
                     1-1j: Q_COL[(0, 1)],
                    -1-1j: Q_COL[(1, 1)],
                    -1+1j: Q_COL[(1, 0)]}
    
    tx_colours = np.array([sym_to_colour.get(complex(round(s.real), round(s.imag)), 'k')
                           for s in tx_flat])

    # Normalise constellation energy (unit average power)
    eq_flat /= np.sqrt(np.mean(np.abs(eq_flat)**2) + 1e-12)

    # --- Plotting ---
    plt.figure(); plt.axhline(0, c='k'); plt.axvline(0, c='k')
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
    plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()


if __name__ == "__main__":

    #------------------import sound file--------------------------------
    # record_audio(480000)
    SAMPLE_RATE, recording = read('rx_recording.wav')
    # recording = output["waveform"] # works flawlessly, which tells me i'm doing the theory correctly, i'm just missing correction details
    # SAMPLE_RATE, transmission = read("tx_sequence.wav")
    chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)


    #------------------run reciever--------------------------------
    payload, start_payload, end_payload = start_end_synchronise(recording, chirp_up, chirp_down)
    time_blocks = time_OFDM_chopper(payload)
    useful_freq_blocks  = freq_domain(time_blocks)
    h_estimated_array = channel_estimation(useful_freq_blocks, np.load(PILOT_NPY), "zf")
    reconstructed_data = reconstruct_data_blocks(useful_freq_blocks, h_estimated_array)
    #when we eventually work with unknown data blocks, we would then need to do maximum likelihood estimation to find the most likely data blocks from the reconstructed data blocks
    #for now we will just plot the equalised blocks and see how they look qualitatively
    plot_equalised_blocks(reconstructed_data[8], output["payload_data_blocks"][8])



    
    #------------------testing outputs--------------------------------
    print(len(recording))
    print ("transmitted signal: ", payload,"start bin: ", start_payload, "end bin: ", end_payload)
    compare_tx_rx(recording, start_payload, end_payload)
    spectrum_plot(recording)



    #-------------------to correct----------------------------------
    """
    //switch from 1 data 1 pilot to 4 data 1 pilot, and implement the padding at the end to finish on 4 data blocks - see 2 standardisation meetings back
    //change the pilot generation to fill only 1-4095 and put the other 2 values to 0 
    //change the chirp to sine version as outlined on slack code dump
    //synchronise for each OFDM symbol using either a) a moving window as per yesterday or b) a cross correlation function, this might not work super well, am gonna test first and plot to see how the peaks look
    //normalise the recieved signal so that your signal has the correct magnitude, getting a factor of 1-2/4 out which is problematic
    //think about cutting out the very beginning of the recording, although probably less necessary with a longer start chirp. 
    //change the channel estimation method to use the nearest n blocks say, which should give more robust channel noise averaging, especially if each estimated channel response is synchronised correctly. 
    //check our constellation mapping is 00 - 10 - 11 - 01 going anticlockwise
    //add a set of "guard" bins set to 0 for the time being after the starting downchirp
    //check what amplitude we want to send our white noise at ( eg scaled by 1, 0.1 ect - might be moot for the speakers)
    //add a time based linear interpolation time compensation (using chatGPT code, MAX) but run it on each block once resynchronised if possible, or run it over the whole thing and still resynchronise, whatever works easier.
    //add in useful data in bin 200 to 2143 (both inclusive), using python's indexing function,  especially useful for knowing when it comes to LDPC codes. this gives a useful data rate of 1944 bits.
    //add padding at the end, a frame is Pilot, data, data, data, data. sometimes doesnt finish in that format, so need to detect that and pad the end (with random noise, or 0's more likely) - the number of blocks can be done after normalisation by taking modulo 5 and adding that many prefixed(or non prefixed whatever) bits to our signal.

    """