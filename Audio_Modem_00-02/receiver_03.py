import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
from scipy import signal, fft
from scipy.io.wavfile import read
from transmitter_01 import generate_chirp, WAV_TX          #  <<< changed

# ------------------------------------------------
#   !!! READ ME !!!
#   
#   Implemented channel estimation for the specific case, where five blocks are transmitted
#   | prefix - pilot | pilot | prefix - data | prefix - pilot | pilot |
#   Average of the pilots is taken for channel estimation
#
#   Things to work on:
#   - Average the channel response and not the input frequency blocks
#   - The transmitted sequence used by Max should have prefix in front of every block, except for the first up chirp
# ------------------------------------------------

# ------------------------------------------------
#   1.  General parameters (unchanged)
# ------------------------------------------------
FS              = 48_000
FFT_LEN         = 8192
CP_LEN          = FFT_LEN // 4
CHIRP_LEN_S     = 2
SILENCE_LEN_S   = 1.0
F0, F1          = 20, 15000
TX_REPS         = 4
WAV_TX          = WAV_TX                               #  keep same name
WAV_RX          = 'rx_recording.wav'
PILOT_NPY       = 'pilot_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

CHIRP_ATTEN     = 0.80
TARGET_PEAK     = 0.80
LENGTH_TOL      = 512

# ------------------------------------------------
#   2.  I/O helpers  (unchanged)
# ------------------------------------------------
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

# ------------------------------------------------
#   3.  Synchronisation  (unchanged)
# ------------------------------------------------
def synchronise(rx:np.ndarray,
                chirp_up:np.ndarray,
                chirp_down:np.ndarray) -> tuple[np.ndarray,int,int]:
    corr_up   = signal.correlate(rx, chirp_up,   mode='valid')
    peak_up   = np.argmax(corr_up)

    corr_down = signal.correlate(rx, chirp_down, mode='valid')
    search_from = peak_up + len(chirp_up)
    peak_down = np.where(corr_down > 0.8*corr_down.max())[0]
    peak_down = peak_down[peak_down > search_from][0]

    start_payload = peak_up + len(chirp_up)
    end_payload   = peak_down
    payload = rx[start_payload:end_payload]
    exp = CP_LEN + TX_REPS*FFT_LEN
    if len(payload) > exp + LENGTH_TOL:
        payload = payload[:exp]
    elif len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))
    return payload, start_payload, end_payload   #  unchanged return

# ------------------------------------------------
#   4.  OFDM helpers  (unchanged)
# ------------------------------------------------
def ofdm_blocks(payload):
    blocks, idx = [], CP_LEN
    for _ in range(TX_REPS):
        blocks.append(payload[idx:idx+FFT_LEN]); idx += FFT_LEN
    return np.stack(blocks)

def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2]

def remove_cpe(rx_fd):
    """
    Decision-directed CPE removal per OFDM symbol.
    Returns corrected rx_fd and list of phi_k.
    """
    corrected = np.empty_like(rx_fd)
    phis = []
    for k, sym in enumerate(rx_fd):
        # exploit QPSK: (±1±j)^4 = 1
        est = np.mean(sym**4)
        phi = 0.25 * np.angle(est)
        phis.append(phi)
        corrected[k] = sym * np.exp(-1j*phi)
    return corrected, np.array(phis)

def refine_cfo(phis):
    """
    Estimate Δf from slope of phis versus symbol index.
    df = (Δφ) / (2π T_sym)
    """
    k = np.arange(len(phis))
    # least-squares slope
    slope = np.polyfit(k, np.unwrap(phis), 1)[0]
    T_sym = FFT_LEN / FS
    df = slope / (2*np.pi*T_sym)
    return df

# ------------------------------------------------
#   5.  Channel estimation  (NEW options)          <<< changed
# ------------------------------------------------
def channel_estimate(rx_fd, pilot, method='zf', noise_var=1e-4):
    eps = 1e-12

    #   PROBLEM: Average over H, not average over Y
    #   Resolved? ()

    Y   = rx_fd[0]
    if method.lower() == 'mmse':
        H_zf  = Y / (pilot + eps)
        Rhh   = np.mean(np.abs(H_zf)**2)
        H_hat = (Rhh / (Rhh + noise_var)) * H_zf
    elif method.lower() == 'tikhonov':
        # Tikhonov regularisation directly from pilot
        H_hat = (np.conj(pilot) * Y) / (np.abs(pilot)**2 + noise_var + eps)
    else:  # zero-forcing
        H_hat = Y / (pilot + eps)
    np.save(CHAN_NPY, H_hat)
    return H_hat

def adaptive_channel_equalise(rx_fd: np.ndarray,
                              pilot: np.ndarray,
                              method: str = 'tikhonov',
                              noise_var: float = 1e-4,
                              n_mean: int = 1,
                              do_decision_directed: bool = True):
    """
    Perform per-block channel estimation, optionally using decision-directed 
    pilot refresh on blocks ≥ n_mean.

    Inputs:
      rx_fd       : shape = (TX_REPS, Ntones), the FFT output of each OFDM block.
                    TX_REPS is typically 4; Ntones = FFT_LEN/2 - 1 (e.g. 4095).
      pilot       : length Ntones, the known complex pilot used on block 0.
      method      : one of 'zf', 'mmse', or 'tikhonov'.
      noise_var   : noise‐variance for MMSE/Tikhonov.
      n_mean      : integer ∈ [1 .. TX_REPS], how many blocks to average
                    for the *initial* channel estimate. Default=1 uses only rx_fd[0].
      do_decision_directed : if True, for k ≥ n_mean use decoded symbols as 
                    pseudo-pilot to refresh Ĥ_k.

    Returns:
      eq_fd_flat    : length = TX_REPS * Ntones, the concatenated equalised symbols.
      H_all         : shape = (TX_REPS, Ntones), the channel estimate for each block.
      decoded_bits  : shape = (TX_REPS, Ntones, 2), the final (b0,b1) decision-bits.
    """
    TX_REPS, Ntones = rx_fd.shape

    # 1) Sanity‐check n_mean
    if not (1 <= n_mean <= TX_REPS):
        raise ValueError(f"n_mean must be between 1 and {TX_REPS}, got {n_mean}")

    # 2) Ensure pilot is flat 1-D complex of length Ntones
    pilot = np.asarray(pilot, dtype=np.complex64).ravel()
    if pilot.size != Ntones:
        raise ValueError(f"pilot length {pilot.size} ≠ Ntones {Ntones}")

    # 3) Compute initial averaged Y over first n_mean blocks
    Y_avg = np.mean(rx_fd[:n_mean, :], axis=0)   # shape = (Ntones,)

    eps = 1e-12
    method = method.lower()

    # 4) Helper to estimate H from arbitrary (Yvec, Pvec) with chosen method
    def _estimate_from_pairs(Yvec: np.ndarray, Pvec: np.ndarray) -> np.ndarray:
        """
        Given Yvec (received subcarriers) and Pvec (pilot symbols, real or
        decision-directed), return Ĥ according to 'zf','mmse','tikhonov'.
        """
        # Convert to complex64 1-D
        Yv = np.asarray(Yvec, dtype=np.complex64).ravel()
        Pv = np.asarray(Pvec, dtype=np.complex64).ravel()
        if Yv.size != Ntones or Pv.size != Ntones:
            raise ValueError("Yvec and Pvec must both have length Ntones")

        if method == 'zf':
            return Yv / (Pv + eps)

        elif method == 'mmse':
            H_zf = Yv / (Pv + eps)
            Rhh  = np.mean(np.abs(H_zf)**2)
            return (Rhh / (Rhh + noise_var)) * H_zf

        elif method == 'tikhonov':
            # Tikhonov: conj(Pv)*Yv / (|Pv|^2 + noise_var)
            denom = np.abs(Pv)**2 + noise_var + eps
            return (np.conj(Pv) * Yv) / denom

        else:
            raise ValueError(f"Unknown method '{method}': choose 'zf','mmse','tikhonov'")

    # 5) Build arrays to hold results
    H_all        = np.zeros((TX_REPS, Ntones), dtype=np.complex64)
    eq_blocks    = []  # list of length TX_REPS, each entry shape=(Ntones,)
    decoded_bits = np.zeros((TX_REPS, Ntones, 2), dtype=int)

    # 6) Estimate H_avg from Y_avg and true pilot
    H_avg = _estimate_from_pairs(Y_avg, pilot)
    # Assign H_avg to blocks 0..(n_mean-1)
    for k in range(n_mean):
        H_all[k,:] = H_avg

    # 7) Equalise & decode blocks 0..(n_mean-1)
    for k in range(n_mean):
        sym_k = rx_fd[k]
        eq_k  = sym_k / H_avg
        eq_blocks.append(eq_k)

        # Decode bits for possible later decision-directed use
        decoded_bits[k, :, 0] = (eq_k.real > 0).astype(int)
        decoded_bits[k, :, 1] = (eq_k.imag > 0).astype(int)

    # 8) For blocks k = n_mean..TX_REPS-1, do decision-directed refresh
    for k in range(n_mean, TX_REPS):
        # (a) Build pseudo-pilot P_dec from decoded bits of block (k-1)
        b0 = decoded_bits[k-1, :, 0]         # shape=(Ntones,)
        b1 = decoded_bits[k-1, :, 1]
        P_dec = (2*b0 - 1) + 1j * (2*b1 - 1)  # 1-D complex, ±1±j

        # (b) Estimate H_k from (Y_k = rx_fd[k], P_dec)
        Yk    = rx_fd[k]
        Hk    = _estimate_from_pairs(Yk, P_dec)
        H_all[k, :] = Hk

        # (c) Equalise block k
        eq_k = Yk / Hk
        eq_blocks.append(eq_k)

        # (d) Decode bits for block k
        decoded_bits[k, :, 0] = (eq_k.real > 0).astype(int)
        decoded_bits[k, :, 1] = (eq_k.imag > 0).astype(int)

    # 9) Concatenate all equalised blocks into a flat vector
    eq_fd_flat = np.concatenate(eq_blocks)  # length = TX_REPS * Ntones

    return eq_fd_flat, H_all, decoded_bits


def equalise(rx_fd:np.ndarray, H:np.ndarray) -> np.ndarray:
    return rx_fd / H

def _normalise_symbols(z):
    """unit-power normalisation after equalisation"""
    return z / (np.sqrt(np.mean(np.abs(z)**2)) + 1e-12)

# ------------------------------------------------
#   6.  New visualisation helpers                 <<< changed
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
    Trim leading silence, cut both TX & RX to [start,end), then
    normalise each separately and overlay.
    """
    tx_sig = load_wav(tx_path)
    tx_start = int(SILENCE_LEN_S * FS)
    seg_len  = end - start

    tx_seg = tx_sig[tx_start : tx_start + seg_len]
    rx_seg = rx[start       : start + seg_len]

    # separate normalisation
    m_tx = np.max(np.abs(tx_seg)) or 1e-3
    m_rx = np.max(np.abs(rx_seg)) or 1e-3
    tx_n = tx_seg / m_tx
    rx_n = rx_seg / m_rx

    # optionally offset RX slightly downward for visibility
    rx_n = rx_n - 0.02

    plt.figure(figsize=(10,3))
    plt.plot(tx_n, label='TX (norm.)', lw=1.2, alpha=0.9)
    plt.plot(rx_n, label='RX (norm.)', lw=0.6, alpha=0.6)
    plt.title("TX vs RX waveform (aligned, silence removed)")
    plt.xlabel("sample"); plt.ylabel("normalised amplitude")
    plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------------------------
#   7.  Spectrum & constellation (unchanged)
# ------------------------------------------------
def _means_by_colour(z_flat, colours_flat):
    ucols = np.unique(colours_flat)
    means = {c: np.mean(z_flat[colours_flat == c]) for c in ucols}
    return means

def spectrum_plot(sig:np.ndarray, fs:int=FS):
    f, Pxx = signal.welch(sig, fs, nperseg=4096)
    plt.figure(); plt.semilogy(f, Pxx)
    plt.title("Received PSD"); plt.xlabel("Hz"); plt.ylabel("PSD [V²/Hz]")
    plt.tight_layout(); plt.show()

def constellation_plot(eq_fd: np.ndarray):
    # ------------------------------------------------------------
    # 1.  Build a colour array that is **exactly** len(eq_fd)
    # ------------------------------------------------------------
    base_col = np.load(COLMAP_NPY, allow_pickle=True)

    # --- unwrap “array([list([...])], dtype=object)” -------------
    if (base_col.ndim == 1 and len(base_col) == 1
            and isinstance(base_col[0], (list, np.ndarray))):
        base_col = np.asarray(base_col[0], dtype=str)

    base_col = np.asarray(base_col, dtype=str)         # make flat 1-D

    if base_col.size == 0:
        base_col = np.array(['k'])                     # fallback colour

    reps     = int(np.ceil(eq_fd.size / base_col.size))
    colours  = np.tile(base_col, reps)[:eq_fd.size]

    # ------------------------------------------------------------
    # 2.  Normalise constellation energy
    # ------------------------------------------------------------
    eq_fd_n = eq_fd / (np.sqrt(np.mean(np.abs(eq_fd)**2)) + 1e-12)

    eq_fd_n = eq_fd_n.ravel()          # <<<  NEW  (make it 1-D)
    colours = colours.ravel()          # <<<  NEW  (defensive; already 1-D)

    # ------------------------------------------------------------
    # 3.  Scatter plot
    # ------------------------------------------------------------
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(eq_fd_n.real, eq_fd_n.imag,
                c=colours, s=12, edgecolors='none', alpha=.82)

    # ------------------------------------------------------------
    # 4.  Legend – map each colour to nominal point
    # ------------------------------------------------------------
    # Use the transmitter’s colour dictionary if available
    try:
        from transmitter_01 import Q_COL
        colour_map = {v:k for k,v in Q_COL.items()}  # colour→bits
        label_map  = {'00':'-1-1j','01':'-1+1j','11':'1+1j','10':'1-1j'}
        legend_elems = []
        for c in np.unique(colours):
            bits = ''.join(map(str, colour_map.get(c, ('?','?'))))
            legend_elems.append(
                Patch(facecolor=c, label=label_map.get(bits, bits)))
        plt.legend(handles=legend_elems, loc='upper right', fontsize='small')
    except ImportError:
        pass  # transmitter not available – skip legend

    # ------------------------------------------------------------
    # 5.  Per-quadrant means (computed on **normalised** points)
    # ------------------------------------------------------------
    means = {c: np.mean(eq_fd_n[colours == c]) for c in np.unique(colours)}
    for c, m in means.items():
        plt.plot(m.real, m.imag, 'kx')
        plt.text(m.real, m.imag,
                 f"{m.real:+.2f}+{m.imag:+.2f}j",
                 fontsize=7, ha='left', va='bottom')

    plt.title("Equalised constellation (unit power)")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

    # ---- console read-out --------------------------------------
    print("\nConstellation means by colour:")
    for c, m in means.items():
        print(f"{c}: {m.real:+.3f}{m.imag:+.3f}j")

def simple_constellation_plot(eq_fd:np.ndarray):
    col = np.load(COLMAP_NPY)
    colours = np.tile(col, TX_REPS)
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(eq_fd.real, eq_fd.imag, c=colours,
                s=10, alpha=.85, edgecolors='none')
    plt.title("Equalised constellation"); plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

def symbol_error_rate(eq_fd: np.ndarray):
    """
    Compute and print symbol‐error‐rate for each of the 4 repeated pilot blocks,
    and overall across all blocks.
    eq_fd may be shape (4, tones) or flat length 4×tones.
    """
    # 1) Load reference pilot bits
    pilot_syms = np.load(PILOT_NPY)         # shape = (tones,)
    base_col   = np.load(COLMAP_NPY, allow_pickle=True)
    from transmitter_01 import Q_COL
    inv_map = {v:k for k,v in Q_COL.items()}  # colour→(b0,b1)

    # unwrap nested
    if base_col.ndim==1 and len(base_col)==1 and isinstance(base_col[0], (list,np.ndarray)):
        base_col = np.asarray(base_col[0])
    ref_bits = np.array([inv_map[c] for c in base_col], dtype=int)  # (tones,2)

    # 2) Shape eq_fd into (4, tones)
    eq = np.asarray(eq_fd)
    tones = pilot_syms.size
    if eq.ndim==1:
        eq = eq.reshape(TX_REPS, tones)
    elif eq.shape[0] != TX_REPS:
        eq = eq.reshape(tones, TX_REPS).T

    # 3) Decode by sign: real>0→1, imag>0→1
    dec = np.zeros((TX_REPS, tones, 2), dtype=int)
    dec[:,:,0] = (eq.real > 0)
    dec[:,:,1] = (eq.imag > 0)

    # 4) Compare & count
    total_per_block = tones
    errs = np.any(dec != ref_bits[None,:,:], axis=2)  # shape (4,tones)

    # 5) Print per‐block
    total_all = TX_REPS*tones
    errs_all  = 0
    print("\nSymbol Error Rate:")
    for k in range(TX_REPS):
        n_err = errs[k].sum()
        pct   = 100 * n_err / total_per_block
        errs_all += n_err
        print(f" Block {k+1}: {n_err}/{total_per_block} errors ({pct:.2f}%)")

    # 6) Overall
    pct_all = 100 * errs_all / total_all
    print(f" Overall : {errs_all}/{total_all} errors ({pct_all:.2f}%)\n")

# record_audio(480000)

SAMPLE_RATE, recording = read('rx_recording.wav')
SAMPLE_RATE, transmission = read("tx_sequence.wav")
pilot = np.load(PILOT_NPY)

chirp_up    = generate_chirp(F0, F1, CHIRP_LEN_S)
chirp_down  = generate_chirp(F1, F0, CHIRP_LEN_S)

# ------------------------------------------------------------
# 1.  First Pass (to estimate Delta f)
# ------------------------------------------------------------

print("---------- First Pass ----------")
print("Length of recording: ", len(recording))
sync = synchronise(recording, chirp_up, chirp_down)
print("Payload: ", sync[0], "Cut-off point 1: ", sync[1], "Cut-off point 2: ", sync[2])

compare_tx_rx(recording, sync[1], sync[2])

split_block = ofdm_blocks(sync[0])
freq_block = freq_domain(split_block)
print("Frequency block shape: ", freq_block.shape)

eq_block, channel, decoded_bits = adaptive_channel_equalise(freq_block, pilot, method="mmse", noise_var=1e-4, n_mean=1, do_decision_directed=True)
plot_channel(channel)

freq_block_corr, phis = remove_cpe(eq_block)
df_est = refine_cfo(phis)
print("Delta f estimate: ", df_est)

# ------------------------------------------------------------
# 2.  Second Pass (to remove CPE)
# ------------------------------------------------------------

n = np.arange(len(recording))
derotator = np.exp(-1j * 2* np.pi * df_est * n / FS)
dr_recording = np.asarray(recording * derotator).real

print("---------- Second Pass ----------")
print("Length of recording: ", len(recording))
re_sync = synchronise(dr_recording, chirp_up, chirp_down)
print("Payload: ", re_sync[0], "Cut-off point 1: ", re_sync[1], "Cut-off point 2: ", re_sync[2])

compare_tx_rx(dr_recording, re_sync[1], re_sync[2])

re_split_block = ofdm_blocks(re_sync[0])
re_freq_block = freq_domain(re_split_block)
print("Frequency block shape: ", re_freq_block.shape)

re_freq_block_corr, re_phis = remove_cpe(re_freq_block)
re_df_est = refine_cfo(re_phis)
print("Delta f estimate: ", re_df_est)

# ------------------------------------------------------------
# 3.  Kept the same
# ------------------------------------------------------------

re_eq_block, re_channel, re_decoded_bits = adaptive_channel_equalise(re_freq_block_corr, pilot, method="tikhonov", noise_var=1e-4, n_mean=1, do_decision_directed=True)
plot_channel(re_channel[-1])
print("Shape of equalised OFDM blocks: ", re_eq_block.shape)
spectrum_plot(dr_recording)
simple_constellation_plot(np.asarray(re_eq_block))
constellation_plot(np.asarray(re_eq_block))
symbol_error_rate(re_eq_block)