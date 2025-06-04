#!/usr/bin/env python3
# ------------------------------------------------------------
#  receiver_v2.py   (for the “0.5-s-chirp / 5 × [pilot,data]” format)
# ------------------------------------------------------------
import argparse, json, time, wave, pathlib
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from scipy.io.wavfile import read

# ---- import transmitter definitions (pilot-map colours etc.) ------------
try:
    from transmitter import generate_chirp, Q_COL, WAV_TX
except ImportError:                       # fallback for earlier filename
    from transmitter_01 import generate_chirp, Q_COL, WAV_TX

# ------------------------------------------------
# 1.  General parameters (ONLY LINES CHANGED ARE MARKED ###)
# ------------------------------------------------
FS              = 48_000
FFT_LEN         = 8192
CP_LEN          = FFT_LEN // 4
CHIRP_LEN_S     = 0.5          ###  old value was 2
SILENCE_LEN_S   = 1.0
F0, F1          = 20, 15_000
TX_PAIRS        = 5            ###  five  [pilot,data]  pairs
TOTAL_BLOCKS    = 2*TX_PAIRS   ###  =10 OFDM blocks in payload
WAV_RX          = 'rx_recording.wav'

PILOT_NPY       = 'pilot_symbols.npy'
DATA_NPY        = 'data_symbols.npy'
COLMAP_NPY      = 'colour_map.npy'
CHAN_NPY        = 'channel_estimate.npy'

CHIRP_ATTEN     = 0.80
TARGET_PEAK     = 0.80
LENGTH_TOL      = 512

# ------------------------------------------------
# 2.  I/O helpers  (unchanged)
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
# 3.  Synchronisation  (unchanged – works with new chirp length)
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

    exp = TOTAL_BLOCKS * (FFT_LEN + CP_LEN)   ### expected length
    if len(payload) > exp + LENGTH_TOL:
        payload = payload[:exp]
    elif len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))

    return payload, start_payload, end_payload

# ------------------------------------------------
# 4.  OFDM helpers
#     *NEW* ofdm_blocks can cope with CP in every block
# ------------------------------------------------
def ofdm_blocks(payload_td:np.ndarray) -> np.ndarray:
    """
    Slice the time-domain payload into TOTAL_BLOCKS blocks,
    stripping the CP from *each* block.
    """
    blocks = []
    idx = 0
    for _ in range(TOTAL_BLOCKS):
        blk = payload_td[idx + CP_LEN : idx + CP_LEN + FFT_LEN]
        blocks.append(blk)
        idx += FFT_LEN + CP_LEN
    return np.stack(blocks)

def freq_domain(blocks_td:np.ndarray) -> np.ndarray:
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2]

#  CPE / CFO helpers are unchanged – omitted for brevity
#  (remove_cpe, refine_cfo, adaptive_channel_equalise, _normalise_symbols)
#  ↳ keep identical to original receiver

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

#  ----------------------------------------------------------
#  5.  Visualisation helpers – we add variants for pilot/data
#  ----------------------------------------------------------
def constellation_plot_by_type(eq_blocks_fd:np.ndarray,
                               ref_syms:list[np.ndarray],
                               types:list[str],
                               title:str):
    """
    Scatter the equalised symbols, colour-coded by *reference* symbol.
    `ref_syms[k]` must correspond to block k.
    """
    tones = eq_blocks_fd.shape[1]
    # build colours using transmitter map
    inv_map = {v:k for k,v in Q_COL.items()}           # colour→bits
    bit2col = {k:v for v,k in inv_map.items()}         # bits→colour

    flat_eq   = []
    flat_col  = []
    for blk, ref in zip(eq_blocks_fd, ref_syms):
        for z, r in zip(blk, ref):
            bits = ((r.real>0), (r.imag>0))            # (b0,b1)
            flat_eq.append(z)
            flat_col.append(bit2col[bits])

    flat_eq  = np.asarray(flat_eq)
    flat_col = np.asarray(flat_col)

    # normalise energy to 1
    flat_eq /= np.sqrt(np.mean(np.abs(flat_eq)**2)) + 1e-12

    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(flat_eq.real, flat_eq.imag,
                c=flat_col, s=12, edgecolors='none', alpha=.82)
    plt.title(title); plt.xlabel('I'); plt.ylabel('Q')
    plt.gca().set_aspect('equal')
    plt.tight_layout(); plt.show()

def per_block_errors(eq_blocks_fd:np.ndarray,
                     pilot_sym:np.ndarray,
                     data_syms:np.ndarray):
    """
    Print symbol errors for each of the TOTAL_BLOCKS.
    `data_syms` is shape (TX_PAIRS, tones)
    """
    tones = pilot_sym.size
    err_total = 0
    for k in range(TOTAL_BLOCKS):
        is_pilot = (k % 2 == 0)
        ref = pilot_sym if is_pilot else data_syms[k//2]
        blk = eq_blocks_fd[k]

        dec_bits = np.empty((tones,2), int)
        dec_bits[:,0] = (blk.real > 0)
        dec_bits[:,1] = (blk.imag > 0)

        ref_bits = np.empty((tones,2), int)
        ref_bits[:,0] = (ref.real > 0)
        ref_bits[:,1] = (ref.imag > 0)

        errs = np.any(dec_bits != ref_bits, axis=1).sum()
        err_total += errs
        pct = 100*errs/tones
        kind = 'pilot' if is_pilot else 'data '
        print(f"Block {k+1:2d} ({kind}) : {errs:4d}/{tones}  ({pct:6.2f} %)")

    pct_total = 100*err_total/(tones*TOTAL_BLOCKS)
    print(f"\nOVERALL         : {err_total}/{tones*TOTAL_BLOCKS}  "
          f"({pct_total:6.2f} %)\n")

# ------------------------------------------------------------
# 6.  MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == '__main__':

    # ---------- load reference material ---------------------
    pilot_sym  = np.load(PILOT_NPY)                # (tones,)
    data_syms  = np.load(DATA_NPY)                 # (TX_PAIRS, tones)

    chirp_up   = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_down = generate_chirp(F1, F0, CHIRP_LEN_S)

    # ---------- acquire recording ---------------------------
    FS_R, recording = read(WAV_RX)                 # already recorded
    recording = recording.astype(np.float32)

    # ---------- FIRST PASS  --------------------------------
    print("\n----------- FIRST PASS (coarse CFO) -------------")
    payload_td, p0, p1 = synchronise(recording, chirp_up, chirp_down)
    blocks_td   = ofdm_blocks(payload_td)
    rx_fd       = freq_domain(blocks_td)

    eq_fd, H_est, dec_bits = adaptive_channel_equalise(
        rx_fd, pilot_sym,
        method='mmse', noise_var=1e-4,
        n_mean=1, do_decision_directed=True)

    eq_fd = eq_fd.reshape(TOTAL_BLOCKS, -1)        # back to (blocks,tones)

    # coarse Δf from per-symbol phase
    from collections import deque                # small helper
    phis = deque(maxlen=TOTAL_BLOCKS)
    for sym in eq_fd:
        est = np.mean(sym**4)
        phis.append(0.25*np.angle(est))
    slope = np.polyfit(np.arange(TOTAL_BLOCKS), np.unwrap(phis), 1)[0]
    df_est = slope / (2*np.pi*FFT_LEN/FS)
    print(f"  coarse Δf  ≈ {df_est:+.1f} Hz")

    # ---------- SECOND PASS (derotate full wave) ------------
    n = np.arange(len(recording))
    derot   = np.exp(-1j*2*np.pi*df_est*n/FS)
    rec_cpl = recording * derot
    rec_td  = rec_cpl.real.astype(np.float32)

    print("\n----------- SECOND PASS (CPE + final EQ) --------")
    payload_td, p0, p1 = synchronise(rec_td, chirp_up, chirp_down)
    blocks_td  = ofdm_blocks(payload_td)
    rx_fd      = freq_domain(blocks_td)

    # per-block common-phase removal
    rx_fd_corr, phis = [], []
    for sym in rx_fd:
        est = np.mean(sym**4)
        phi = 0.25*np.angle(est)
        rx_fd_corr.append(sym*np.exp(-1j*phi))
        phis.append(phi)
    rx_fd_corr = np.stack(rx_fd_corr)

    # -- adaptive channel, decision-directed on every block --
    eq_fd, H_est, _ = adaptive_channel_equalise(
        rx_fd_corr, pilot_sym,
        method='tikhonov', noise_var=1e-4,
        n_mean=1, do_decision_directed=True)

    eq_fd = eq_fd.reshape(TOTAL_BLOCKS, -1)

    # ---------- plots & error stats -------------------------
    #   * pilots (even indices)
    pli = [k for k in range(TOTAL_BLOCKS) if k%2==0]
    constellation_plot_by_type(eq_fd[pli],
                               [pilot_sym]*len(pli),
                               ['pilot']*len(pli),
                               "Pilot blocks – equalised constellation")

    #   * data  (odd indices)
    dli = [k for k in range(TOTAL_BLOCKS) if k%2==1]
    constellation_plot_by_type(eq_fd[dli],
                               data_syms,
                               ['data']*len(dli),
                               "Data blocks – equalised constellation")

    #   * per-block error report
    per_block_errors(eq_fd, pilot_sym, data_syms)
