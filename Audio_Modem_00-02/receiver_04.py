"""
Receiver for the new waveform:
    • Removes up-/down-chirps (down-chirp now has its own prefix).
    • Parses five OFDM blocks (each preceded by CP).
    • Two-pass CPE & CFO removal retained.
    • Channel = element-wise average of the **four** pilot blocks.
    • Equalises pilots & data, plots channel and DATA constellation,
      and prints SER separately for pilots & data.
"""

# ------------------------------------------------------------
# 0.  Imports & run-time parameters
# ------------------------------------------------------------
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from transmitter_02 import generate_chirp, Q_COL          # re-use helpers

FS,  FFT_LEN, CP_LEN = 48_000, 8192, 8192//4
CHIRP_LEN_S, SILENCE_LEN_S = 2.0, 1.0
F0,  F1 = 20, 15000

# waveform structure
TOTAL_BLOCKS   = 5
CP_FLAGS       = [True]*TOTAL_BLOCKS
PILOT_IDXS     = [0,1,3,4]
DATA_IDX       = 2

# files
WAV_RX          = "rx_recording.wav"
PILOT_NPY       = "pilot_symbols.npy"
COLMAP_NPY      = "colour_map.npy"
DATA_SYMS_NPY   = "data_symbols.npy"
DATA_COLMAP_NPY = "data_colour_map.npy"
CHAN_NPY        = "channel_estimate.npy"

LENGTH_TOL      = 512   # samples


# ------------------------------------------------------------
# 1.  Very small helpers
# ------------------------------------------------------------
def load_wav(path):
    sig, sr = sf.read(path, always_2d=False)
    assert sr == FS, "Unexpected sample-rate"
    return sig.astype(np.float32)


def synchronise(rx, chirp_up, chirp_down):
    """Return payload (CP+5 OFDM) and its [start,end) indices in rx."""
    corr_up   = signal.correlate(rx, chirp_up, mode='valid')
    start_up  = np.argmax(corr_up)                     # up-chirp start
    pay_start = start_up + len(chirp_up)               # after up-chirp

    corr_dn   = signal.correlate(rx, chirp_down, mode='valid')
    # first good peak *after* the OFDM payload
    after = pay_start + (CP_LEN+FFT_LEN)*TOTAL_BLOCKS
    start_dn = np.where(corr_dn > 0.8*corr_dn.max())[0]
    start_dn = start_dn[start_dn > after][0]           # down-chirp start
    pay_end  = start_dn - CP_LEN                       # remove prefix before dn

    payload  = rx[pay_start:pay_end]

    exp = TOTAL_BLOCKS * (CP_LEN + FFT_LEN)
    if len(payload) > exp + LENGTH_TOL:
        payload = payload[:exp]
    elif len(payload) < exp - LENGTH_TOL:
        raise RuntimeError("Payload shorter than expected")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))
    return payload, pay_start, pay_end


def ofdm_blocks(payload):
    """Strip prefixes according to CP_FLAGS → (TOTAL_BLOCKS, FFT_LEN)."""
    blocks, idx = [], 0
    for cp in CP_FLAGS:
        if cp:
            idx += CP_LEN
        blocks.append(payload[idx:idx+FFT_LEN])
        idx += FFT_LEN
    return np.stack(blocks).astype(np.float32)


def freq_domain(blks_td):          # (B,N/2-1)
    return fft.fft(blks_td, axis=1)[:, 1:FFT_LEN//2]


# ------------------- CFO / CPE helpers ----------------------
def remove_cpe(rx_fd):
    new, phis = [], []
    for sym in rx_fd:
        phi = .25*np.angle(np.mean(sym**4))
        phis.append(phi)
        new.append(sym*np.exp(-1j*phi))
    return np.stack(new), np.array(phis)


def refine_cfo(phis):
    k = np.arange(len(phis))
    slope = np.polyfit(k, np.unwrap(phis), 1)[0]
    return slope / (2*np.pi*(FFT_LEN/FS))


# ------------------- Channel & EQ ---------------------------
def channel_estimate(pilot_fd, ref_pilot, noise_var=1e-4):
    """Average *four* pilots element-wise → MMSE-like H."""
    eps  = 1e-12
    Y    = pilot_fd.mean(axis=0)
    H_zf = Y / (ref_pilot + eps)
    Rhh  = np.mean(np.abs(H_zf)**2)
    H    = (Rhh / (Rhh + noise_var)) * H_zf
    np.save(CHAN_NPY, H)
    return H


def equalise(z, H):               return z / H
def _norm(z):                     return z / (np.sqrt(np.mean(np.abs(z)**2))+1e-12)


# ------------------- Visuals --------------------------------
def plot_channel(H):
    f, ax = plt.subplots(2,1,figsize=(8,4), sharex=True)
    ax[0].plot(20*np.log10(np.abs(H)+1e-12)); ax[0].set_ylabel("|H| [dB]")
    ax[1].plot(np.angle(H));                  ax[1].set_ylabel("∠H [rad]")
    ax[1].set_xlabel("sub-carrier"); ax[0].set_title("Estimated channel")
    plt.tight_layout(); plt.show()


def plot_constellation(z, colours, title):
    z = _norm(z).ravel(); colours = colours.ravel()
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(z.real, z.imag, c=colours, s=12, edgecolors='none', alpha=.82)
    means = {c: z[colours==c].mean() for c in np.unique(colours)}
    for c,m in means.items():
        plt.plot(m.real, m.imag, 'kx')
    # legend (colour→symbol)
    inv = {v:k for k,v in Q_COL.items()}
    lbl = {'00':'-1-1j','01':'-1+1j','11':'1+1j','10':'1-1j'}
    pats = [Patch(facecolor=c,label=lbl.get(inv.get(c,''),'')) for c in means]
    plt.legend(handles=pats, fontsize='small')
    plt.title(title); plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()


# ------------------- SER helpers ----------------------------
def ser_block(rx_syms, ref_syms):
    ref_bits = np.column_stack((ref_syms.real>0, ref_syms.imag>0))
    dec_bits = np.column_stack((rx_syms.real>0,  rx_syms.imag>0))
    return np.any(dec_bits!=ref_bits, axis=1).sum()


# ------------------------------------------------------------
# 2.  Processing work-flow (runs immediately)
# ------------------------------------------------------------
recording   = load_wav(WAV_RX)
chirp_up    = generate_chirp(F0,  F1, CHIRP_LEN_S)
chirp_down  = generate_chirp(F1,  F0, CHIRP_LEN_S)

# ---- pass 1 -- CFO rough-estimate --------------------------
payload0, s0, e0 = synchronise(recording, chirp_up, chirp_down)
blk_td0          = ofdm_blocks(payload0)
blk_fd0          = freq_domain(blk_td0)
blk_fd0_corr, ph0= remove_cpe(blk_fd0)
df_est           = refine_cfo(ph0)

# ---- pass 2 -- derotate + re-extract -----------------------
n            = np.arange(recording.size)
derot        = np.exp(-1j*2*np.pi*df_est*n/FS)
rec_rot      = (recording*derot).real

payload1, s1, e1 = synchronise(rec_rot, chirp_up, chirp_down)
blk_td1          = ofdm_blocks(payload1)
blk_fd1          = freq_domain(blk_td1)
blk_fd1_corr, _  = remove_cpe(blk_fd1)

# ---- channel estimation -----------------------------------
pilot_ref    = np.load(PILOT_NPY)
H_est        = channel_estimate(blk_fd1_corr[PILOT_IDXS], pilot_ref)
plot_channel(H_est)

# ---- equalisation & SER -----------------------------------
eq_pilots_fd = equalise(blk_fd1_corr[PILOT_IDXS], H_est)
eq_data_fd   = equalise(blk_fd1_corr[DATA_IDX],  H_est)

pilot_ser = sum(ser_block(eq_pilots_fd[k], pilot_ref) for k in range(4))
data_ser  = ser_block(eq_data_fd, np.load(DATA_SYMS_NPY))
tones     = pilot_ref.size

print("\nSymbol-Error-Rate:")
for k,idx in enumerate(PILOT_IDXS,1):
    n_err = ser_block(eq_pilots_fd[k-1], pilot_ref)
    print(f" Pilot {idx+1}: {n_err}/{tones} ({100*n_err/tones:.2f} %)")
print(f" DATA     : {data_ser}/{tones} ({100*data_ser/tones:.2f} %)\n")

# ---- plot DATA constellation ------------------------------------
data_cols = np.load(DATA_COLMAP_NPY, allow_pickle=True).squeeze()
plot_constellation(eq_data_fd, data_cols, "Equalised DATA constellation")
