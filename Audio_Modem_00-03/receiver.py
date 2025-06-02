import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal, fft
from transmitter import generate_chirp, WAV_TX, Q_COL

# ------------------------------------------------------------
#                      PARAMETERS
# ------------------------------------------------------------
FS              = 48_000
FFT_LEN         = 8192
CP_LEN          = FFT_LEN // 4
CHIRP_LEN_S     = 2.0
SILENCE_LEN_S   = 1.0
F0,  F1         = 20, 15000

# OFDM structure (must match transmitter)
TOTAL_BLOCKS    = 5
CP_FLAGS        = [True, False, True, True, False]      # CP? for each block
PILOT_IDXS      = [0, 1, 3, 4]                          # pilot blocks
DATA_IDX        = 2

# I/O
WAV_RX          = "rx_recording.wav"
PILOT_NPY       = "pilot_symbols.npy"
COLMAP_NPY      = "colour_map.npy"
DATA_SYMS_NPY   = "data_symbols.npy"
DATA_COLMAP_NPY = "data_colour_map.npy"
CHAN_NPY        = "channel_estimate.npy"

LENGTH_TOL      = 512

# ------------------------------------------------------------
#                      HELPERS
# ------------------------------------------------------------
def load_wav(path):
    sig, sr = sf.read(path, always_2d=False)
    assert sr == FS, "sample-rate mismatch"
    return sig.astype(np.float32)

def synchronise(rx, chirp_up, chirp_down):
    """Return payload (chirps trimmed) + start & end indices in rx."""
    corr_up   = signal.correlate(rx, chirp_up,   mode="valid")
    start     = np.argmax(corr_up) + len(chirp_up)

    corr_down = signal.correlate(rx, chirp_down, mode="valid")
    peak_down = np.where(corr_down > 0.8*corr_down.max())[0]
    peak_down = peak_down[peak_down > start][0]          # first good hit
    end       = peak_down

    payload   = rx[start:end]

    exp = FFT_LEN*TOTAL_BLOCKS + CP_LEN*sum(CP_FLAGS)
    if len(payload) > exp + LENGTH_TOL:
        payload = payload[:exp]
    elif len(payload) < exp - LENGTH_TOL:
        raise RuntimeError(f"payload {len(payload)} << expected {exp}")
    elif len(payload) < exp:
        payload = np.pad(payload, (0, exp-len(payload)))
    return payload, start, end

def ofdm_blocks(payload):
    """Strip CPs according to CP_FLAGS → (TOTAL_BLOCKS, FFT_LEN)."""
    blocks, idx = [], 0
    for cp in CP_FLAGS:
        if cp:
            idx += CP_LEN          # skip prefix
        blocks.append(payload[idx:idx+FFT_LEN])
        idx += FFT_LEN
    return np.stack(blocks)

def freq_domain(blocks_td):
    return fft.fft(blocks_td, axis=1)[:, 1:FFT_LEN//2]

def remove_cpe(rx_fd):
    corr, phis = [], []
    for sym in rx_fd:
        phi = .25*np.angle(np.mean(sym**4))
        phis.append(phi)
        corr.append(sym*np.exp(-1j*phi))
    return np.stack(corr), np.array(phis)

def refine_cfo(phis):
    k = np.arange(len(phis))
    slope = np.polyfit(k, np.unwrap(phis), 1)[0]
    return slope / (2*np.pi*(FFT_LEN/FS))

# ---------------- Channel / EQ -----------------
def channel_estimate(pilot_fd, pilot_ref, noise_var=1e-4):
    """Element-wise average of the 4 pilot blocks, MMSE-like regularisation."""
    Y   = np.mean(pilot_fd, axis=0)
    eps = 1e-12
    H_zf  = Y / (pilot_ref + eps)
    Rhh   = np.mean(np.abs(H_zf)**2)
    H_hat = (Rhh / (Rhh + noise_var)) * H_zf
    np.save(CHAN_NPY, H_hat)
    return H_hat

def equalise(z, H):          return z / H
def _norm(z):                return z / (np.sqrt(np.mean(np.abs(z)**2))+1e-12)

# ---------------- Visuals ----------------------
def plot_channel(H):
    fig, ax = plt.subplots(2,1,figsize=(9,4), sharex=True)
    ax[0].plot(20*np.log10(np.abs(H)+1e-12)); ax[0].set_ylabel("|H| [dB]")
    ax[1].plot(np.angle(H));                  ax[1].set_ylabel("∠H [rad]")
    ax[1].set_xlabel("sub-carrier"); ax[0].set_title("Estimated channel")
    plt.tight_layout(); plt.show()

def _colour_means(z, colours):
    return {c: np.mean(z[colours==c]) for c in np.unique(colours)}

def plot_constellation(z, colours, title="Constellation"):
    z = _norm(z)
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(z.real, z.imag, c=colours, s=12, edgecolors='none', alpha=.83)

    means = _colour_means(z, colours)
    for c, m in means.items():
        plt.plot(m.real, m.imag, 'kx')

    # legend ­–– colour → nominal symbol
    try:
        inv = {v:k for k,v in Q_COL.items()}
        label = {'00':'-1-1j','01':'-1+1j','11':'1+1j','10':'1-1j'}
        patches = [Patch(facecolor=c, label=label[inv[c]]) for c in means]
        plt.legend(handles=patches, fontsize='small')
    except Exception:
        pass

    plt.title(title); plt.xlabel("I"); plt.ylabel("Q")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

# ---------------- SER helpers ------------------
def ser_pilots(eq_pilot_fd):
    ref_syms  = np.load(PILOT_NPY)           # (tones,)
    colours   = np.load(COLMAP_NPY, allow_pickle=True)
    if colours.ndim==1 and len(colours)==1 and isinstance(colours[0],(list,np.ndarray)):
        colours = np.asarray(colours[0])

    tones     = ref_syms.size
    ref_bits  = np.column_stack((ref_syms.real>0, ref_syms.imag>0)).astype(int)

    eq_pilot_fd = eq_pilot_fd.reshape(-1, tones)
    for k, blk in enumerate(eq_pilot_fd):
        dec_bits = np.column_stack((blk.real>0, blk.imag>0)).astype(int)
        n_err    = np.any(dec_bits != ref_bits, axis=1).sum()
        print(f"Pilot block {k+1} SER: {n_err}/{tones}  ({100*n_err/tones:.2f}%)")

def ser_data(eq_data_fd):
    ref_syms = np.load(DATA_SYMS_NPY)
    tones    = ref_syms.size
    ref_bits = np.column_stack((ref_syms.real>0, ref_syms.imag>0)).astype(int)
    dec_bits = np.column_stack((eq_data_fd.real>0, eq_data_fd.imag>0)).astype(int)
    n_err    = np.any(dec_bits != ref_bits, axis=1).sum()
    print(f"DATA block SER : {n_err}/{tones}  ({100*n_err/tones:.2f}%)")

# ------------------------------------------------------------
#                      MAIN WORK-FLOW
# ------------------------------------------------------------
recording     = load_wav(WAV_RX)

chirp_up      = generate_chirp(F0,  F1, CHIRP_LEN_S)
chirp_down    = generate_chirp(F1,  F0, CHIRP_LEN_S)

# ---- Pass-1 : crude CFO estimate ---------------------------------
pay0, s0, e0  = synchronise(recording, chirp_up, chirp_down)
blk0_td       = ofdm_blocks(pay0)
blk0_fd       = freq_domain(blk0_td)
blk0_fd_corr, phis0 = remove_cpe(blk0_fd)
df_est        = refine_cfo(phis0)

# ---- Pass-2 : derotate & process ---------------------------------
n             = np.arange(recording.size)
derot         = np.exp(-1j*2*np.pi*df_est*n/FS)
rec_rot       = (recording * derot).real

pay1, s1, e1  = synchronise(rec_rot, chirp_up, chirp_down)
blk1_td       = ofdm_blocks(pay1)
blk1_fd       = freq_domain(blk1_td)
blk1_fd_corr, _ = remove_cpe(blk1_fd)

# ---- Channel estimation (4 pilots) -------------------------------
pilot_ref     = np.load(PILOT_NPY)
H_est         = channel_estimate(blk1_fd_corr[PILOT_IDXS], pilot_ref)
plot_channel(H_est)

# ---- Equalisation ------------------------------------------------
eq_pilots_fd  = equalise(blk1_fd_corr[PILOT_IDXS], H_est)
eq_data_fd    = equalise(blk1_fd_corr[DATA_IDX],  H_est)

# ---- Results -----------------------------------------------------
print("\n----------------  Symbol-Error-Rates  ----------------")
ser_pilots(eq_pilots_fd)
ser_data(_norm(eq_data_fd))

# ---- Visualise DATA constellation --------------------------------
data_col      = np.load(DATA_COLMAP_NPY, allow_pickle=True)
if data_col.ndim==1 and len(data_col)==1 and isinstance(data_col[0],(list,np.ndarray)):
    data_col = np.asarray(data_col[0])
plot_constellation(eq_data_fd, data_col, title="Equalised DATA constellation")
