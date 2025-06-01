import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.io.wavfile import read

import transmitter as tx          # <-- reuse every constant & helper!
#add a from reciver import * here, to avoid duplicate copies of helpers in reciever

FS            = tx.FS
FFT_LEN       = tx.FFT_LEN
CP_LEN        = tx.CP_LEN
CHIRP_LEN_S   = tx.CHIRP_LEN_S
SILENCE_S     = tx.SILENCE_LEN_S
F0, F1        = tx.F0, tx.F1
TX_REPS       = tx.TX_REPS

# -----------------------------------------------------------------
# helpers copied from the receiver section of transmitter.py
# -----------------------------------------------------------------
def synchronise(rx:np.ndarray,
                chirp_up:np.ndarray,
                chirp_dn:np.ndarray):
    """Return (payload, start_idx, end_idx)."""
    cu = signal.correlate(rx, chirp_up, mode='valid')
    p_up = np.argmax(cu)

    cd = signal.correlate(rx, chirp_dn, mode='valid')
    scr = p_up + len(chirp_up)
    cand = np.where(cd > 0.8*cd.max())[0]
    p_dn = cand[cand>scr][0]

    start = p_up + len(chirp_up)
    end   = p_dn
    return rx[start:end], start, end

def split_ofdm_blocks(payload):
    exp = CP_LEN + TX_REPS*FFT_LEN
    if len(payload) != exp:
        raise RuntimeError(f"Payload {len(payload)} ≠ {exp}")
    blks, idx = [], CP_LEN
    for _ in range(TX_REPS):
        blks.append(payload[idx:idx+FFT_LEN]); idx += FFT_LEN
    return np.stack(blks)

def freq_domain(blks):
    return fft.fft(blks, axis=1)[:,1:FFT_LEN//2]

# -----------------------------------------------------------------
# Test-1  : self-correlation of a chirp
# -----------------------------------------------------------------
def test_chirp_edge():
    chirp_td = tx.generate_chirp(F0, F1, CHIRP_LEN_S)
    corr     = signal.correlate(chirp_td, chirp_td, mode='full')
    # shift (peak index) relative to reference
    shift    = np.argmax(corr) - (len(chirp_td)-1)

    print("\n[TEST-1] Chirp self-correlation")
    print(f"  expected peak shift : 0 samples")
    print(f"  measured peak shift : {shift} samples")

    plt.figure(figsize=(8,2.5))
    plt.plot(chirp_td, lw=.6, label='chirp')
    plt.axvline(0,  c='g', lw=2, label='true start')
    plt.axvline(shift, c='r', ls='--', label='corr. peak')
    plt.title("Chirp and correlation peak (Test-1)")
    plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------------------
# Test-2  : synchronisation on a shifted TX waveform
# -----------------------------------------------------------------
def test_waveform_sync():
    # --- build the canonical TX waveform once ---
    wav_dict     = tx.prepare_tx_sequence()
    base_wave    = wav_dict['waveform']
    silence_len  = int(SILENCE_S*FS)
    chirp_len    = int(CHIRP_LEN_S*FS)

    # deterministic ground-truth indices in *unshifted* waveform
    gt_up_start  = silence_len
    gt_dn_start  = silence_len + chirp_len + CP_LEN + TX_REPS*FFT_LEN

    # --- apply arbitrary integer shift ---------------------------
    shift_by     = 13720           # choose any non-trivial offset
    shifted      = np.concatenate([np.zeros(shift_by, base_wave.dtype),
                                   base_wave])
    chirp_up     = tx.generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_dn     = tx.generate_chirp(F1, F0, CHIRP_LEN_S)

    _payload, st, ed = synchronise(shifted, chirp_up, chirp_dn)

    print("\n[TEST-2] Full-waveform synchronisation")
    print("  shift applied             :", shift_by)
    print("  true up-chirp start index :", gt_up_start + shift_by)
    print("  detected up-chirp start   :", st - chirp_len)
    print("  true down-chirp start     :", gt_dn_start + shift_by)
    print("  detected down-chirp start :", ed)

    # --- visual check --------------------------------------------
    plt.figure(figsize=(10,2.5))
    plt.plot(shifted, lw=.4, label='shifted TX')
    plt.axvline(gt_up_start + shift_by, c='g', lw=1.8, label='GT up-chirp')
    plt.axvline(gt_dn_start + shift_by, c='g', lw=1.8, label='GT down-chirp')
    plt.axvline(st - chirp_len, c='r', ls='--', label='detected up')
    plt.axvline(ed,            c='r', ls='--', label='detected down')
    plt.title("Synchronisation on shifted sequence (Test-2)")
    plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------------------
# Test-3  : channel-estimation accuracy
# -----------------------------------------------------------------
def test_channel_estimate():
    # build clean TX waveform & OFDM pilot
    wav_dict   = tx.prepare_tx_sequence()
    tx_wave    = wav_dict['waveform']
    pilot      = np.load(tx.PILOT_NPY)

    # --- create a synthetic channel impulse response --------------
    L          = 256                          # taps
    h_time     = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2*L)

    # convolve (linear) and keep same dtype
    rx_clean   = np.convolve(tx_wave.astype(np.complex64), h_time, mode='full')

    # add same shift for realism
    shift_by   = 25000
    rx_shifted = np.concatenate([np.zeros(shift_by, rx_clean.dtype),
                                 rx_clean]).real.astype(np.float32)

    # --- run synchroniser & FFT pipeline --------------------------
    chirp_up   = tx.generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_dn   = tx.generate_chirp(F1, F0, CHIRP_LEN_S)

    payload, _, _ = synchronise(rx_shifted, chirp_up, chirp_dn)
    blocks_td     = split_ofdm_blocks(payload)
    rx_fd         = freq_domain(blocks_td)
    H_est         = rx_fd[0] / pilot           # simple ZF

    # ground-truth frequency response
    H_true = fft.fft(h_time, FFT_LEN)[1:FFT_LEN//2]

    mse = np.mean(np.abs(H_true - H_est)**2)

    print("\n[TEST-3] Channel-estimation")
    print(f"  impulse-response length      : {L} taps")
    print(f"  synthetic time-shift         : {shift_by} samples")
    print(f"  mean-square-error (|H-Ĥ|²)    : {mse:.4e}")

    # --- plot magnitude responses --------------------------------
    plt.figure(figsize=(8,3))
    plt.plot(np.abs(H_true), label='|H_true|')
    plt.plot(np.abs(H_est),  label='|H_est|', ls='--')
    plt.title("Channel magnitude response (Test-3)")
    plt.xlabel("sub-carrier"); plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------------------
# RUN all three tests
# -----------------------------------------------------------------
if __name__ == '__main__':
    test_chirp_edge()
    test_waveform_sync()
    test_channel_estimate()