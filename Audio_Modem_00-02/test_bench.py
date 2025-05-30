#!/usr/bin/env python3
# -------------------------------------------------------------
#   test_bench.py  –  dynamic test-suite for GF-3 audio modem
# -------------------------------------------------------------
"""
This script relies on *receiver.py* and *transmitter.py* being importable
from the same folder.  It automatically discovers the receiver’s helpers
so you do not have to edit the tests every time the modem evolves.
"""

from __future__ import annotations
import importlib
import argparse, types, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import pathlib

# ------------------------------------------------------------------
#  0.  IMPORT TRANSMITTER & RECEIVER  (receiver has priority)
# ------------------------------------------------------------------
rx: types.ModuleType = importlib.import_module('receiver')      # current DUT
tx: types.ModuleType = importlib.import_module('transmitter')   # reference TX

# ------------------------------------------------------------------
#  1.  RESOLVE CONSTANTS  (prefer receiver’s values)
# ------------------------------------------------------------------
def pick(name: str):
    return getattr(rx, name, getattr(tx, name))

FS           = pick('FS')
FFT_LEN      = pick('FFT_LEN')
CP_LEN       = pick('CP_LEN')
CHIRP_LEN_S  = pick('CHIRP_LEN_S')
SILENCE_S    = pick('SILENCE_LEN_S')
F0, F1       = pick('F0'), pick('F1')
TX_REPS      = pick('TX_REPS')
LENGTH_TOL   = pick('LENGTH_TOL') if hasattr(rx, 'LENGTH_TOL') else 0

# ------------------------------------------------------------------
#  2.  RESOLVE HELPER FUNCTIONS  (with graceful fallback)
# ------------------------------------------------------------------
def fn(module, name, fallback=None):
    """Get function <name> from <module> or return <fallback>."""
    f = getattr(module, name, None)
    if callable(f):
        return f
    if fallback is not None:
        return fallback
    raise AttributeError(f"Neither receiver nor transmitter defines {name}")

generate_chirp  = fn(rx, 'generate_chirp', tx.generate_chirp)
synchronise     = fn(rx, 'synchronise')
ofdm_blocks     = fn(rx, 'ofdm_blocks')
freq_domain     = fn(rx, 'freq_domain')
channel_estimate= fn(rx, 'channel_estimate')

# ------------------------------------------------------------------
#  3.  GENERIC UTILITIES
# ------------------------------------------------------------------
def build_reference_wave() -> tuple[np.ndarray, np.ndarray]:
    """
    Calls transmitter.prepare_tx_sequence() and returns:
         waveform, pilot_symbols
    It will overwrite pilot_symbols.npy, but that’s fine for testing.
    """
    pkt = tx.prepare_tx_sequence()
    wav = pkt['waveform'] if isinstance(pkt, dict) else pkt
    pilot = np.load(tx.PILOT_NPY)         # produced by transmitter
    return wav.astype(np.float32), pilot

def mse(a, b):
    return np.mean(np.abs(a-b)**2)

# ------------------------------------------------------------------
#  4.  TEST-CASES
# ------------------------------------------------------------------
def test1_chirp_edge():
    """Verify that auto-correlation peak is at zero lag."""
    chirp_td = generate_chirp(F0, F1, CHIRP_LEN_S)
    corr     = signal.correlate(chirp_td, chirp_td, mode='full')
    peak     = np.argmax(corr) - (len(chirp_td)-1)

    print("\n[TEST-1] Chirp self-correlation")
    print(f"expected peak shift : 0")
    print(f"measured peak shift : {peak}")

    plt.figure(figsize=(8,2.5))
    plt.plot(chirp_td, lw=.6, label='chirp')
    plt.axvline(0, c='g', lw=2, label='true start')
    plt.axvline(peak, c='r', ls='--', label='corr. peak')
    plt.title("Chirp & correlation peak (Test-1)")
    plt.legend(); plt.tight_layout(); plt.show()

def test2_waveform_sync():
    """
    Shift the whole TX waveform by an arbitrary offset, then let the
    receiver’s synchroniser find both chirps.
    """
    base_wave, _ = build_reference_wave()

    silence_len  = int(SILENCE_S*FS)
    chirp_len    = int(CHIRP_LEN_S*FS)
    gt_up_start  = silence_len
    gt_dn_start  = silence_len + chirp_len + CP_LEN + TX_REPS*FFT_LEN

    shift_by     = 13720
    shifted      = np.concatenate([np.zeros(shift_by, base_wave.dtype),
                                   base_wave])

    chirp_up, chirp_dn = generate_chirp(F0, F1, CHIRP_LEN_S), \
                         generate_chirp(F1, F0, CHIRP_LEN_S)

    payload, st, ed = synchronise(shifted, chirp_up, chirp_dn)

    print("\n[TEST-2] Waveform synchronisation")
    print(f"shift applied              : {shift_by}")
    print(f"true up-chirp start index  : {gt_up_start + shift_by}")
    print(f"detected up-chirp start    : {st - chirp_len}")
    print(f"true down-chirp start      : {gt_dn_start + shift_by}")
    print(f"detected down-chirp start  : {ed}")
    print(f"payload length (samples)   : {len(payload)}")

    plt.figure(figsize=(10,2.5))
    plt.plot(shifted, lw=.4, label='shifted TX')
    plt.axvline(gt_up_start + shift_by, c='g', lw=1.8, label='GT up-chirp')
    plt.axvline(gt_dn_start + shift_by, c='g', lw=1.8, label='GT down-chirp')
    plt.axvline(st - chirp_len, c='r', ls='--', label='detected up')
    plt.axvline(ed,            c='r', ls='--', label='detected down')
    plt.title("Synchronisation on shifted sequence (Test-2)")
    plt.legend(); plt.tight_layout(); plt.show()

def test3_channel_estimate():
    """
    Convolve the TX signal with a random (complex) channel, shift it in
    time, then let the receiver synchronise, estimate the channel and
    equalise.  Finally draw the constellation in quadrant colours.
    """
    # ---------- synthetic channel & received waveform ----------
    wav, pilot = build_reference_wave()

    L      = 256
    h_time = (np.random.randn(L) + 1j*np.random.randn(L)) / np.sqrt(2*L)
    rx_sig = np.convolve(wav.astype(np.complex64), h_time, mode='full')

    shift_by   = 25_000
    rx_shifted = np.concatenate([np.zeros(shift_by, rx_sig.dtype),
                                 rx_sig])                       # keep complex!

    # ---------- receiver front-end -----------------------------
    chirp_up   = generate_chirp(F0, F1, CHIRP_LEN_S)
    chirp_dn   = generate_chirp(F1, F0, CHIRP_LEN_S)

    payload, *_ = synchronise(rx_shifted.real.astype(np.float32),  # synchroniser is real
                              chirp_up, chirp_dn)

    blks_td     = ofdm_blocks(payload)
    rx_fd       = freq_domain(blks_td)                # (reps, tones)

    # ---------- channel estimate & equalisation ---------------
    H_est = channel_estimate(rx_fd, pilot).squeeze()
    H_true = fft.fft(h_time, FFT_LEN)[1:FFT_LEN//2]

    eq_fd = (rx_fd[1:] / H_est).flatten()             # skip pilot block

    # ---------- console read-out ------------------------------
    print("\n[TEST-3] Channel estimation")
    print(f"impulse-response length : {L} taps")
    print(f"time shift applied      : {shift_by} samples")
    print(f"MSE(|H|)                : {mse(H_true, H_est):.4e}")

    # ---------- magnitude response plot -----------------------
    plt.figure(figsize=(8,3))
    plt.plot(np.abs(H_true), label='|H_true|')
    plt.plot(np.abs(H_est),  label='|H_est|', ls='--')
    plt.title("Channel magnitude response (Test-3)")
    plt.xlabel("sub-carrier")
    plt.legend(); plt.tight_layout(); plt.show()

    # ---------- constellation plot ----------------------------
    base_col = np.load(tx.COLMAP_NPY)                 # Npilot colours
    colours  = np.tile(base_col, TX_REPS-1)           # cover all data blks
    plt.figure(); plt.axhline(0,c='k'); plt.axvline(0,c='k')
    plt.scatter(eq_fd.real, eq_fd.imag,
                c=colours[:len(eq_fd)], s=8, alpha=.85, edgecolors='none')
    plt.title("Equalised constellation (Test-3)")
    plt.gca().set_aspect('equal'); plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
#  5.  CLI WRAPPER
# ------------------------------------------------------------------
TESTS = {
    1: test1_chirp_edge,
    2: test2_waveform_sync,
    3: test3_channel_estimate,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, choices=[1,2,3],
                    help="run a single test (1/2/3) – default: run all")
    args = ap.parse_args()

    if args.test:
        TESTS[args.test]()
    else:
        for t in (1,2,3):
            TESTS[t]()

if __name__ == '__main__':
    main()
