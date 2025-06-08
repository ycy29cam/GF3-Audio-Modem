# gf3_ldpc_testbench.py

from __future__ import annotations
import math, numpy as np, scipy.signal as sig, matplotlib.pyplot as plt
from pathlib import Path

# ---------- LDPC library --------------------------------------------------
import ldpc_jossy.py.ldpc as ldpc
CODE = ldpc.code(standard="802.11n", rate="1/2", z=81)  # K=972, N=1944

############################################################################
# Utility helpers                                                           #
############################################################################

def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """Gray‑coded QPSK (unit power)"""
    b = bits.reshape(-1, 2)
    return ((1 - 2*b[:, 0]) + 1j*(1 - 2*b[:, 1])) / np.sqrt(2)

def qpsk_demod(sym: np.ndarray) -> np.ndarray:
    return np.vstack([(sym.real < 0), (sym.imag < 0)]).astype(np.uint8).T.reshape(-1)

def qpsk_llr(sym: np.ndarray, sigma2: float) -> np.ndarray:
    L_I = 2*sym.real / sigma2
    L_Q = 2*sym.imag / sigma2
    return np.vstack([L_I, L_Q]).T.reshape(-1)

############################################################################
# Pilot (FULL 8192‑length, with conjugate symmetry)                         #
############################################################################
FFT_LEN = 8192
CP_LEN  = FFT_LEN // 4
DATA_IDXS = np.arange(200, 2144)                 # 1944 tones
_rng = np.random.default_rng(0)

# --- build pilot freq vector ---
_pilot_bits = _rng.integers(0, 2, DATA_IDXS.size*2, dtype=np.uint8)
_pilot_syms = qpsk_mod(_pilot_bits)              # 1944 QPSK symbols
PILOT_FREQ  = np.zeros(FFT_LEN, dtype=np.complex128)
PILOT_FREQ[DATA_IDXS] = _pilot_syms
# fill remaining positive‑freq bins (exclude DC, Nyquist)
all_pos = np.arange(1, FFT_LEN//2)
unused  = np.setdiff1d(all_pos, DATA_IDXS, assume_unique=True)
PILOT_FREQ[unused] = _rng.choice([1+0j, -1+0j, 1j, -1j], size=unused.size)
# impose conjugate symmetry
a = PILOT_FREQ
PILOT_FREQ[FFT_LEN//2+1:] = np.conj(a[FFT_LEN//2-1:0:-1])
PILOT_FREQ[0] = 0  # DC
PILOT_FREQ[FFT_LEN//2] = 0  # Nyquist

############################################################################
# LDPC wrappers                                                             #
############################################################################
class LDPCEncoder:
    def __init__(self, c: ldpc.code = CODE):
        self.c = c
    def encode_codewords(self, bits: np.ndarray) -> np.ndarray:
        bits = bits.astype(np.uint8)
        pad = (-bits.size) % self.c.K
        if pad:
            bits = np.hstack([bits, np.zeros(pad, np.uint8)])
        blocks = bits.reshape(-1, self.c.K)
        return np.vstack([self.c.encode(b) for b in blocks])

class LDPCDecoder:
    def __init__(self, c: ldpc.code = CODE):
        self.c = c
    def decode(self, llrs: np.ndarray) -> np.ndarray:
        llrs = llrs.reshape(-1, self.c.N)
        out = []
        for y in llrs:
            app,_ = self.c.decode(y, 'sumprod2')
            out.append((app < 0).astype(np.uint8)[: self.c.K])
        return np.hstack(out)

############################################################################
# Transmitter                                                               #
############################################################################
class Transmitter:
    def __init__(self, enc: LDPCEncoder):
        self.enc = enc
    # ------------------------------------------------------------------
    def _data_block_fd(self, two_cw_bits: np.ndarray) -> np.ndarray:
        assert two_cw_bits.size == 2*CODE.N
        freq = np.zeros(FFT_LEN, complex)
        freq[DATA_IDXS] = qpsk_mod(two_cw_bits)
        # randomise other positive‑freq bins
        pos = np.arange(1, FFT_LEN//2)
        unused = np.setdiff1d(pos, DATA_IDXS, assume_unique=True)
        freq[unused] = _rng.choice([1, -1, 1j, -1j], size=unused.size)
        freq[FFT_LEN//2+1:] = np.conj(freq[FFT_LEN//2-1:0:-1])
        return freq
    # ------------------------------------------------------------------
    def build_frames(self, payload_bits: np.ndarray) -> np.ndarray:
        cws = self.enc.encode_codewords(payload_bits)
        pairs = []
        for i in range(0, len(cws), 2):
            if i+1 < len(cws):
                pairs.append(np.hstack([cws[i], cws[i+1]]))
            else:
                pairs.append(np.hstack([cws[i], np.zeros_like(cws[i])]))
        blocks_td = []
        idx = 0
        while idx < len(pairs):
            # Pilot
            pilot_td = np.fft.ifft(PILOT_FREQ) * np.sqrt(FFT_LEN)
            blocks_td.append(np.hstack([pilot_td[-CP_LEN:], pilot_td]))
            # Four data blocks
            for _ in range(4):
                if idx < len(pairs):
                    fd = self._data_block_fd(pairs[idx])
                else:
                    fd = np.zeros_like(PILOT_FREQ)
                td = np.fft.ifft(fd) * np.sqrt(FFT_LEN)
                blocks_td.append(np.hstack([td[-CP_LEN:], td]))
                idx += 1
        return np.hstack(blocks_td)

############################################################################
# Channel                                                                   #
############################################################################

def random_channel(L: int = 64) -> np.ndarray:
    taps = (_rng.standard_normal(L) + 1j*_rng.standard_normal(L))
    taps *= np.exp(-np.arange(L)/10)
    taps /= np.linalg.norm(taps)
    return taps

def apply_channel(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return sig.fftconvolve(x, h)[: x.size]

############################################################################
# Receiver                                                                  #
############################################################################
class Receiver:
    def __init__(self, dec: LDPCDecoder):
        self.dec = dec
    def _blocks(self, sig: np.ndarray):
        blen = FFT_LEN + CP_LEN
        for i in range(0, sig.size, blen):
            yield sig[i:i+blen]
    # ------------------------------------------------------------------
    def process(self, rx_sig: np.ndarray):
        blocks = list(self._blocks(rx_sig))
        info = []
        blen = FFT_LEN + CP_LEN
        assert len(blocks) % 5 == 0
        for g in range(0, len(blocks), 5):
            # Pilot estimation
            pilot_fd = np.fft.fft(blocks[g][CP_LEN:]) / np.sqrt(FFT_LEN)
            H_est = np.zeros_like(pilot_fd)
            nz = PILOT_FREQ != 0
            H_est[nz] = pilot_fd[nz] / PILOT_FREQ[nz]
            sigma2 = np.mean(np.abs(pilot_fd[nz] - PILOT_FREQ[nz]*H_est[nz])**2)
            # Data blocks
            for d in range(1,5):
                data_fd = np.fft.fft(blocks[g+d][CP_LEN:]) / np.sqrt(FFT_LEN)
                eq = data_fd[DATA_IDXS] / H_est[DATA_IDXS]
                llrs = qpsk_llr(eq, sigma2)
                if not np.any(llrs):
                    continue
                info.append(self.dec.decode(llrs))
        return np.hstack(info) if info else np.array([], np.uint8)

############################################################################
# Main                                                                      #
############################################################################

def main():
    N_INFO = CODE.K * 200
    payload = _rng.integers(0,2,N_INFO,np.uint8)

    tx = Transmitter(LDPCEncoder())
    tx_sig = tx.build_frames(payload)
    h = random_channel()
    rx_sig = apply_channel(tx_sig, h)
    rx_bits = Receiver(LDPCDecoder()).process(rx_sig)[:N_INFO]

    errs = np.count_nonzero(payload ^ rx_bits)
    print(f"Post‑LDPC BER: {errs}/{N_INFO} = {errs/N_INFO:.2e}")

    # quick plots
    plt.figure(); plt.title("|H| on data carriers")
    H = np.fft.fft(h, FFT_LEN)
    plt.plot(np.abs(H)[DATA_IDXS]); plt.grid(); plt.show()

if __name__ == "__main__":
    main()
