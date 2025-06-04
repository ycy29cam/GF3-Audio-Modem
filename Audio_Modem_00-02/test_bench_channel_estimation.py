#!/usr/bin/env python3
# ------------------------------------------------------------
# Test-bench: compare ZF, MMSE and Tikhonov channel estimation
# on one known pilot OFDM block (N = 8192, CP = N/4).
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ---------- parameters -------------------------------------------------
N      = 8192                 # FFT size
CP     = N // 4               # cyclic prefix
tones  = N//2 - 1             # number of positive data sub-carriers
noise_var = 1e-4              # σ² used in MMSE / Tikh.

# ---------- helper: build conjugate-symmetric QPSK pilot ---------------
rng   = np.random.default_rng(42)
pilot_data = (2*rng.integers(0, 2, tones) - 1) \
           + 1j*(2*rng.integers(0, 2, tones) - 1)     # ±1 ± j

pilot_fd  = np.zeros(N, np.complex64)
pilot_fd[1: N//2]        = pilot_data               # +ve tones
pilot_fd[N//2+1: ]       = np.conj(pilot_data[::-1])  # –ve tones

# ---------- time-domain pilot (+CP) ------------------------------------
pilot_td = np.fft.ifft(pilot_fd).real.astype(np.float32)  # guaranteed real
tx_block = np.concatenate([pilot_td[-CP:], pilot_td])     # one CP+IFFT OFDM

# ---------- build a realistic (difficult) channel ----------------------
Lch = 256                                   # impulse length
exp_decay = np.exp(-np.arange(Lch)/80)      # 10-dB/80-tap decay
h_time = exp_decay * (rng.standard_normal(Lch)
                      + 1j*rng.standard_normal(Lch)) / np.sqrt(2*Lch)
h_time = h_time.astype(np.complex64)

# ---------- transmit through channel (+ AWGN) --------------------------
y_lin = np.convolve(tx_block, h_time, mode='full')
# take first CP+N samples (wrap-around effect OK because CP ≥ Lch)
y_rx  = y_lin[:CP+N]
y_rx  += np.sqrt(noise_var/2)*(rng.standard_normal(CP+N)
                               + 1j*rng.standard_normal(CP+N))

# ---------- receiver: remove CP, FFT -----------------------------------
y_no_cp = y_rx[CP:]
Y_fd    = np.fft.fft(y_no_cp)        # full N
Y_pos   = Y_fd[1:N//2]               # usable positive tones
X_pos   = pilot_data                 # true pilot

# ---------- true channel in freq domain -------------------------------
H_true_full = np.fft.fft(h_time, N)
H_true_pos  = H_true_full[1:N//2]

# ---------- estimator choices ------------------------------------------
eps = 1e-12
H_zf  = Y_pos / (X_pos + eps)

Rhh   = np.mean(np.abs(H_zf)**2)
H_mmse = (Rhh/(Rhh+noise_var)) * H_zf

denom = np.abs(X_pos)**2 + noise_var + eps
H_tikh = (np.conj(X_pos)*Y_pos) / denom

estimators = dict(ZF=H_zf, MMSE=H_mmse, Tikhonov=H_tikh)

# ---------- metrics ----------------------------------------------------
def mse(Hhat):   # mean-square error (unnormalised)
    return np.mean(np.abs(Hhat - H_true_pos)**2)

def nmse(Hhat):  # normalised MSE  (already used before)
    return mse(Hhat) / np.mean(np.abs(H_true_pos)**2)

def ser(eq_sym): # symbol-error rate after equalisation
    dec  = ((eq_sym.real > 0).astype(int),
            (eq_sym.imag > 0).astype(int))
    tru  = ((X_pos.real  > 0).astype(int),
            (X_pos.imag  > 0).astype(int))
    errs = np.any(np.vstack(dec) != np.vstack(tru), axis=0)
    return errs.mean()

# ---------- figure: channel magnitude & phase --------------------------
fig, ax = plt.subplots(2, 1, figsize=(9,4), sharex=True)
ax[0].plot(20*np.log10(np.abs(H_true_pos)+1e-12), 'k', lw=1.5, label='True')
ax[1].plot(np.angle(H_true_pos),                    'k', lw=1.5, label='True')

styles = dict(ZF='r--', MMSE='g-.', Tikhonov='b:')
for name, Hhat in estimators.items():
    ax[0].plot(20*np.log10(np.abs(Hhat)+1e-12), styles[name], label=name)
    ax[1].plot(np.angle(Hhat),                  styles[name], label=name)

ax[0].set_ylabel('|H| [dB]')
ax[1].set_ylabel('∠H [rad]'); ax[1].set_xlabel('Sub-carrier')
ax[0].legend(ncol=4, fontsize='small'); ax[1].legend(ncol=4, fontsize='small')
fig.suptitle('Channel magnitude & phase: True vs. Estimates')
plt.tight_layout(); plt.show()

# ---------- equalise, scatter & SER / MSE ------------------------------
colormap = dict(ZF='r', MMSE='g', Tikhonov='b')
plt.figure(figsize=(5,5))
plt.axhline(0,c='k'); plt.axvline(0,c='k')

for name, Hhat in estimators.items():
    S_eq = Y_pos / (Hhat + eps)
    mse_val  = mse(Hhat)
    nmse_val = nmse(Hhat)
    ser_val  = ser(S_eq)
    print(f'{name:9s}  MSE={mse_val:.4e}   NMSE={nmse_val:.4e}   SER={ser_val:.4f}')
    plt.scatter(S_eq.real, S_eq.imag, c=colormap[name], s=6, alpha=.6,
                label=f'{name}: SER={ser_val:.3f}')

# ideal constellation overlay
plt.scatter(X_pos.real, X_pos.imag, c='k', s=10, marker='x', label='Pilot (ideal)')
plt.gca().set_aspect('equal')
plt.title('Equalised constellation per estimator')
plt.xlabel('In-phase'); plt.ylabel('Quadrature')
plt.legend(fontsize='small'); plt.tight_layout(); plt.show()

