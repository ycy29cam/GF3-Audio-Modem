import numpy as np

def ofdm_constellation_mapper(binary_array, conjugate_symmetry=False, apply_fft=False):
    if len(binary_array) % 2 != 0:
        raise ValueError("Binary array length must be even for QPSK mapping.")

    # Gray-coded QPSK mapping: '00'→-1-j, '01'→-1+j, '11'→1+j, '10'→1-j
    mapping = {
        (0, 0): -1 - 1j,
        (0, 1): -1 + 1j,
        (1, 1):  1 + 1j,
        (1, 0):  1 - 1j
    }

    bits_reshaped = binary_array.reshape(-1, 2)
    symbols = np.array([mapping[tuple(b)] for b in bits_reshaped], dtype=np.complex128)

    if conjugate_symmetry:
        N = len(symbols) * 2 + 2
        symm_freq = np.zeros(N, dtype=np.complex128)
        symm_freq[1:N//2] = symbols
        symm_freq[N//2+1:] = np.conj(symbols[::-1])
        # 0th and middle frequencies remain 0
        sequence = symm_freq
    else:
        sequence = symbols

    if apply_fft:
        signal = np.fft.ifft(sequence)
        print("Sequence is in the time domain.")
    else:
        signal = sequence
        print("Sequence is in the frequency domain.")

    if np.allclose(signal.imag, 0, atol=1e-10):
        print("Sequence is real-valued.")
    else:
        print("Sequence is complex-valued.")

    return signal
