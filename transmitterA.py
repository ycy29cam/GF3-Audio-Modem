import numpy as np
import sounddevice as sd
import time
# --- (Include your generate_mls function here) ---
# def generate_mls(degree, taps=None, initial_state=None): ...
def generate_mls(degree, taps=None, initial_state=None):
    """
    Generates a Maximum Length Sequence (MLS) using a Linear Feedback Shift Register (LFSR).

    Args:
        degree (int): The degree of the generator polynomial (e.g., 10 for a 2^10 - 1 sequence).
        taps (list, optional): List of tap positions (1-indexed) for the LFSR.
                               If None, tries to use a common primitive polynomial.
        initial_state (np.ndarray, optional): The initial state of the shift register.
                                              If None, defaults to all ones.

    Returns:
        np.ndarray: The generated MLS sequence (0s and 1s).
        int: The length of the sequence (2^degree - 1).
    """
    length = 2 ** degree - 1

    default_taps = {
        4: [1],  # x^4 + x^1 + 1
        5: [2],  # x^5 + x^2 + 1
        6: [1],  # x^6 + x^1 + 1
        7: [1],  # x^7 + x^1 + 1 (or [3])
        8: [1, 2, 7],  # x^8 + x^7 + x^2 + x^1 + 1 (many options)
        9: [4],  # x^9 + x^4 + 1
        10: [3],  # x^10 + x^3 + 1
        11: [2],  # x^11 + x^2 + 1
        12: [1, 4, 6],  # x^12 + x^6 + x^4 + x^1 + 1
        13: [1, 3, 4],  # x^13 + x^4 + x^3 + x^1 + 1
        14: [1, 6, 10],  # x^14 + x^10 + x^6 + x^1 + 1 (check source for optimal)
        15: [1],  # x^15 + x^1 + 1
        16: [1, 3, 12]  # x^16 + x^12 + x^3 + x^1 + 1 (e.g., for AES's S-box field)
    }

    if taps is None:
        if degree in default_taps:
            taps_to_use = default_taps[degree]
            print(f"Using default taps for degree {degree}: {taps_to_use}")
        else:
            raise ValueError(f"No default taps for degree {degree}. Please provide taps.")
    else:
        taps_to_use = taps


    if initial_state is None:
        sr = np.ones(degree, dtype=int)  # Shift register, cannot be all zeros
    else:
        if len(initial_state) != degree:
            raise ValueError("Initial state length must match degree.")
        if np.all(initial_state == 0):
            raise ValueError("Initial state cannot be all zeros.")
        sr = np.array(initial_state, dtype=int)

    mls_sequence = np.zeros(length, dtype=int)


    actual_tap_indices = [degree - t for t in taps_to_use]  # 0-indexed

    print(
        f"Shift register of degree {degree}. Effective tap indices for XOR: {actual_tap_indices} (from sr[0]) plus always sr[0] for feedback.")

    for i in range(length):
        mls_sequence[i] = sr[-1]  # Output bit (from the "rightmost" stage)

        # Calculate feedback bit (Fibonacci LFSR with external XOR)
        feedback_bit = sr[0]  # This is for the x^D term (e.g. sr[degree-degree])
        for tap_idx in actual_tap_indices:
            feedback_bit ^= sr[tap_idx]

        # Shift register to the right, new bit comes in at sr[0]
        sr[1:] = sr[:-1]
        sr[0] = feedback_bit

    return mls_sequence, length



SAMPLERATE = 48000
MLS_DEGREE = 12
SAMPLES_PER_CHIP = 1
SILENCE_BEFORE_SEC = 0.5
SILENCE_AFTER_SEC = 2.0 # This silence is part of the played signal
                        # The receiver needs to record for longer than this

# --- Generate MLS and Probe Signal (same as before) ---
mls_binary, mls_len = generate_mls(MLS_DEGREE)
mls_bipolar = (mls_binary * 2 - 1).astype(np.float32)
if SAMPLES_PER_CHIP > 1:
    mls_bipolar_stretched = np.repeat(mls_bipolar, SAMPLES_PER_CHIP)
else:
    mls_bipolar_stretched = mls_bipolar

# THIS IS THE KERNEL THE RECEIVER NEEDS
correlation_kernel_to_save = mls_bipolar_stretched[:mls_len * SAMPLES_PER_CHIP]
np.save('mls_kernel_for_receiver.npy', correlation_kernel_to_save)
print(f"MLS kernel saved to 'mls_kernel_for_receiver.npy'. Transfer this to Laptop B.")

silence_before_samples = int(SILENCE_BEFORE_SEC * SAMPLERATE)
silence_after_samples = int(SILENCE_AFTER_SEC * SAMPLERATE) # Silence in the played signal
probe_signal_core = mls_bipolar_stretched
probe_signal = np.concatenate([
    np.zeros(silence_before_samples, dtype=np.float32),
    probe_signal_core,
    np.zeros(silence_after_samples, dtype=np.float32)
])
probe_signal /= (np.max(np.abs(probe_signal)) / 0.8)

print(f"Total probe signal duration to be played: {len(probe_signal) / SAMPLERATE:.2f} seconds")
input("Press Enter on Laptop A to PLAY the probe signal...")
sd.play(probe_signal, SAMPLERATE, blocking=True)
print("Playback finished on Laptop A.")