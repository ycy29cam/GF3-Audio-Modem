import numpy as np
import scipy.signal
import sounddevice as sd
import matplotlib.pyplot as plt
import time

SAMPLERATE = 48000 # Must match transmitter
# SILENCE_AFTER_SEC_RECORDING determines how much reverberation to capture AFTER the probe ends
SILENCE_AFTER_SEC_RECORDING = 2.5 # Should be >= SILENCE_AFTER_SEC on transmitter + margin

# --- Load the kernel (transfer this file from Laptop A) ---
try:
    correlation_kernel = np.load('mls_kernel_for_receiver.npy')
    print("MLS kernel loaded.")
except FileNotFoundError:
    print("ERROR: 'mls_kernel_for_receiver.npy' not found. Please transfer it from Laptop A.")
    exit()

# Estimate duration needed for recording
# Duration of the actual MLS part of the probe signal
probe_core_duration_sec = len(correlation_kernel) / SAMPLERATE


duration_of_played_probe_approx = (0.5 + # SILENCE_BEFORE_SEC from TX
                                  len(correlation_kernel)/SAMPLERATE +
                                  2.0)   # SILENCE_AFTER_SEC from TX

# Total recording duration on Laptop B:
# Enough to capture the full played signal from A, plus extra tail, plus manual sync buffer
recording_duration_sec = duration_of_played_probe_approx + SILENCE_AFTER_SEC_RECORDING + 2.0 # Extra 2s buffer

print(f"Will record for approximately {recording_duration_sec:.2f} seconds on Laptop B.")
input("Press Enter on Laptop B to START RECORDING (then quickly start playback on Laptop A)...")

recorded_signal = sd.rec(int(recording_duration_sec * SAMPLERATE),
                         samplerate=SAMPLERATE,
                         channels=1, # Record mono
                         blocking=True)
recorded_signal = recorded_signal.flatten()
print("Recording complete on Laptop B.")

# --- Save recorded signal for analysis ---
np.savez('measurement_from_B.npz',
         recorded_signal=recorded_signal,
         samplerate=SAMPLERATE,
         correlation_kernel=correlation_kernel)
print("Recorded signal and kernel saved to 'measurement_from_B.npz'.")
print("Now you can run the analysis.")

# --- Analysis (can be in a separate script or here) ---
print("Starting analysis...")
impulse_response_raw = scipy.signal.correlate(recorded_signal, correlation_kernel, mode='full')
impulse_response_scaled = impulse_response_raw / np.max(np.abs(impulse_response_raw))

# Plotting (simplified, adapt from your original script)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(recorded_signal)) / SAMPLERATE, recorded_signal)
plt.title('Recorded Signal (Laptop B)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.subplot(2, 1, 2)
# Determine a good segment to plot for IR
ir_meaningful_start_index = len(correlation_kernel) -1
ir_plot_segment = impulse_response_scaled[ir_meaningful_start_index : ir_meaningful_start_index + int((SILENCE_AFTER_SEC_RECORDING + 0.2) * SAMPLERATE)]
time_axis_segment = np.arange(len(ir_plot_segment)) / SAMPLERATE
plt.plot(time_axis_segment, ir_plot_segment)
plt.title('Estimated Impulse Response (Scaled)')
plt.xlabel('Time (s from estimated IR start)')
plt.grid(True)

plt.tight_layout()
plt.show()