from Constants import *
from emitter import *
from scipy.io.wavfile import read

def sync(recording, chirp_start=None, chirp_end=None):
    if chirp_start is None:
        chirp_start = sequence_generator()[1]
    if chirp_end   is None:
        chirp_end   = sequence_generator()[2]

    # Peel off the first second
    if recording.ndim == 1:
        y = recording[int(SAMPLE_RATE * 1):]
    else:
        y = recording[int(SAMPLE_RATE * 1):, 0]

    sync_start_full =  scipy.signal.correlate(y, chirp_start, mode='valid')
    sync_end_full =  scipy.signal.correlate(y, chirp_end, mode='valid')

    start_bin_true = np.argmax(sync_start_full)  # start within y
    end_bin_true = np.argmax(sync_end_full)

    start_bin = int(SAMPLE_RATE * 1) + start_bin_true
    end_bin = int(SAMPLE_RATE * 1) + end_bin_true

    return start_bin, end_bin, sync_start_full, sync_end_full


"-------------------------------Recording code-----------------------------------"

if __name__ == "__main__":
    """
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2) # records audio from 2 channels, time = DURATION
    #recording, fs = sf.read('frame.wav')
    sd.wait()
    start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)
    write('signal_recorded_2.wav', SAMPLE_RATE, recording)
    """

    # Longer frame
    RECORD_DUR = 12  # e.g. 12 s for a ~7 s frame
    print(f"Recording for {RECORD_DUR}s. Hit Play on your phone now...")
    recording = sd.rec(int(RECORD_DUR * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)
    sd.wait()
    print("Done recording—running sync() to locate frame…")

    # 2) Locate start/end via chirp correlation
    start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)
    print(f"Chirp detected from sample {start_bin} to {end_bin}.")
    print(f"Transmission length {end_bin - start_bin}.")

    # 3) Extract just the frame
    payload = recording[start_bin: end_bin]

    # 4) Save and hand to your decoder
    write('frame_extracted.wav', SAMPLE_RATE, payload)
    print("Saved frame_extracted.wav—ready for OFDM decoding.")

    # Construct an x‐axis so that sync_start_data[i] plots at sample index i + fs*1
    xs = np.arange(len(sync_start_data)) + int(SAMPLE_RATE * 1)

    # find the peaks in the correlation arrays
    i_start = np.argmax(sync_start_data)
    i_end   = np.argmax(sync_end_data)

    # Now plot against xs instead of 0…N_corr-1
    plt.figure(figsize=(10, 4))
    plt.plot(xs, np.abs(sync_start_data), label='Start Chirp Correlation', color='blue')
    plt.plot(xs, np.abs(sync_end_data), label='End   Chirp Correlation', color='red')

    # Mark the peaks at the *absolute* indices start_bin/end_bin
    plt.plot(start_bin, np.abs(sync_start_data[i_start]), 'go', markersize=8,
             label=f'Start Peak (Bin: {start_bin})')
    plt.plot(end_bin, np.abs(sync_end_data[i_end]), 'ms', markersize=8,
             label=f'End   Peak (Bin: {end_bin})')

    plt.title('Recorded Plot Correlation')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
