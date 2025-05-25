from Constants import *
from emitter import *
from scipy.io.wavfile import read


def sync(recording, chirp_start = None, chirp_end = None):  # use cross correlation function on emitted and recieved signal to pick out start and end points
    if chirp_start is None:
        chirp_start = sequence_generator()[1]
    if chirp_end is None: 
        chirp_end = sequence_generator()[2]
    channel_1 = recording[int(SAMPLE_RATE*1):, 0] # extracts signal recieved from audio channel 1 for processing, removes starting
    # artifacts by removing 1s of recording
    sync_start = scipy.signal.correlate(channel_1,chirp_start, mode='full', method='auto')
    sync_end = scipy.signal.correlate(channel_1,chirp_end, mode='full', method='auto')
    start_bin = np.argmax(sync_start)
    end_bin = np.argmax(sync_end)
    return start_bin, end_bin, sync_start, sync_end

"-------------------------------Recording code-----------------------------------"

if __name__ == "__main__":
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2) # records audio from 2 channels, time = DURATION
    sd.wait()
    start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)
    write('signal_recorded_2.wav', SAMPLE_RATE, recording)


# synchronisation plot -  use HiBy music to play sound from phone
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(sync_start_data), label='Start Chirp Correlation', color='blue')
    plt.plot(np.abs(sync_end_data), label='End Chirp Correlation', color='red') # Mark the start peak with a green circle
    plt.plot(start_bin, np.abs(sync_start_data[start_bin]), 'go', markersize=8, label=f'Start Peak (Bin: {start_bin})')
    plt.plot(end_bin, np.abs(sync_end_data[end_bin]), 'ms', markersize=8, label=f'End Peak (Bin: {end_bin})') # Mark the end peak with a magenta square
    plt.title('Recorded Plot Correlation')
    plt.xlabel('Sample Index (Lag)')
    plt.ylabel('Absolute Amplitude') 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()