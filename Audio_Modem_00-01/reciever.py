from Constants import *
from emitter import *


def sync(recording, chirp_start = None, chirp_end = None):  # use cross correlation function on emitted and recieved signal to pick out start and end
    if chirp_start is None:
        chirp_start = sequence_generator()[1]
    if chirp_end is None: 
        chirp_end = sequence_generator()[2]
    channel_1 = recording[:, 0] # extracts signal recieved from audio channel 1 for processing
    sync_start = scipy.signal.correlate(channel_1,chirp_start, mode='full', method='auto')
    sync_end = scipy.signal.correlate(channel_1,chirp_end, mode='full', method='auto')
    start_bin = np.argmax(sync_start)
    end_bin = np.argmax(sync_end)
    return start_bin, end_bin, sync_start, sync_end

# find a way to save sequence file to phone, and potentially add filller to aviod noise cancellation artifacts
# find an app to avoid noise cancellation artifacts and check phone emission frequency to match 48KHz or 44.1 KHz


# chop blocks into correct lengths, and compute DFT
# plot each frequency decoded block, assigning a different colour to each decoded symbol as to what the symbol should
# be from the input (i.e. yellow = 1 + i was meant to be recieved, ect...)
# use ML estimation to estimate H(w) and then use a weiner filter to make it better with some good value of SNR

# don't worry too much about LPDC yet, it will come with time

"-------------------------------Recording code-----------------------------------"

if __name__ == "__main__":
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2) # records audio from 2 channels, time = DURATION
    sd.wait()
    start_bin, end_bin, sync_start_data, sync_end_data = sync(recording) # Capture the outputs from sync


    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(sync_start_data), label='Start Chirp Correlation', color='blue')
    plt.plot(np.abs(sync_end_data), label='End Chirp Correlation', color='red')

    # Mark the start peak with a green circle
    plt.plot(start_bin, np.abs(sync_start_data[start_bin]), 'go', markersize=8, label=f'Start Peak (Bin: {start_bin})')
    # Mark the end peak with a magenta square
    plt.plot(end_bin, np.abs(sync_end_data[end_bin]), 'ms', markersize=8, label=f'End Peak (Bin: {end_bin})')
    plt.title('Recorded Plot Correlation')
    plt.xlabel('Sample Index (Lag)') # Changed for accuracy
    plt.ylabel('Absolute Amplitude') # Changed for accuracy
    plt.grid(True)
    plt.legend() # Show the legend
    plt.tight_layout()
    plt.show()
    
