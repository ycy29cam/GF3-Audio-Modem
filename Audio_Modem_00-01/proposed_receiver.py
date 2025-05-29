from Constants import *
from emitter import *
from scipy.io.wavfile import read
labels = np.load("labels.npy")
pilot = np.load("pilot_no_zero.npy")
print("Shape of pilot: ", pilot.shape)


def sync(recording, chirp_start = None, chirp_end = None):  # use cross correlation function on emitted and recieved signal to pick out start and end points
    if chirp_start is None:
        chirp_start = sequence_generator()[1]
    if chirp_end is None: 
        chirp_end = sequence_generator()[2]
    print("Recording shape in sync: ", np.asarray(recording).shape)
    channel_1 = recording[int(SAMPLE_RATE*1):] # extracts signal recieved from audio channel 1 for processing, removes starting
    # artifacts by removing 1s of recording
    sync_start = scipy.signal.correlate(channel_1,chirp_start, mode='full', method='auto')
    sync_end = scipy.signal.correlate(channel_1,chirp_end, mode='full', method='auto')
    start_bin = np.argmax(sync_start)
    end_bin = np.argmax(sync_end)
    return start_bin, end_bin, sync_start, sync_end

def Chopped_array(recorded_array = None):
    """
    truncates recorded signal by removing chirps and returning a chopped array of prediced sequence blocks
    accomodates non equal block lengths using block_length + also removes prefixes if there are any
    :return chopped_array: nested list of OFDM symbols - used for time domain
    
    if recorded_array == None:
        recorded_array = recording
    """
    recorded_array = recorded_array[start_bin:end_bin - len(chirp_end)]# effectively removes start chirp + end chirp

    print("Shape of recording: ", np.asarray(recorded_array).shape)

    chopped_array = []

    for i in block_lengths:
         print("Block length: ", i)

    #further down the line will need to correct for drift, and recompute synchronised position - probably OK over 3 blocks though
    cumulative_block_lengths = np.cumsum(block_lengths) # creates a sum of lengths for easy indexing
    cumulative_block_lengths = np.insert(cumulative_block_lengths, 0, 0)

    for j in cumulative_block_lengths:
         print("Cumulative sum: ", j)

    for i in range(len(block_lengths)):
        block = recorded_array[cumulative_block_lengths[i]:cumulative_block_lengths[i+1]]
        if block_type_ids[i] == 2: #add other prefix indices
             block = block[CP:]
        else:
             pass
        print("Length of OFDM block: ", len(block))
        chopped_array.append(block)
    return chopped_array

def compute_freq_symbols(chopped_array): #finds frequencies of a single chopped block
    frequency_array = []
    for i in chopped_array:
         print("Shape of i in chopped array: ", np.asarray(i).shape)
         freq_block_with_conjugates = scipy.fft.fft(i, n=8192)
         freq_block = freq_block_with_conjugates[:length] # takes only relevant conjugate part, assuming all have the same symbol length
         frequency_array.append(freq_block)
         print("No symmetry: ", len(freq_block))
    return frequency_array  

def transmission_visualisation(sequence: np.array, label: np.array):
    # Assume that sequence is always 2D
    
    if np.asarray(sequence).shape[1] != labels.shape[0]:
         raise ValueError("The decoded block length doesn't match the label length")

    points = np.asarray(sequence).flatten()
    extend_labels = np.repeat(labels, np.asarray(sequence).shape[0])

    print("Length of labels: ", len(extend_labels))
    print("Length of symbols: ", len(points))

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(points.real, points.imag, c=extend_labels, edgecolors='k', alpha=0.7)

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("In-Phase (Real)")
    plt.ylabel("Quadrature (Imag)")
    plt.title("Complex Symbols Colored by Column Labels")
    plt.grid(True)
    # plt.gca().set_aspect('equal')

    plt.show()
    return

SAMPLE_RATE, recording = read('signal_recorded.wav')
print(SAMPLE_RATE)
start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)
signal, chirp_start, chirp_end, block_lengths, length, block_type_ids, BLOCK_TYPES, pattern_signal  = sequence_generator()
chopped_array = Chopped_array(recording)

# --------------------------------------------------------------------------------------------------------------

frequency_array = np.asarray(compute_freq_symbols(chopped_array))
mean_array = np.mean(np.stack(frequency_array, axis=0), axis=0)
channel_1 = frequency_array[0] / pilot
channel = mean_array / pilot

# --------------------------------------------------------------------------------------------------------------

print("Mean array shape: ", mean_array.shape)
print(chopped_array[0])
print("Shape of frequency array (FFT'ed)", np.asarray(compute_freq_symbols(chopped_array)).shape)
plt.plot(recording) # right, basically issue of loudness, ask tomorrow
plt.show()

plt.plot(channel_1)
plt.show()

"""

# --------------------------------------------------------------------------------------------------------------

transmission_visualisation(compute_freq_symbols(chopped_array), labels)

# --------------------------------------------------------------------------------------------------------------

points = frequency_array.flatten()
extend_labels = np.repeat(labels, frequency_array.shape[0])
extend_channel = np.repeat(channel, frequency_array.shape[0])

points /= extend_channel

plt.figure(figsize=(6, 6))

scatter = plt.scatter(points.real, points.imag, c=extend_labels, edgecolors='k', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("In-Phase (Real)")
plt.ylabel("Quadrature (Imag)")
plt.title("Complex Symbols Colored by Column Labels")
plt.grid(True)

plt.show()
"""