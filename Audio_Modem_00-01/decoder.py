from emitter import *
from reciever import *


def Chopped_array(recorded_array = None):
    """
    truncates recorded signal by removing chirps and returning a chopped array of prediced sequence blocks
    accomodates non equal block lengths using block_length + also removes prefixes if there are any
    :return chopped_array: nested list of OFDM symbols - used for time domain
    """
    if recorded_array == None:
        recorded_array = recording
    recorded_array = recorded_array[start_bin:end_bin - len(chirp_end)]# effectively removes start chirp + end chirp
    
    chopped_array = []
    #further down the line will need to correct for drift, and recompute synchronised position - probably OK over 3 blocks though
    cumulative_block_lengths = np.cumsum(block_lengths) # creates a sum of lengths for easy indexing
    cumulative_block_lengths = np.insert(cumulative_block_lengths, 0, 0)
    for i in range(len(block_lengths)):
        block = signal[cumulative_block_lengths[i-1]:cumulative_block_lengths[i]]
        if block_type_ids[i] == 2:
             block = block[CP:]
        else:
             pass
        chopped_array.append(block)
    return chopped_array



def compute_freq_symbols(chopped_array): #finds frequencies of a single chopped block
    frequency_array = []
    for i in chopped_array:
         freq_block_with_conjugates = np.fft.fft(i)
         freq_block = freq_block_with_conjugates[:length] # takes only relevant conjugate part, assuming all have the same symbol length
         frequency_array.append(freq_block)
    return frequency_array

def channel_estimation(frequency_array):
    red = []
    yellow = []
    green = []
    blue = []
    rows, cols = frequency_array.shape

    for i in range(rows):
        for j in range(cols):
            z = frequency_array[i, j]
            if z == complex(1, 1):
                red.append(z)
            elif z == complex(1, -1):
                yellow.append(z)
            elif z == complex(-1, -1):
                green.append(z)
            elif z == complex(1, -1):  # Now explicitly check for blue
                blue.append(z)
            else:
                raise ValueError(f"Unexpected complex value at ({i}, {j}): {z}")

    return red, yellow, green, blue



#and compute DFT
# plot each frequency decoded block, assigning a different colour to each decoded symbol as to what the symbol should
# be from the input (i.e. yellow = 1 + i was meant to be recieved, ect...)
# use ML estimation to estimate H(w) and then use a weiner filter to make it better with some good value of SNR
# don't worry too much about LPDC yet, it will come with time

"-------------------------------Channel estimation-----------------------------------"

if __name__ == "__main__":
        # processing prerecorded data
        # do some import bullshit tomorrow to reduce the amount of renamed variables    
        SAMPLE_RATE, recording = read('signal_recorded_2.wav')
        start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)
        signal, chirp_start, chirp_end, block_lengths, length, block_type_ids, BLOCK_TYPES, pattern_signal  = sequence_generator()
        chopped_array = Chopped_array()
        plt.plot(recording) # right, basically issue of loudness, ask tomorrow
        plt.show()
        # frequency_array = compute_freq_symbols(chopped_array)
        # symbol_map = OFDM_pilot()[1]
        # print(channel_estimation(frequency_array))


