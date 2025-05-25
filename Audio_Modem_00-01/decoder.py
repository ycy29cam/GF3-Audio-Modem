from emitter import *
from reciever import *


# processing prerecorded data    
SAMPLE_RATE, recording = read('signal_recorded.wav')
start_bin, end_bin, sync_start_data, sync_end_data = sync(recording)

# chop blocks into correct lengths, and compute DFT
# plot each frequency decoded block, assigning a different colour to each decoded symbol as to what the symbol should
# be from the input (i.e. yellow = 1 + i was meant to be recieved, ect...)
# use ML estimation to estimate H(w) and then use a weiner filter to make it better with some good value of SNR
# don't worry too much about LPDC yet, it will come with time