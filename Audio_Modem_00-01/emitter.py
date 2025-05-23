import Constants
import numpy as np
import scipy
import sounddevice as sd
import time
import matplotlib.pyplot as plt
from Constants import *

def chirp(fstart,fend, duration, fs,):
    """
    Generate a chirp signal.
    :param fstart: Start frequency of the chirp in Hz
    :param fend: end frequency of the chirt in Hz
    :param duration: Duration of the chirp in seconds
    :param fs: Sampling frequency in Hz
    :return: Chirp signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return scipy.signal.chirp(t, f0=fstart, f1=fend, t1=duration, method='linear') * 0.5

def OFDM_pilot(length):
    """
    Generates a random array of constellation symbols
    Uses a fixed seed for random generation
    4 symbol grey code mapping
    :param length: Length of pilot in symbols
    :return: random array of complex QPSK symbols in freq domain
    """
    np.random.seed(seed=42)
    B_matrix = np.random.randint(2, size=(length,2))
    symbol_map = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j
    }
    return np.array([symbol_map[tuple(b)] for b in B_matrix])

def noise(length):
    np.random.seed(seed=42)
    noise = np.random.normal(0, 1, length).astype('float32')
    return noise

"-------------------------------Emission Code-----------------------------------"

sd.play(chirp(100,20000,2,SAMPLE_RATE), samplerate=SAMPLE_RATE)
#sd.play(noise(BLOCK_LENGTH*20), samplerate=SAMPLE_RATE)
sd.wait()