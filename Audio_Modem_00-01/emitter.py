import Constants
import numpy as np
import scipy
import sounddevice as sd
import time
import matplotlib.pyplot as plt


def chirp(frequency, duration, fs,):
    """
    Generate a chirp signal.
    :param frequency: Frequency of the chirp in Hz
    :param duration: Duration of the chirp in seconds
    :param fs: Sampling frequency in Hz
    :return: Chirp signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return scipy.signal.chirp(t, f0=frequency, f1=frequency * 2, t1=duration, method='linear') * 0.5

def OFDM_pilot():
    return True