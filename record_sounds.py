import numpy as np
import sounddevice as sd
import soundfile as sf
FS                   = 48_000
FFT_LEN              = 8192  
WAV_RX_1             = 'rx_recording_group1.wav'
WAV_RX_2             = 'rx_recording_group2.wav'
WAV_RX_3             = 'rx_recording_group3.wav'
WAV_RX_4             = 'rx_recording_group4.wav'
WAV_RX_5             = 'rx_recording_group5.wav'
WAV_RX_5             = 'rx_recording_group6.wav'
WAV_RX_7             = 'rx_recording_group7.wav'

def record_audio(expected_len: int, fs: int = FS) -> np.ndarray:
    print(f"Recording ≈{expected_len / fs:.2f} s …")
    rec = sd.rec(expected_len, samplerate=fs, channels=1,
                 dtype='float32').squeeze()  # removes extra unused dimension
    sd.wait()
    sf.write(WAV_RX_1, rec, fs)
    return rec

record_audio(20*FS)