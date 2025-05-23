import numpy as np
import matplotlib.pyplot as plt
m=100
scale = 1
freq = np.fft.fftfreq(2*m, d=scale)
print (freq)

plt.plot(freq)
plt.title('FFT Frequencies')
plt.show()