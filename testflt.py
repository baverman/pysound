#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pysound import FREQ, square_unlim_t, osc, fft_plot
from timeit_helper import timeit

sig = osc(square_unlim_t)(np.full(2048, 300.0))
# sig = np.random.rand(2048)
# sig = sig - (np.max(sig) - np.min(sig)) / 2

def f(size, sig):
    window = np.hanning(size)
    window = window / window.sum()
    return np.convolve(window, sig, mode='same')


def lowpass_filter(dst, cutoff, acc=0):
    alpha = cutoff / 44100
    for i in range(len(dst)):
        acc += alpha * (dst[i] - acc)
        dst[i] = acc
    return dst


# timeit('f(20, sig)')

plt.subplot(311)
plt.plot(*fft_plot(sig, crop=5))

plt.subplot(312)
plt.plot(*fft_plot(lowpass_filter(sig, 10000), crop=5))

plt.subplot(313)
plt.plot(lowpass_filter(sig, 10000))

plt.show()
