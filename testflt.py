#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pysound import FREQ, square_unlim_t, osc, fft_plot, env_ahr, fps, dcfilter, sps, noise, lowpass, sin_t
from pysound import cdcfilter, ffidcfilter
from timeit_helper import timeit
import cProfile

def dcblock():
    mean = 0
    def do(sig):
        nonlocal mean
        newmean = np.mean(sig)
        sig = sig - newmean
        mean = newmean
        return sig
    return do


def delay(max_duration=0.5):
    buf = np.full(sps(max_duration), 0, dtype=np.float32)
    def process(sig, delay, feedback):
        size = len(sig)
        shift = sps(delay)
        buf[:-size] = buf[size:]
        buf[-size:] = sig * feedback
        buf[-size:] += buf[-size-shift:-shift]
        # plt.plot(buf[-size:].copy())
        # plt.show()
        return buf[-size:].copy()
    return process


def delay_slow(max_duration=0.5):
    buf = np.full(sps(max_duration), 0, dtype=np.float32)
    def process(sig, delay, feedback):
        size = len(sig)
        shift = sps(delay)
        buf[:-size] = buf[size:]
        for i, v in enumerate(sig, len(buf)-size):
            buf[i] = buf[i-shift] + v * feedback
        return buf[-size:].copy()
    return process


o = osc(sin_t)
env = env_ahr(0, 30, 0)
d = delay_slow()
f = lowpass()
sig = np.concatenate([d(noise()*env(), 0.011, 0.99) for _ in range(fps(0.1))])

ss = sig[:512]

flt1 = dcfilter()
flt2 = cdcfilter()
flt3 = ffidcfilter()

# timeit('flt1(ss)')
# timeit('flt2(ss)')
# timeit('flt3(ss)')

# def boo():
#     for _ in range(100000):
#         flt2(ss)
#
# from ctypes import addressof
# cProfile.run("boo()", sort='tottime')

# ss = o(440)
#
figure, axis = plt.subplots(3, 1)

axis[0].plot(flt1(ss))
axis[1].plot(flt3(ss))
axis[2].plot(ss)

plt.show()
