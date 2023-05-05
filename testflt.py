#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pysound as ps
import cfilters
from timeit_helper import timeit
import cProfile


def gen(*items):
    result = []
    for it in items:
        for x in it:
            result.append(x.copy())
    return np.concatenate(result)


saw = ps.poly_saw(0.5)
sq = ps.poly_square()

data = gen(saw(300) for _ in range(10))
plt.plot(data)

data = gen(sq(300) for _ in range(10))
plt.plot(data)

def filter_im_res():
    z = np.zeros(ps.BUFSIZE, dtype=np.float32)
    impulse = z.copy()
    impulse[20] = 1.0
    parts = [impulse, *([z]*10)]

    lp = ps.lowpass()
    # lp = ps.pdvcf()
    # lp = ps.moog()
    # lp = ps.flt12()

    cutoff = 0.7
    res = 0.3
    ff = ps.lowpass
    # ff = ps.flt12
    # ff = ps.moog

    def response_res(cutoff):
        data = gen(parts)
        plt.plot(*ps.fft_plot(data, crop=1))

        for res in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            lp = ff()
            data = gen(lp(it, cutoff, res) for it in parts)
            plt.plot(*ps.fft_plot(data, crop=1), label=str(res))


    def response_f(res):
        data = gen(parts)
        plt.plot(*ps.fft_plot(data, crop=1))

        for cutoff in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1):
            lp = ff()
            data = gen(lp(it, cutoff, res) for it in parts)
            plt.plot(*ps.fft_plot(data, crop=1), label=str(cutoff))

    response_res(0.4)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper left')

plt.show()
