#PYTHONPATH=/home/bobrov/work python
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pysound as ps
from pysound import cfilters
from timeit_helper import timeit
import cProfile

from functools import partial, lru_cache


def gen(*items):
    result = []
    for it in items:
        for x in it:
            result.append(x.copy())
    return np.concatenate(result)


@lru_cache(None)
def delta():
    z = np.zeros(ps.BUFSIZE, dtype=np.float32)
    impulse = z.copy()
    impulse[20] = 1.0
    return [impulse, *([z]*10)]


def filter_response(flt, label=None):
    data = gen(flt(it) for it in delta())
    plt.plot(*ps.fft_plot(data, crop=1), label=label)


def spole():
    yp = 0
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        nonlocal yp
        x = math.exp(-6.28 * cutoff*10000/ps.FREQ)
        a = 1 - x
        b = x
        for i in range(len(sig)):
            yp = sig[i]*a + yp*b
            result[i] = yp
        return result
    return sgen


def stage4():
    y = [0.0, 0.0, 0.0, 0.0]
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        x = math.exp(-6.28 * cutoff/4)
        a0 = (1-x)**4
        b1 = 4*x
        b2 = -6*x*x
        b3= 4*x**3
        b4 =-(x**4)
        for i in range(len(sig)):
            yi = sig[i]*a0 + y[0]*b1 + y[1]*b2 + y[2]*b3 + y[3]*b4
            result[i] = yi
            y[3] = y[2]
            y[2] = y[1]
            y[1] = y[0]
            y[0] = yi
        return result
    return sgen


def biquad_calc(dst, src, state, a, b):
    x1, x2, y1, y2 = state
    for i in range(len(src)):
        y = dst[i] = (b[0]*src[i] + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2)/a[0]
        x2 = x1
        x1 = src[i]
        y2 = y1
        y1 = y
    state[0] = x1
    state[1] = x2
    state[2] = y1
    state[3] = y2


def biquad_lp():
    state = [0.0, 0.0, 0.0, 0.0]
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        q = 0.7 + res*4.0;
        f = 6.28*cutoff*10000/ps.FREQ
        sf = math.sin(f)
        cf = math.cos(f)
        alpha = sf/2/q
        a = 1+alpha, -2*cf, 1 - alpha
        b = (1-cf)/2, 1-cf, (1-cf)/2
        biquad_calc(result, sig, state, a, b)
        return result
    return sgen


def pole2():
    s1, s2 = 0.0, 0.0
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        nonlocal s1, s2
        R = 1 - res
        g = math.tan(cutoff*1.4)
        g = cutoff * 3.14;
        g1 = 2*R + g
        d = 1 / (1 + 2*R*g + g**2)
        for i in range(len(sig)):
            HP = (sig[i] - g1*s1 - s2) * d
            v1 = g*HP; BP = v1 + s1; s1 = BP + v1
            v2 = g*BP; LP = v2 + s2; s2 = LP + v2
            result[i] = LP
        return result
    return sgen


def pole1():
    s = [0.0]
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        g = math.tan(cutoff*1.4)
        g = cutoff * 3.14;
        # print(cutoff, g)
        G = g / (1 + g)
        for i in range(len(sig)):
            v = (sig[i] - s[0]) * G
            result[i] = v + s[0]
            s[0] = result[i] + v
        return result
    return sgen


def hipass():
    x1 = 0
    y1 = 0
    result = np.zeros(ps.BUFSIZE, dtype=np.float32)
    def sgen(sig, cutoff, res):
        nonlocal y1, x1
        x = math.exp(-6.28 * cutoff/ps.FREQ)
        a0 = (1+x)/2
        a1 = -a0
        b1 = x
        for i in range(len(sig)):
            y1 = sig[i]*a0 + x1*a1 + y1*b1
            x1 = sig[i]
            result[i] = y1
        return result
    return sgen


def filter_im_res():
    def response_res(flt, cutoff, min=0, max=1, step=0.1):
        data = gen(delta())
        plt.plot(*ps.fft_plot(data, crop=1), label='orig')

        res = min
        while res <= max:
            lp = flt(cutoff, res)
            filter_response(lp, label=str(round(res, 1)))
            res += step


    def response_f(flt, res):
        data = gen(delta())
        plt.plot(*ps.fft_plot(data, crop=1), label='orig')

        for cutoff in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9):
            lp = flt(cutoff, res)
            filter_response(lp, label=str(round(cutoff, 2)))


    def makeflt(flt_init, *args):
        def fltgen(cutoff, res):
            lp = flt_init(*args)
            def inner(sig):
                return lp(sig, cutoff, res)
            return inner
        return fltgen

    def set_axes():
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(0.00001, 0.01)
        plt.legend(loc='upper left')

    # plt.subplot(211)
    # response_res(makeflt(ps.bqlp), 0.4)
    # set_axes()
    #
    # plt.subplot(212)
    # response_res(makeflt(ps.lowpass), 0.4)
    # set_axes()

    # response_res(makeflt(ps.moog), 0.4, max=1)
    # response_f(makeflt(pole2), 0.4)
    response_f(makeflt(ps.pole2), 0.9)
    set_axes()

# env = ps.env_ahdsr(0.0)
# args = 5, 100, 0.8, 50
# data = gen(env(*args) for _ in range(ps.fps(0.2)))
# env.stop(True)
# data = gen([data], (env(*args) for _ in range(ps.fps(0.2))))
# env.trigger()
# data = gen([data], (env(*args) for _ in range(ps.fps(0.5))))
# env.stop()
# data = gen([data], (env(*args) for _ in range(ps.fps(0.5))))
#
# p1 = ps.phasor()
# p2 = ps.phasor_new()
# timeit('p1(480)')
# timeit('p2(480)')
# data = gen(p2(480) for _ in range(ps.fps(0.2)))
# plt.plot(data, '.')


def exp_rise(dst, i, size, last, dur, th):
    r = th ** (1/dur)
    d = 1 / (th - 1)
    o = -d
    p = max(0, (last - o) / d)
    while p > th and i < size:
        p = p*r
        dst[i] = p*d + o
        i += 1
    return i


def exp_fall(dst, i, size, last, dur, th, a, b):
    r = th ** (1/dur)
    d = a - b
    o = b - th * d
    p = max(0, (last - o) / d)
    while p > th and i < size:
        p = p*r
        dst[i] = p*d + o
        i += 1
    return i


# d = [0] * 10000
# # exp_rise(d, 0, len(d), 0, 10*441, 0.5)
# exp_fall(d, 0, len(d), 1.5, 10*441, 0.1, 1, 0.5)
# print(d[:5], d[10*441-5:][:5])

# N = 44100 // 10
# a = 0.05
# r = 0.001
# p = 1 + r
# a = math.exp(-math.log((1 + r) / r) / N) - 1
# # for i in range(N):
# while (p >= r):
#     d.append(p - r)
#     p = p + p*a

# ideal = [(1.05 ** i - 1) / 1000 for i in range(100)]
# plt.plot(d)
# plt.plot(ideal)

# filter_im_res()

def genv_until_stop(env, *args):
    for _ in range(ps.fps(3)):
        if not env.active:
            break
        yield env(*args)

env = ps.env_ahdsr(rise_th=0.5, fall_th=0.2)
args = 500, 100, 0.8, 500
env.stop(True)
data = gen(env(*args) for _ in range(ps.fps(0.2)))
data = gen([data], genv_until_stop(env, *args))
plt.plot(data)

plt.show()
