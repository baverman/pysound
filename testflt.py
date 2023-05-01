#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pysound as ps
import cfilters
from timeit_helper import timeit
import cProfile

def poly_blep(t, dt):
    if t < dt:
        t /= dt
        return t+t - t*t - 1.
    elif t > 1. - dt:
        t = (t - 1.) / dt
        return t*t + t+t + 1.
    return 0.0


def poly_saw_s(t, dt):
    naive_saw = 2.*t - 1.
    return naive_saw - poly_blep(t, dt), t


def poly_saw_n():
    t = 0.0
    def gen(freq):
        nonlocal t
        dt = ps.ensure_buf(freq) / ps.FREQ
        result = np.zeros(ps.BUFSIZE, dtype=np.float32)
        for i in range(ps.BUFSIZE):
            if t >= 1.:
                t -= 1.

            result[i], t = poly_saw_s(t, dt[i])
            # print(t, t < dt[i], t > 1. - dt[i], poly_blep(t, dt[i]))
            t += dt[i]
        return result
    return gen


def gen(*items):
    result = []
    for it in items:
        for x in it:
            result.append(x)
    return np.concatenate(result)


def pulse(ssig):
    ssig = (ssig + 1) / 2
    return (ssig - ((ssig + 0.5) % 1.0))


p = poly_saw_n()
# p = ps.poly_saw()
pp = ps.phasor()
sqr = ps.poly_square()
dc = ps.dcfilter()

lfo = ps.osc(ps.sin_t)

# plt.plot(gen(pulse(p(100)) for _ in range(10)))
# plt.show()

# plt.plot(gen(p(100 + lfo(10)*10.0) for _ in range(10)), 'o')

# plt.plot(gen(sqr(1000) for _ in range(4)), '.')
# plt.show()
# 1/0

f = 3969
# s = gen(ps.phasor_apply(pp(f), ps.square_t) for _ in range(10))
# plt.plot(*ps.fft_plot(s))

# s = gen(ps.phasor_apply(pp(f), ps.saw_t) for _ in range(10))
# plt.plot(*ps.fft_plot(s))

# s = gen((pp(f) > 0.5) - 0.5 for _ in range(10))
# plt.plot(*ps.fft_plot(s))

# s = gen(pulse(p(f)) for _ in range(10))
# plt.plot(*ps.fft_plot(s, crop=1))

# s = gen(pp(f) for _ in range(10))
# plt.plot(*ps.fft_plot(s, crop=1))

s = gen(sqr(f) for _ in range(10))
plt.plot(*ps.fft_plot(s, crop=1))


# plt.plot(gen(noise_t[1][p(1) * 2048] for _ in range(ps.fps(0.1))))
# plt.plot(gen(l(o(300), 0.1) for _ in range(ps.fps(0.15))))
plt.show()
1/0

o = ps.osc(ps.sin_t)
env = ps.env_ahr(1, 30, 1)
d = ps.delay()

def ndelay():
    buf = np.full(ps.sps(0.5), 0, dtype=np.float32)
    pos = 0
    bsize = len(buf)
    fflt = ps.lowpass()
    def process(sig, delay, _):
        nonlocal pos
        result = sig.copy()
        ds = ps.sps(delay)
        for i in range(len(result)):
            s = buf[(pos + i - ds) % bsize]
            result[i] = sig[i] + s
            buf[(pos+i) % bsize] = result[i]
        # plt.plot(buf)
        # plt.show()
        buf[pos:pos+len(result)] = fflt(result, 18000)
        pos += len(result)
        return result
    return process


def boo_delay():
    buf = cfilters.init_ring_buf(ps.sps(0.5))
    fflt = ps.lowpass()
    def process(sig, delay, _):
        delays = ps.ensure_buf(ps.sps(delay), np.int32)
        result = ps.ensure_buf(0)
        cfilters.delmix(buf, result, sig, delays)
        cfilters.delwrite(buf, fflt(result, 18000))
        plt.plot(np.diff(result, 2))
        plt.show()
        return result
    return process


d = ndelay()
d = boo_delay()
flt = ps.lowpass()
n = ps.seed_noise(1)
# plt.plot(n() * env())
# plt.show()

sig = np.concatenate([flt(n(), 8000) * env() for _ in range(ps.fps(0.06))])
dsig = np.concatenate([d(sig[i:i+512], 0.010, 1) for i in range(0, len(sig), 512)])

# plt.plot(sig)
# plt.plot(np.diff(dsig, 2))
plt.plot(dsig)
plt.show()

