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
            result.append(x)
    return np.concatenate(result)

p = ps.poly_saw()

plt.plot(gen(p(50) for _ in range(ps.fps(0.1))))

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

