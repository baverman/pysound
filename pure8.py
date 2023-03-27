#padsp python
# Plucked strings
# https://www.youtube.com/watch?v=5rVveOpGsWo&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=8
import numpy as np
from pysound import (
    GUI, Var, mtof, fps, choicer, poly, env_ahr, lowpass, seed_noise, cfilters, sps,
    ensure_buf, FREQ, dcfilter, mono)
from tonator import Scales

gui = GUI(
    Var('tempo', 240, 50, 600),
    Var('attack', 15, 1, 100),
    Var('hold', 300, 1, 1000),
    Var('release', 2000, 1, 4000),
    Var('feedback', 0.97, 0.95, 1, resolution=0.0001),
    Var('cutoff', 18000, 5000, 20000),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth(ctl):
    def gen(f, last):
        buf = cfilters.init_ring_buf(sps(0.5))
        n = seed_noise(1)
        flt = lowpass()
        fltmain = lowpass()
        tline = env_ahr(0, 30, 0)
        line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'], last)
        delays = ensure_buf(FREQ/f, np.int32)
        while True:
            sig = fltmain(n(), 8000) * tline()
            result = ensure_buf(0)
            cfilters.delmix(buf, result, sig, delays)
            # cfilters.delwrite(buf, flt(result*ctl['feedback'], ctl['cutoff']))
            cfilters.delwrite(buf, flt(result, ctl['cutoff']))
            # cfilters.delwrite(buf, result*ctl['feedback'])
            # yield result * line()**2, line.last
            yield result, 0

    return gen


def _gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    dc = dcfilter()
    while True:
        f = mtof(60 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            # yield dc(p())
            yield p()


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    m = mono()
    g = synth(ctl)
    while True:
        f = mtof(60 + next(notes))
        m.set(g(f, m.last))
        for _ in range(fps(60/ctl['tempo'])):
            yield m()


if __name__ == '__main__':
    from pysound import render_to_file
    # render_to_file('/tmp/boo.wav', gui.ctl, gen(gui.ctl), 1)
    gui.play(gen(gui.ctl))
