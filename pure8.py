# python
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
    Var('feedback', 0.99, 0.95, 1, resolution=0.0001),
    Var('cutoff', 0.72, 0, 1, resolution=0.01),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth():
    buf = cfilters.init_ring_buf(sps(0.5))
    n = seed_noise(1)
    flt = lowpass()
    def gen(ctl, params):
        tline = env_ahr(0, 30, 0)
        delays = ensure_buf(FREQ/params['freq'], np.int32)
        while True:
            sig = n() * tline()
            result = ensure_buf(0)
            cfilters.delmix(buf, result, sig, delays)
            result = flt(result*ctl['feedback'], ctl['cutoff'])
            cfilters.delwrite(buf, result)
            yield result

    return gen


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    dc = dcfilter()
    s = synth()
    while True:
        f = mtof(60 + next(notes) - 24)
        g = s(ctl, {'freq': f, 'volume': 1.0})
        for _ in range(fps(60/ctl['tempo'])):
            yield dc(next(g))


if __name__ == '__main__':
    from pysound import render_to_file
    # render_to_file('/tmp/boo.wav', gui.ctl, gen(gui.ctl), 1)
    gui.play(gen(gui.ctl))
