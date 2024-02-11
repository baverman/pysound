# python
# Plucked strings
# https://www.youtube.com/watch?v=5rVveOpGsWo&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=8
import numpy as np
from pysound import (
    mtof, fps, choicer, poly, env_ahdsr, lowpass, seed_noise, cfilters, sps,
    ensure_buf, FREQ, dcfilter, mono, delay)
from pysound.gui import GUI, Var
from pysound.tonator import Scales

gui = GUI(
    Var('tempo', 240, 50, 600),
    Var('attack', 15, 1, 100),
    Var('hold', 300, 1, 1000),
    Var('release', 2000, 1, 4000),
    Var('feedback', 0.99, 0.8, 1, resolution=0.0001),
    Var('cutoff', 0.72, 0, 1, resolution=0.01),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth():
    n = seed_noise(1)
    flt = lowpass()
    def gen(ctl, params):
        eg = env_ahdsr()
        e = env_ahdsr()
        d = delay(0.5)
        delays = ensure_buf(FREQ/params['freq'], np.int32)
        while True:
            sig = n() * eg(0, 0, 0, 0, 30)
            sig = d(sig, delays, ctl['feedback'])
            sig = sig * e(ctl['attack'], ctl['release'], 0, 0)
            yield flt(sig, ctl['cutoff'])

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
