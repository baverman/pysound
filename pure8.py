# python
# Plucked strings
# https://www.youtube.com/watch?v=5rVveOpGsWo&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=8
from dataclasses import dataclass

import numpy as np

from pysound import (
    FREQ, choicer, dcfilter, delay, ensure_buf, env_adsr, fps, lowpass, mtof, ngui,
    seed_noise)
from pysound.ngui import field, pow2
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(240, range=(50, 600), nl=True)
    attack: float = field(15, range=(1, 100), map=pow2)
    hold: float = field(300, range=(1, 1000), map=pow2)
    release: float = field(2000, range=(1, 4000), map=pow2)
    feedback: float = field(0.99, range=(0.8, 1), resolution=0.0001)
    cutoff: float = field(0.72, range=(0, 1), resolution=0.01)


def synth(ctl, f):
    n = seed_noise(1)
    flt = lowpass()
    eg = env_adsr()
    e = env_adsr()
    d = delay(0.5)
    delays = ensure_buf(FREQ / f, np.int32)
    while True:
        sig = n() * eg(0, 0, 0, 0, 30)
        sig = d(sig, delays, ctl.feedback)
        sig = sig * e(ctl.attack, ctl.release, 0, 0)
        yield flt(sig, ctl.cutoff)


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    dc = dcfilter()
    while True:
        f = mtof(60 + next(notes) - 24)
        voice = synth(ctl, f)
        for _ in range(fps(60 / ctl.tempo)):
            yield dc(next(voice))


if __name__ == '__main__':
    ctl = ngui.make(Controls)
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
