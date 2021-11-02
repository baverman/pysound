#padsp python
# Envelope
# https://www.youtube.com/watch?v=gqpvIwYko3o&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=2
import numpy as np
from random import choice
from pysound import Var, gui_play, mtof, sps, BUFSIZE
from tonator import Scales
from pure1 import oscs

ctl = {
    'attack': Var('Attack', 5, 1, 100),
    'hold': Var('Hold', 10, 1, 1000),
    'release': Var('Release', 500, 1, 2000),
    'master': Var('Master volume', 0.5, 0, 1, resolution=0.01),
}


def seq(fpq, notes):
    while True:
        yield choice(notes), fpq


def env(attack, hold, release, last=0):
    acnt = sps(attack / 1000)
    hcnt = sps(hold / 1000)
    rcnt = sps(release / 1000)
    e = np.concatenate([
        np.linspace(last, 1, acnt, endpoint=False, dtype=np.float32),
        np.full(hcnt, 1, dtype=np.float32),
        np.linspace(1, 0, rcnt, dtype=np.float32),
        np.full(BUFSIZE, 0, dtype=np.float32),
    ])

    samples = 0
    def gen():
        nonlocal samples
        if samples < acnt + hcnt + rcnt:
            result = e[samples:samples+BUFSIZE]
        else:
            result = e[-BUFSIZE:]
        samples += BUFSIZE
        gen.last = result[-1]
        return result
    return gen


def gen(ctl):
    notes = [it.value for it in Scales.major.notes]
    s = oscs()
    last = 0  # we need a previous envelope value to stich frames without click
    for note, fcount in seq(15, notes):
        line = env(ctl['attack'].val, ctl['hold'].val, ctl['release'].val, last)
        for _ in range(fcount):
            f = mtof(60 + note)
            yield s(f) * line()**4
        last = line.last


if __name__ == '__main__':
    gui_play(ctl, gen(ctl))
