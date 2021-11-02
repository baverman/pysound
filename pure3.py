#padsp python
# Polyphony
# https://www.youtube.com/watch?v=jGgu77pCfHU&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=3
import numpy as np
from random import choice
from pysound import Var, gui_play, mtof, BUFSIZE, osc, sin_t
from tonator import Scales
from pure2 import env

ctl = {
    'attack': Var('Attack', 8, 1, 100),
    'hold': Var('Hold', 70, 1, 1000),
    'release': Var('Release', 1200, 1, 2000),
    'master': Var('Master volume', 0.2, 0, 1, resolution=0.01),
}


def seq(fpq, notes):
    while True:
        yield choice(notes), fpq


def synth(ctl, f):
    o = osc(sin_t)
    line = env(ctl['attack'].val, ctl['hold'].val, ctl['release'].val)
    while True:
        yield o(f) * line()**4
        if line.last == 0:
            break

class poly:
    def __init__(self):
        self.gens = set()

    def add(self, gen):
        self.gens.add(gen)

    def __call__(self):
        toremove = []
        result = np.full(BUFSIZE, 0, dtype=np.float32)
        for g in self.gens:
            data = next(g, None)
            if data is None:
                toremove.append(g)
            else:
                result += data
        self.gens.difference_update(toremove)
        return result


def gen(ctl):
    notes = [it.value for it in Scales.major.notes]
    p = poly()
    for note, fcount in seq(15, notes):
        f = mtof(60 + note)
        p.add(synth(ctl, f))
        for _ in range(fcount):
            yield p()


if __name__ == '__main__':
    gui_play(ctl, gen(ctl))
