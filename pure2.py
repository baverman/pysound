#padsp python
# Envelope
# https://www.youtube.com/watch?v=gqpvIwYko3o&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=2
import numpy as np
from pysound import GUI, Var, mtof, sps, BUFSIZE, choicer, fps, env_ahr
from tonator import Scales
from pure1 import oscs

gui = GUI(
    Var('tempo', 250, 50, 600),
    Var('attack', 5, 1, 100),
    Var('hold', 10, 1, 1000),
    Var('release', 500, 1, 2000),
    Var('master-volume', 0.5, 0, 1, resolution=0.01),
)


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    s = oscs()
    line = None  # we need a previous envelope to stich frames without a click
    while True:
        f = mtof(60 + next(notes))
        line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'], line and line.last)
        for _ in range(fps(60/ctl['tempo'])):
            yield s(f) * line()**4


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
