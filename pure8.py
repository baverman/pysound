#padsp python
# Plucked strings
# https://www.youtube.com/watch?v=5rVveOpGsWo&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=8
from pysound import (
    GUI, Var, mtof, fps, choicer, poly, env_ahr, lowpass, noise, delay)
from tonator import Scales

gui = GUI(
    Var('tempo', 240, 50, 600),
    Var('attack', 15, 1, 100),
    Var('hold', 300, 1, 1000),
    Var('release', 2000, 1, 4000),
    Var('feedback', 0.99, 0.97, 0.9999, resolution=0.0001),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth(ctl, f):
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    tline = env_ahr(0, 30, 0)
    d = delay()
    flt = lowpass()
    while line.running:
        sig = flt(d(noise() * tline(), 1/f, ctl['feedback']), 10000)
        yield sig * line()**4


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    while True:
        f = mtof(60 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            yield p()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
