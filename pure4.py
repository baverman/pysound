#padsp python
# Acid
# https://www.youtube.com/watch?v=_qImZOcHz2U&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=4
from random import choice
from pysound import GUI, Var, mtof, lowpass, phasor, fps, poly, choicer, env_ahr
from tonator import Scales

gui = GUI(
    Var('tempo', 250, 50, 600),
    Var('attack', 8, 1, 100),
    Var('hold', 100, 1, 1000),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-hold', 1, 1, 1000),
    Var('filter-release', 200, 1, 2000),
    Var('filter-cutoff', 15000, 100, 20000),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth(ctl, f):
    o = phasor()
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    fline = env_ahr(ctl['filter-attack'], ctl['filter-hold'], ctl['filter-release'])
    lp = lowpass()
    while line.running:
        sig = o(f)
        if ctl['filter-on'] > 0:
            sig = lp(sig, fline()**4*ctl['filter-cutoff'])
        yield sig * line()**4


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    while True:
        f = mtof(30 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            yield p()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
