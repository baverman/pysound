#padsp python
# Synth strings
# https://www.youtube.com/watch?v=XVcQuI5Urdw&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=7
from pysound import (
    GUI, Var, mtof, osc, sin_t, fps, choicer, poly, env_ahr,
    lowpass, square, dcfilter)
from tonator import Scales

gui = GUI(
    Var('tempo', 75, 50, 600),
    Var('attack', 50, 1, 100),
    Var('hold', 350, 1, 1000),
    Var('release', 1500, 1, 2000),
    Var('modulation-freq', 4, 0, 30, resolution=0.1),
    Var('modulation-depth', 0.3, 0, 1, resolution=0.01),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def square_mod():
    o = square()
    wm = osc(sin_t)
    flt = lowpass()
    def sig(f, mfreq, mdepth):
        duty = 0.5 + mdepth * wm(mfreq) / 2.0
        return flt(o(f, duty), 10000)
    return sig


def synth(ctl, f):
    o1 = square_mod()
    o2 = square_mod()
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    mfreq = ctl['modulation-freq']
    mdepth = ctl['modulation-depth']
    while line.running:
        sig = (o1(f, mfreq, mdepth) + o2(f*1.004, mfreq, mdepth)) / 2.0
        yield sig * line()**4


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    dc = dcfilter()
    while True:
        f = mtof(60 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            yield dc(p())


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
