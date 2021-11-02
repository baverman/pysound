#padsp python
from random import choice
from pysound import osc, sin_t, Var, gui_play, fps, mtof
from tonator import Scales, Note

ctl = {
    'master': Var('Master volume', 0.1, 0, 1, resolution=0.01),
}


def seq(fpq, notes):
    while True:
        note = choice(notes)
        for _ in range(fpq):
            yield note


def oscs():
    o1 = osc(sin_t)
    o2 = osc(sin_t)
    o3 = osc(sin_t)
    o4 = osc(sin_t)
    o5 = osc(sin_t)
    return lambda f: o1(f) + o2(f*1.015) + o3(f*0.503) + o4(f*1.496)*0.5 + o5(f*2.01)*0.25


def gen(ctl):
    notes = [it.value for it in Scales.major.notes]
    s1 = oscs()
    s2 = oscs()
    s3 = oscs()
    for note in seq(15, notes):
        f = mtof(60 + note)
        yield s1(f) + s2(2*f) + s3(0.5*f)


gui_play(ctl, gen(ctl))
