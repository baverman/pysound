#padsp python
# Acid
# https://www.youtube.com/watch?v=_qImZOcHz2U&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=4
from random import choice
from pysound import Var, gui_play, mtof, lowpass, phasor
from tonator import Scales
from pure3 import env, poly

ctl = {
    'attack': Var('Attack', 8, 1, 100),
    'hold': Var('Hold', 100, 1, 1000),
    'release': Var('Release', 500, 1, 2000),
    'fattack': Var('Filter Attack', 12, 1, 100),
    'fhold': Var('Filter Hold', 1, 1, 1000),
    'frelease': Var('Filter Release', 100, 1, 2000),
    'cutoff': Var('Cutoff', 10000, 100, 20000),
    'master': Var('Master volume', 0.2, 0, 1, resolution=0.01),
}


def synth(ctl, f):
    o = phasor()
    line = env(ctl['attack'].val, ctl['hold'].val, ctl['release'].val)
    fline = env(ctl['fattack'].val, ctl['fhold'].val, ctl['frelease'].val)
    lp = lowpass()
    while True:
        yield lp(o(f), fline()*ctl['cutoff'].val) * line()**4
        if line.last == 0:
            break


def gen(ctl):
    notes = [it.value for it in Scales.major.notes]
    p = poly()
    while True:
        note = choice(notes)
        f = mtof(30 + note)
        p.add(synth(ctl, f))
        for _ in range(15):
            yield p()


if __name__ == '__main__':
    gui_play(ctl, gen(ctl))
