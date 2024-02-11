# python
# Basic rich oscilator
# https://www.youtube.com/watch?v=I88Cxi86Zu8&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=1
from pysound import osc, sin_t, fps, mtof, choicer, GUI, Var
from tonator import Scales

gui = GUI(
    Var('tempo', 250, 50, 600),
    Var('master-volume', 0.5, 0, 1, resolution=0.01),
)


def oscs():
    o1 = osc(sin_t)
    o2 = osc(sin_t)
    o3 = osc(sin_t)
    o4 = osc(sin_t)
    o5 = osc(sin_t)
    return lambda f: (o1(f) + o2(f*1.015) + o3(f*0.503) + o4(f*1.496)*0.5 + o5(f*2.01)*0.25) / 3.75


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    s1 = oscs()
    s2 = oscs()
    s3 = oscs()
    while True:
        f = mtof(60 + next(notes))
        for _ in range(fps(60/ctl['tempo'])):
            yield (s1(f) + s2(2*f) + s3(0.5*f)) / 3


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
