# python
# Synth strings
# https://www.youtube.com/watch?v=XVcQuI5Urdw&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=7
from dataclasses import dataclass

from pure3 import mix
from pysound import choicer, dcfilter, env_ahr, fps, lowpass, mtof, ngui, osc, sin_t, square
from pysound.ngui import field, pow2
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(75, range=(50, 600), nl=True)
    attack: float = field(50, range=(1, 100), map=pow2)
    hold: float = field(350, range=(1, 1000), map=pow2)
    release: float = field(1500, range=(1, 2000), map=pow2)
    modulation_freq: float = field(4, range=(0, 30), resolution=0.1)
    modulation_depth: float = field(0.3, range=(0, 1), resolution=0.01)


def square_mod():
    o = square()
    wm = osc(sin_t)
    flt = lowpass()

    def sig(f, mfreq, mdepth):
        duty = 0.5 + mdepth * wm(mfreq) / 2.0
        return flt(o(f, duty), 0.3)

    return sig


def synth(ctl, f):
    o1 = square_mod()
    o2 = square_mod()
    line = env_ahr()
    while line.active:
        mfreq = ctl.modulation_freq
        mdepth = ctl.modulation_depth
        sig = (o1(f, mfreq, mdepth) + o2(f * 1.004, mfreq, mdepth)) / 2.0
        yield sig * line(ctl.attack, ctl.hold, ctl.release)


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    dc = dcfilter()
    voices = []
    while True:
        f = mtof(60 + next(notes))
        voices.append(synth(ctl, f))
        for _ in range(fps(60 / ctl.tempo)):
            yield dc(mix(voices))


if __name__ == '__main__':
    ctl = ngui.make(Controls)
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
