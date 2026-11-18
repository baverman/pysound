# python
# FM
# https://www.youtube.com/watch?v=mvtN7de6Oko&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=6
from dataclasses import dataclass

from pure3 import mix
from pysound import choicer, env_ahr, fps, gui, mtof, osc, sin_t
from pysound.gui import field, pow2
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(330, range=(50, 600), nl=True)
    harmonicity: float = field(1.5, range=(0, 4), resolution=0.01)
    modulation_index: float = field(4, range=(0, 10), resolution=0.1)
    modulation_release: float = field(450, range=(1, 1000), map=pow2)
    attack: float = field(10, range=(1, 100), map=pow2)
    hold: float = field(50, range=(1, 1000), map=pow2)
    release: float = field(500, range=(1, 2000), map=pow2)


def fm():
    o = osc(sin_t)
    m = osc(sin_t)

    def sig(f, harmonicity, modindex):
        depth = f * harmonicity * modindex
        mf = m(f * harmonicity)
        return o(f + mf * depth)

    return sig


def synth(ctl, f):
    o = fm()
    line = env_ahr()
    mline = env_ahr()
    while line.active:
        envelope = line(ctl.attack, ctl.hold, ctl.release)
        modulation = mline(10, 1, ctl.modulation_release)
        yield o(f, ctl.harmonicity, modulation * ctl.modulation_index) * envelope


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    voices = []
    while True:
        f = mtof(48 + next(notes))
        voices.append(synth(ctl, f))
        for _ in range(fps(60 / ctl.tempo)):
            yield mix(voices)


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
