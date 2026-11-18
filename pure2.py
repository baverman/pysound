# python
# Envelope
# https://www.youtube.com/watch?v=gqpvIwYko3o&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=2
from dataclasses import dataclass

from pure1 import oscs
from pysound import choicer, env_ahr, fps, gui, mtof
from pysound.gui import field, pow2
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.5, range=(0, 1), resolution=0.01)
    tempo: float = field(250, range=(50, 600), nl=True)
    attack: float = field(5, range=(1, 100), map=pow2)
    hold: float = field(10, range=(1, 1000), map=pow2)
    release: float = field(500, range=(1, 2000), map=pow2)


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    s = oscs()
    env = env_ahr()
    while True:
        f = mtof(60 + next(notes))
        env.trigger()
        for _ in range(fps(60 / ctl.tempo)):
            e = env(ctl.attack, ctl.hold, ctl.release)
            yield s(f) * e


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
