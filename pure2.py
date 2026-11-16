# python
# Envelope
# https://www.youtube.com/watch?v=gqpvIwYko3o&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=2
from dataclasses import dataclass

from pure1 import oscs
from pysound import choicer, env_adsr, fps, mtof, ngui
from pysound.tonator import Scales


@dataclass
class Controls:
    tempo: float = ngui.field(250, min=50, max=600)
    attack: float = ngui.field(5, min=1, max=100)
    hold: float = ngui.field(10, min=1, max=1000)
    release: float = ngui.field(500, min=1, max=2000)
    master_volume: float = ngui.field(0.5, min=0, max=1, resolution=0.01)


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    s = oscs()
    env = env_adsr()
    while True:
        f = mtof(60 + next(notes))
        env.trigger()
        for _ in range(fps(60 / ctl.tempo)):
            e = env(ctl.attack, ctl.release, 0, 0, hold=ctl.hold)
            yield s(f) * e**4


if __name__ == '__main__':
    ctl = Controls()
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
