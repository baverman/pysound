# python
from dataclasses import dataclass

import pysound as ps
from pysound import choicer, fps, gui, mono, osc, sin_t
from pysound.gui import field
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.5, range=(0, 1), resolution=0.01)
    tempo: float = field(250, range=(50, 600), nl=True)


def synth(ctl, params):
    o = osc(sin_t)
    env = ps.env_ahr()
    params.env = env

    def gen():
        return o(params.freq) * env(15, 100, 1000) ** 2

    return gen


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    player = mono(ctl, synth)
    while True:
        player.note_on(0, 60 + next(notes), 1.0)
        for _ in range(fps(60 / ctl.tempo)):
            yield player()


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
