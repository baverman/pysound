# python
# Polyphony
# https://www.youtube.com/watch?v=jGgu77pCfHU&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=3
from dataclasses import dataclass

from pysound import choicer, env_ahr, fps, gui, mtof, osc, pass_buf, sin_t
from pysound.gui import field, pow2
from pysound.tonator import Scales


@dataclass
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(250, range=(50, 600), nl=True)
    attack: float = field(8, range=(1, 100), map=pow2)
    hold: float = field(70, range=(1, 1000), map=pow2)
    release: float = field(1200, range=(1, 2000), map=pow2)


def synth(ctl, f):
    o = osc(sin_t)
    env = env_ahr()
    while env.active:
        e = env(ctl.attack, ctl.hold, ctl.release)
        yield o(f) * e

def mix(voices):
    active = []
    result = pass_buf(None)
    for voice in voices:
        frame = next(voice, None)
        if frame is not None:
            active.append(voice)
            result += frame
    voices[:] = active
    return result


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    voices = []
    while True:
        f = mtof(60 + next(notes))
        voices.append(synth(ctl, f))
        for _ in range(fps(60 / ctl.tempo)):
            yield mix(voices)


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
