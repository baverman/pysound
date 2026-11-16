# python
# Polyphony
# https://www.youtube.com/watch?v=jGgu77pCfHU&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=3
from dataclasses import dataclass

from pysound import choicer, env_adsr, fps, mtof, ngui, osc, sin_t
from pysound.tonator import Scales


@dataclass
class Controls:
    tempo: float = ngui.field(250, min=50, max=600)
    attack: float = ngui.field(8, min=1, max=100)
    hold: float = ngui.field(70, min=1, max=1000)
    release: float = ngui.field(1200, min=1, max=2000)
    master_volume: float = ngui.field(0.2, min=0, max=1, resolution=0.01)


def synth(ctl, f):
    o = osc(sin_t)
    env = env_adsr()
    while env.active:
        e = env(ctl.attack, ctl.release, 0, 0, hold=ctl.hold)
        yield o(f) * e**4


def mix(voices):
    active = []
    result = 0
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
    ctl = Controls()
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
