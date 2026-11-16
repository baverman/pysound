# python
# Acid
# https://www.youtube.com/watch?v=_qImZOcHz2U&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=4
import itertools
from dataclasses import dataclass

from pure3 import mix
from pysound import env_adsr, fps, lowpass, mtof, ngui, phasor


@dataclass
class Volume:
    vol: float = ngui.field(0.5, min=0, max=1, resolution=0.01)
    attack: float = ngui.field(8, min=1, max=100)
    hold: float = ngui.field(100, min=1, max=1000)
    release: float = ngui.field(100, min=1, max=2000)


@dataclass
class Filter:
    on: float = ngui.field(1, min=0, max=1)
    attack: float = ngui.field(5, min=1, max=100)
    hold: float = ngui.field(1, min=1, max=500)
    release: float = ngui.field(260, min=1, max=2000)
    cutoff: float = ngui.field(0.7, min=0, max=1, resolution=0.01)
    resonance: float = ngui.field(0.7, min=0.5, max=1, resolution=0.001)


@dataclass
class Voice:
    volume: Volume = ngui.field(default_factory=Volume)
    filter: Filter = ngui.field(default_factory=Filter)


@dataclass
class Controls:
    tempo: float = ngui.field(320, min=50, max=600)
    master_volume: float = ngui.field(0.2, min=0, max=1, resolution=0.01)
    _voices: ngui.Group = ngui.group(ngui.Panel, orientation='horizontal')
    voice_1: Voice = ngui.field(default_factory=Voice, group=_voices)
    voice_2: Voice = ngui.field(default_factory=Voice, group=_voices)
    voice_3: Voice = ngui.field(default_factory=Voice, group=_voices)


def synth(ctl, f, note_vol):
    ctlf = ctl.filter
    ctlv = ctl.volume
    o = phasor()
    env = env_adsr(decay_next_state=6)
    filter_env = env_adsr(decay_next_state=6)
    lp = lowpass()
    while env.active:
        sig = o(f)
        fenv = filter_env(ctlf.attack, ctlf.release, 0, 0, hold=ctlf.hold)
        lp_sig = lp(sig, fenv**4 * ctlf.cutoff, ctlf.resonance)
        if ctlf.on > 0:
            sig = lp_sig
        amp = env(ctlv.attack, ctlv.release, 0, 0, hold=ctlv.hold)
        yield sig * amp**4 * ctlv.vol * note_vol


def gen(ctl):
    notes = itertools.cycle([10, 2, 9, 2, 7, 2, 5, 3, 2])
    notes_vol = itertools.cycle([1, 0.7, 1, 0.7, 1, 0.7, 1, 1, 1])

    notes2 = itertools.cycle([10, 2, 14, 2, 12, 2, 10, 9, 7])
    notes2_vol = itertools.cycle([1, 0.7, 1, 0.7, 1, 0.7, 1, 1, 1])

    notes3 = itertools.cycle([0, 0, 0, 0, 0, 21, 22, 15, 14])
    notes3_vol = itertools.cycle([0, 0, 0, 0, 0, 1, 1, 0.85, 0.6])

    voices = []
    while True:
        f, fv = mtof(36 + next(notes)), next(notes_vol)
        voices.append(synth(ctl.voice_1, f, fv))
        f, fv = mtof(36 + next(notes2)), next(notes2_vol)
        voices.append(synth(ctl.voice_2, f, fv))
        f, fv = mtof(36 + next(notes3)), next(notes3_vol)
        voices.append(synth(ctl.voice_3, f, fv))
        for _ in range(fps(60 / ctl.tempo)):
            yield mix(voices)


if __name__ == '__main__':
    ctl = Controls()
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
