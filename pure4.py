# python
# Acid
# https://www.youtube.com/watch?v=_qImZOcHz2U&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=4
import itertools
from dataclasses import dataclass

from pure3 import mix
from pysound import env_ahr, fps, lowpass, mtof, ngui, phasor
from pysound.ngui import field, pow2


@dataclass
class Volume:
    vol: float = field(0.5, range=(0, 1), resolution=0.01)
    attack: float = field(8, range=(1, 100), map=pow2)
    hold: float = field(100, range=(1, 1000), map=pow2)
    release: float = field(100, range=(1, 2000), map=pow2)


@dataclass
class Filter:
    on: float = field(1, range=(0, 1))
    attack: float = field(5, range=(1, 100), map=pow2)
    hold: float = field(1, range=(1, 500), map=pow2)
    release: float = field(260, range=(1, 2000), map=pow2)
    cutoff: float = field(0.7, range=(0, 1), map=pow2, resolution=0.01)
    resonance: float = field(0.7, range=(0.5, 1), resolution=0.001)


@dataclass
class Voice:
    volume: Volume
    filter: Filter


@dataclass(kw_only=True)
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(320, range=(50, 600), resolution=1, nl=True)
    _voices: ngui.Group = ngui.group(orientation='vertical')
    voice_1: Voice = field(group=_voices)
    voice_2: Voice = field(group=_voices)
    voice_3: Voice = field(group=_voices)


def synth(ctl, f, note_vol):
    ctlf = ctl.filter
    ctlv = ctl.volume
    o = phasor()
    env = env_ahr()
    filter_env = env_ahr()
    lp = lowpass()
    while env.active:
        sig = o(f)
        fenv = filter_env(ctlf.attack, ctlf.hold, ctlf.release)
        lp_sig = lp(sig, fenv * ctlf.cutoff, ctlf.resonance)
        if ctlf.on > 0:
            sig = lp_sig
        amp = env(ctlv.attack, ctlv.hold, ctlv.release)
        yield sig * amp * ctlv.vol * note_vol


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
    ctl = ngui.make(Controls)
    gui = ngui.GUI(Controls)
    gui.bind(ctl)
    gui.play(gen(ctl))
