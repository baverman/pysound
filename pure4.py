#padsp python
# Acid
# https://www.youtube.com/watch?v=_qImZOcHz2U&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=4
from pysound import GUI, Var, mtof, lowpass, phasor, fps, poly, choicer, env_ahr, VarGroup
from tonator import Scales
import itertools

svars = [
    Var('vol', 0.5, 0, 1, resolution=0.01),
    Var('attack', 8, 1, 100),
    Var('hold', 100, 1, 1000),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-hold', 1, 1, 1000),
    Var('filter-release', 260, 1, 2000),
    Var('filter-cutoff', 15000, 100, 20000),
    Var('filter-resonance', 0.7, 0, 1, resolution=0.01),
]

gui = GUI(
    Var('tempo', 320, 50, 600),
    VarGroup('voice-1', svars),
    VarGroup('voice-2', svars),
    VarGroup('voice-3', svars),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth(ctl, f, note_vol):
    o = phasor()
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    fline = env_ahr(ctl['filter-attack'], ctl['filter-hold'], ctl['filter-release'])
    lp = lowpass()
    while line.running:
        sig = o(f)
        lp_sig = lp(sig, fline()**4 * ctl['filter-cutoff'], ctl['filter-resonance'])
        if ctl['filter-on'] > 0:
            sig = lp_sig
        yield sig * line()**4 * ctl['vol'] * note_vol


def gen(ctl):
    # notes2 = choicer(it.value for it in Scales.major.notes)

    notes = itertools.cycle([10, 2, 9, 2, 7, 2, 5, 3, 2])
    notes_vol = itertools.cycle([1,  0.7, 1, 0.7, 1, 0.7, 1, 1, 1])

    notes2 = itertools.cycle([10, 2, 14, 2, 12, 2, 10, 9, 7])
    notes2_vol = itertools.cycle([1,  0.7, 1, 0.7, 1, 0.7, 1, 1, 1])

    notes3 = itertools.cycle(  [0, 0, 0, 0,   0, 21, 22, 15, 14])
    notes3_vol = itertools.cycle([0, 0, 0,  0, 0, 1, 1,   0.85, 0.6])
    p = poly()
    while True:
        f, fv = mtof(36 + next(notes)), next(notes_vol)
        p.add(synth(ctl['voice-1'], f, fv))
        f, fv = mtof(36 + next(notes2)), next(notes2_vol)
        p.add(synth(ctl['voice-2'], f, fv))
        f, fv = mtof(36 + next(notes3)), next(notes3_vol)
        p.add(synth(ctl['voice-3'], f, fv))
        for _ in range(fps(60/ctl['tempo'])):
            yield p()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
