#padsp python
import itertools

from pysound import (
    GUI, Var, mtof, lowpass, phasor, fps, env_ahr, VarGroup,
    env_adsr, Trigger, BUFSIZE, FREQ, poly)

import tonator
import tntparser

svars = [
    Var('vol', 0.5, 0, 1, resolution=0.01, midi_ctrl=0),
    Var('attack', 8, 1, 100),
    Var('sustain', 0.8, 0, 1, resolution=0.01),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-hold', 1, 1, 500, midi_ctrl=1),
    Var('filter-release', 260, 1, 2000),
    Var('filter-cutoff', 15000, 100, 20000, midi_ctrl=2),
    Var('filter-resonance', 0.7, 0.5, 1, resolution=0.001, midi_ctrl=3),
]

gui = GUI(
    Var('tempo', 320, 50, 600),
    VarGroup('voice-1', svars, midi_channel=0),
    # VarGroup('voice-2', svars, midi_channel=1),
    # VarGroup('voice-3', svars, midi_channel=2),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    preset_prefix='acid-',
)


def synth(ctl, trig, f, note_vol):
    o = phasor()
    line = env_adsr(trig, ctl['attack'], ctl['attack'], ctl['sustain'], ctl['release'])
    fline = env_ahr(ctl['filter-attack'], ctl['filter-hold'], ctl['filter-release'])
    lp = lowpass()
    while line.running:
        sig = o(f)
        lp_sig = lp(sig, fline()**4 * ctl['filter-cutoff'], ctl['filter-resonance'])
        if ctl['filter-on'] > 0:
            sig = lp_sig
        yield sig * line()**4 * ctl['vol'] * note_vol


def gen(ctl):
    n1 = tntparser.make_events("o=2 amul=2 !7b 2 !6 | 2 !5 2 !4 3b !2")
    notes = tntparser.mix_events([n1])
    taker = tntparser.take_until(notes.loop())

    pos = tntparser.F(0)
    tmul = tntparser.F(FREQ) / BUFSIZE * 60 * 4
    p = poly()
    triggers = {}
    while taker.running or p:
        for _, etype, evalue in taker(pos):
            if etype == tntparser.NOTE_ON:
                ch, (o, offset), volume = evalue
                note = 12 + o*12 + offset
                if note in triggers:
                    triggers[note].set(False)
                t = triggers[note] = Trigger()
                p.add(synth(ctl['voice-1'], t, mtof(note), volume / 100))
            elif etype == tntparser.NOTE_OFF:
                ch, (o, offset), volume = evalue
                note = 12 + o*12 + offset
                if note in triggers:
                    triggers[note].set(False)

        yield p()
        pos += int(ctl['tempo']) / tmul


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
