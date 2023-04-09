#MIDI_SOURCE=16:0 padsp python
from pysound import (
    GUI, Var, lowpass, phasor, env_ahr, VarGroup,
    env_adsr, BUFSIZE, FREQ, poly, Player)

import tonator
import tntparser

svars = [
    Var('vol', 0.5, 0, 1, resolution=0.01, midi_ctrl=0),
    Var('attack', 8, 1, 100),
    Var('sustain', 0.8, 0, 1, resolution=0.01),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-sustain', 1, 0, 1, resolution=0.01, midi_ctrl=1),
    Var('filter-release', 260, 1, 2000),
    Var('filter-cutoff', 15000, 100, 20000, midi_ctrl=2),
    Var('filter-resonance', 0.7, 0.5, 1, resolution=0.001, midi_ctrl=3),
]

gui = GUI(
    Var('tempo', 320, 50, 600),
    VarGroup('voice-1', svars, midi_channel=0),
    VarGroup('voice-2', svars, midi_channel=1),
    # VarGroup('voice-3', svars, midi_channel=2),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    preset_prefix='test1-',
)


def synth(ctl, trig, f, note_vol):
    o = phasor()
    line = env_adsr(trig, ctl['attack'], ctl['attack'], ctl['sustain'], ctl['release'])
    fline = env_adsr(trig, ctl['filter-attack'], ctl['filter-attack'], ctl['filter-sustain'], ctl['filter-release'])
    lp = lowpass()
    while line.running:
        sig = o(f)
        if ctl['filter-on'] > 0:
            sig = lp(sig, fline()**4 * ctl['filter-cutoff'], ctl['filter-resonance'])
        else:
            sig = lp(sig, ctl['filter-cutoff'], ctl['filter-resonance'])
        yield sig * line()**4 * ctl['vol'] * note_vol


def play_event_adapter(player, etype, evalue):
    if etype == tntparser.NOTE_ON:
        ch, (o, offset), volume = evalue
        note = 12 + o*12 + offset
        player.note_on(ch, note, volume / 100)
    elif etype == tntparser.NOTE_OFF:
        ch, (o, offset), _ = evalue
        note = 12 + o*12 + offset
        player.note_off(ch, note)


def gen(ctl):
    n1 = tntparser.make_events("c=0 o=2 amul=2  !7b 2 !6 2 !5 2 !4  3b !2")
    # n2 = tntparser.make_events("c=1 o=3 amul=2  4   -  4 -  4 -  7b -   6")
    notes = tntparser.mix_events([n1])
    taker = tntparser.take_until(notes.loop())

    p = poly()
    player = Player(ctl, p)
    player.set_voice(0, 'voice-1', synth)
    player.set_voice(1, 'voice-2', synth)
    player.set_voice(2, 'voice-3', synth)

    pos = tntparser.F(0)
    tmul = tntparser.F(FREQ) / BUFSIZE * 60 * 4
    while taker.running or p or True:
        for _, etype, evalue in taker(pos):
            play_event_adapter(player, etype, evalue)

        while ctl['midi_notes']:
            etype, (ch, note, velocity) = ctl['midi_notes'].popleft()
            if etype == 0:
                player.note_off(ch, note)
            elif etype == 1:
                player.note_on(ch, note, velocity / 127)

        yield p()
        pos += int(ctl['tempo']) / tmul


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
