#MIDI_SOURCE=16:0 padsp python
from pysound import (
    GUI, Var, lowpass, phasor, env_ahr, VarGroup,
    BUFSIZE, FREQ, poly_adsr, Player)

import time
import tonator
import tntparser

svars = [
    Var('volume', 0.5, 0, 1, resolution=0.01, midi_ctrl=0),
    Var('attack', 8, 1, 100),
    Var('sustain', 0.8, 0, 1, resolution=0.01),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-hold', 5, 1, 500),
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


def synth(ctl, params):
    o = phasor()
    fline = env_ahr(ctl['filter-attack'], ctl['filter-hold'], ctl['filter-release'])
    lp = lowpass()
    while True:
        sig = o(params['freq'])
        if ctl['filter-on'] > 0:
            sig = lp(sig, fline()**4 * ctl['filter-cutoff'], ctl['filter-resonance'])
        else:
            sig = lp(sig, ctl['filter-cutoff'], ctl['filter-resonance'])
        yield sig


def midi_player(player):
    notes = player.ctl['midi_notes']
    while notes:
        etype, (ch, note, velocity) = notes.popleft()
        print(time.time(), etype, ch, note)
        if etype == 0:
            player.note_off(ch, note)
        elif etype == 1:
            player.note_on(ch, note, velocity / 127)


def gen(ctl):
    n1 = tntparser.make_events("c=0 o=2 amul=2  !7b 2 !6 2 !5 2 !4  3b !2")
    # n2 = tntparser.make_events("c=1 o=3 amul=2  4   -  4 -  4 -  7b -   6")
    notes = tntparser.mix_events([n1])
    taker = tntparser.take_until(notes.loop())

    player = Player()
    player.set_voice(0, poly_adsr(ctl['voice-1'], synth))

    seq = tntparser.player_event_adapter(FREQ, BUFSIZE)
    while True:
        seq(player, taker, ctl['tempo'])
        # midi_player(player)
        yield player()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
