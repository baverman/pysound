#MIDI_SOURCE=16:0 python
from pysound import (
    GUI, Var, lowpass, phasor, env_ahr, VarGroup, mono,
    BUFSIZE, FREQ, poly_adsr, Player, midi_player, kbd_player)

import time
import tonator
import tntparser

import sh101

svars = [
    Var('volume', 0.5, 0, 1, resolution=0.01, midi_ctrl=0),
    Var('attack', 8, 1, 100),
    Var('sustain', 0.8, 0, 1, resolution=0.01),
    Var('release', 500, 1, 2000),
    Var('filter-on', 1, 0, 1),
    Var('filter-attack', 5, 1, 100),
    Var('filter-hold', 5, 1, 500),
    Var('filter-release', 260, 1, 2000),
    Var('filter-cutoff', 0.7, 0.0, 1.0, resolution=0.001, midi_ctrl=2),
    Var('filter-resonance', 0.7, 0.5, 1, resolution=0.001, midi_ctrl=3),
]

gui = GUI(
    Var('tempo', 220, 50, 600),
    VarGroup('voice-1', sh101.svars, midi_channel=0),
    # VarGroup('voice-2', svars, midi_channel=1),
    # VarGroup('voice-3', svars, midi_channel=2),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    preset_prefix='sh101-',
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


def gen(ctl):
    me = tntparser.make_events
    notes = tntparser.mix_events([
        # me("c=0 o=2 amul=2 k=C !7b 2 !6 2 !5 2 !4  3b !2"),
        me("c=0 o=3 amul=2 k=C 1 3 5 7 8 7 5 3"),
        # me("c=1 o=3 amul=2  4   -  4 -  4 -  7b -   6")
    ])
    taker = tntparser.take_until(notes.loop())

    player = Player()
    player.set_voice(0, mono(ctl['voice-1'], sh101.synth, env_exp=1))
    # player.set_voice(1, poly_adsr(ctl['voice-2'], synth))

    seq = tntparser.player_event_adapter(FREQ, BUFSIZE)
    kp = kbd_player(channel=0)
    while True:
        midi_player(ctl, player)
        kp(ctl, player)
        seq(player, taker, ctl['tempo'])
        yield player()


if __name__ == '__main__':
    # gui.play(gen(gui.ctl), output='/tmp/boo.wav')
    gui.play(gen(gui.ctl))
