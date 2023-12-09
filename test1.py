#MIDI_SOURCE=16:0 python
from pysound import (
    GUI, Var, lowpass, phasor, env_ahr, VarGroup, mono,
    BUFSIZE, FREQ, poly_adsr, Player, midi_player, kbd_player)

import time
import tonator
import tntparser

import sh101
import dfam

gui = GUI(
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    Var('tempo', 220, 50, 600),
    # VarGroup('dfam-1', dfam.svars),
    VarGroup('sh101-1', sh101.svars, midi_channel=0),
    # VarGroup('voice-2', svars, midi_channel=1),
    # VarGroup('voice-3', svars, midi_channel=2),
    preset_prefix='sh101-',
)


def gen(ctl):
    me = tntparser.make_events
    notes = tntparser.mix_events([
        me("c=0 o=2 amul=2 k=C !7b 2 !6 2 !5 2 !4  3b !2"),
        # me("c=0 o=3 amul=2 k=C 1 3 5 7 8 7 5 3"),
        # me("c=0 o=3 k=C amul=2  G F# E <E G <D- D | D- E <C G- >A <B | >B- >C <E- G F# <D | - E D <C '>B '>G '>A ''>>B")

        # me("c=1 o=3 amul=2  4   -  4 -  4 -  7b -   6")
        # me("c=0 o=3 amul=2 k=C 3 _ _ 3 _ _ 10 _ _ 3 _ _ 4 _ _ 5 | 1 _ _ 1 _ _ 8 _ _ 1 _ _ 6 _ _ 5"),
        # me("c=1 o=3 amul=2 k=C 1       1       1       1       | 1       1       1       1"),
    ])
    taker = tntparser.take_until(notes.loop())

    player = Player()
    # player.set_voice(0, mono(ctl['dfam-1'], dfam.synth, env_exp=2, env_factory=dfam.env_factory))
    # player.set_voice(0, mono(ctl['sh101-1'], sh101.synth, env_exp=2))
    player.set_voice(0, mono(ctl['sh101-1'], sh101.synth, env_exp=2))
    # player.set_voice(0, mono(ctl['voice-1'], sh101.synth, env_exp=1))
    # player.set_voice(1, mono(ctl['voice-2'], sh101.synth, env_exp=1))
    # player.set_voice(1, poly_adsr(ctl['voice-2'], synth))

    seq = tntparser.player_event_adapter(FREQ, BUFSIZE)
    kp = kbd_player(channel=0)
    while True:
        midi_player(ctl, player)
        kp(ctl, player)
        # seq(player, taker, ctl['tempo'])
        yield player()


if __name__ == '__main__':
    # gui.play(gen(gui.ctl), output='/tmp/boo.wav')
    gui.play(gen(gui.ctl))
