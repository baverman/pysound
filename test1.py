#MIDI_SOURCE=16:0 python
from dataclasses import dataclass

import sh101
# import dfam
from pysound import BUFSIZE, FREQ, Player, gui, kbd_player, mono, tntparser
from pysound.gui import field
from pysound.gui import field


@dataclass(kw_only=True)
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(220, range=(50, 600), nl=True)
    # dfam: dfam.Dfam
    sh101: sh101.Sh101
    # voice_2: sh101.Sh101
    # voice_3: sh101.Sh101


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
    # player.set_voice(0, mono(ctl.dfam, dfam.synth))
    player.set_voice(0, mono(ctl.sh101, sh101.synth))
    # player.set_voice(0, mono(ctl.voice_1, sh101.synth))
    # player.set_voice(1, mono(ctl.voice_2, sh101.synth))

    seq = tntparser.player_event_adapter(FREQ, BUFSIZE)
    kp = kbd_player(channel=0)
    while True:
        kp(ctl, player)
        # seq(player, taker, ctl.tempo)
        yield player()


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
