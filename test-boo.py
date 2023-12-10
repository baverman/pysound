import pysound as ps
from pysound import (
    GUI, Var, lowpass, phasor, env_ahdsr, VarGroup, mono, phasor_apply, HStack, VSlide,
    BUFSIZE, FREQ, Player, midi_player, kbd_player, sinsum, tri_partials)

import tntparser

gui = GUI(
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    Var('tempo', 220, 50, 600),
    Var('attack', 0.1, 0, 1, label='A'),
    Var('decay', 0.1, 0, 1, label='D'),
    Var('sustain', 0, 0, 1, label='S'),
    Var('release', 0, 0, 1, label='R'),
    preset_prefix='boo-',
)


tri_t = sinsum(4096, tri_partials(20))
square_t = ps.sinsum(4096, ps.square_partials(20))

def synth(ctl, params):
    env = env_ahdsr(stopped=True, speed=1)
    p = phasor()
    pgate = 0

    while True:
        if params['gate'] != pgate:
            pgate = params['gate']
            if pgate:
                env.trigger()
            else:
                env.stop(True)

        if params['retrigger']:
            env.trigger()
            params['retrigger'] = False

        e = env(ctl['attack']**2.0*10000, ctl['decay']**2.0*10000, ctl['sustain'], ctl['release']**2.0*10000)
        sig = p(params['freq'])
        sig = phasor_apply(sig, ps.sin_t)
        yield sig * e


def gen(ctl):
    player = Player()
    player.set_voice(0, mono(ctl, synth))

    me = tntparser.make_events
    notes = tntparser.mix_events([
        me("c=0 o=2 amul=2 k=C gate=80 !7b 2 !6 2 !5 2 !4  3b !2"),
    ])
    taker = tntparser.take_until(notes.loop())

    kp = kbd_player(channel=0)
    seq = tntparser.player_event_adapter(FREQ, BUFSIZE)
    while True:
        kp(ctl, player)
        # seq(player, taker, ctl['tempo'])
        yield player()


if __name__ == '__main__':
    # gui.play(gen(gui.ctl), output='/tmp/boo.wav')
    gui.play(gen(gui.ctl))
