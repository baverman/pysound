import pysound as ps
from pysound.gui import GUI, Var
from pysound import tntparser

gui = GUI(
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    Var('tempo', 220, 50, 600),
    Var('attack', 100, 5, 10000, func='exp', label='A'),
    Var('decay', 100, 5, 10000, func='exp', label='D'),
    Var('sustain', 0, 0, 1, label='S'),
    Var('release', 5, 5, 10000, func='exp', label='R'),
    Var('cutoff', 1, 0, 1, func='exp'),
    Var('resonance', 0, 0, 1),
    preset_prefix='boo-',
)


tri_t = ps.sinsum(4096, ps.tri_partials(20))
square_t = ps.sinsum(4096, ps.square_partials(20))

def synth(ctl, params):
    env = params['env'] = ps.env_ahdsr(wait_decay=True)
    p = ps.phasor()
    flt = ps.lowpass()

    def gen():
        e = env(ctl['attack'], ctl['decay'], ctl['sustain'], ctl['release'])
        phase = p(params['freq'])
        sig = ps.phasor_apply(phase, tri_t)
        return flt(sig * e ** 2.0, ctl['cutoff'] * e ** 2.0, ctl['resonance'])

    return gen


def gen(ctl):
    player = ps.Player()
    # player.set_voice(0, mono(ctl, synth, retrigger=True))
    player.set_voice(0, ps.poly(ctl, synth))

    me = tntparser.make_events
    notes = tntparser.mix_events([
        me("c=0 o=2 amul=2 k=C gate=80 !7b 2 !6 2 !5 2 !4  3b !2"),
    ])
    taker = tntparser.take_until(notes.loop())

    kp = ps.kbd_player(channel=0)
    seq = tntparser.player_event_adapter(ps.FREQ, ps.BUFSIZE)
    while True:
        kp(ctl, player)
        # seq(player, taker, ctl['tempo'])
        yield player()


if __name__ == '__main__':
    # gui.play(gen(gui.ctl), output='/tmp/boo.wav')
    gui.play(gen(gui.ctl))
