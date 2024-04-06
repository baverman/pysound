# python
import pysound as ps
from pysound.gui import GUI, Var, VarGroup, VSlide, HStack
from pysound import tntparser

env_vars = HStack(
    VSlide('attack', 100, 5, 10000, func='exp', label='A'),
    VSlide('decay', 100, 5, 10000, func='exp', label='D'),
    VSlide('sustain', 0, 0, 1, label='S'),
    VSlide('release', 5, 5, 10000, func='exp', label='R'),
    label = 'Env',
)

f_vars = HStack(
    VSlide('cutoff', 1, 0, 1, func='exp', label='F'),
    VSlide('resonance', 0, 0, 1, label='R'),
    VSlide('fattack', 100, 5, 10000, func='exp', label='AD'),
    VSlide('fenv', 0, 0, 1, label='E'),
    label='Filter',
)

fx_vars = HStack(
    VSlide('delay', 0, 0, 2, func='exp'),
    VSlide('feedback', 0, 0, 1),
    label='FX',
)

osc_vars = HStack(
    VSlide('detune', 0, 0, 0.02),
    VSlide('pwm', 0, 0, 0.45),
    label='Osc',
)

gui = GUI(
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
    Var('tempo', 220, 50, 600),
    HStack(osc_vars, env_vars, f_vars),
    HStack(fx_vars),
    preset_prefix='boo-',
)


tri_t = ps.sinsum(4096, ps.tri_partials(20))
square_t = ps.sinsum(4096, ps.square_partials(20))

def synth(ctl, params):
    env = ps.env_adsr(wait_decay=True)
    fenv = ps.env_adsr(wait_decay=True)
    params['env'] = ps.menv(env, fenv)
    p1 = ps.phasor()
    p2 = ps.phasor()
    p3 = ps.phasor()
    flt = ps.pole2()
    osc = ps.poly_saw()
    # osc = ps.poly_square()

    def gen():
        e = env(ctl['attack'], ctl['decay'], ctl['sustain'], ctl['release'])
        fe = fenv(ctl['fattack'], ctl['fattack'], 0, 0)
        f = params['freq']
        p1(f)
        p2(f*2**ctl['detune'])
        p3(f*2**(-ctl['detune']))
        sig = (osc(p1, ctl['pwm']) + osc(p2, ctl['pwm']) + osc(p3, ctl['pwm'])) / 3
        sig = flt(sig * e, ctl['cutoff'] + ctl['fenv'] * fe, ctl['resonance'])
        return sig

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

    delay = ps.delay(3)

    kp = ps.kbd_player(channel=0)
    seq = tntparser.player_event_adapter(ps.FREQ, ps.BUFSIZE)
    while True:
        kp(ctl, player)
        # seq(player, taker, ctl['tempo'])
        yield delay(player(), ps.sps(ctl['delay']), ctl['feedback'])
        # yield player()


if __name__ == '__main__':
    # gui.play(gen(gui.ctl), output='/tmp/boo.wav')
    gui.play(gen(gui.ctl))
