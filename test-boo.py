# python
from dataclasses import dataclass

import pysound as ps
from pysound import modules as psm
from pysound import gui, tntparser
from pysound.gui import field, pow2


@dataclass
class Osc:
    detune: float = field(0, range=(0, 0.02))
    pwm: float = field(0, range=(0, 0.45), nl=True)


@dataclass
class Env:
    attack: float = field(100, range=(5, 10000), map=pow2)
    decay: float = field(100, range=(5, 10000), map=pow2)
    sustain: float = field(0, range=(0, 1))
    release: float = field(5, range=(5, 10000), map=pow2)


@dataclass
class Filter:
    cutoff: float = field(1, range=(0, 1), map=pow2)
    resonance: float = field(0, range=(0, 1))
    fattack: float = field(100, range=(5, 10000), map=pow2)
    fenv: float = field(0, range=(0, 1))


@dataclass
class Fx:
    delay: float = field(0, range=(0, 2), map=pow2)
    feedback: float = field(0, range=(0, 1))


@dataclass
class Arp:
    length: float = field(7, range=(1, 16), resolution=1)
    div: float = field(4, range=(1, 16), resolution=1)
    octaves: float = field(1, range=(0, 4), resolution=1)


@dataclass(kw_only=True)
class Controls:
    master_volume: float = field(0.2, range=(0, 1), resolution=0.01)
    tempo: float = field(220, range=(50, 600), nl=True)
    osc: Osc
    env: Env
    filter: Filter
    fx: Fx
    arp: Arp


tri_t = ps.sinsum(4096, ps.tri_partials(20))
square_t = ps.sinsum(4096, ps.square_partials(20))


def synth(ctl, params):
    env = ps.env_adsr(wait_decay=True)
    fenv = ps.env_adsr(wait_decay=True)
    params.env = ps.menv(env, fenv)
    p1 = ps.phasor()
    p2 = ps.phasor()
    p3 = ps.phasor()
    flt = ps.pole2()
    osc = ps.poly_saw()
    # osc = ps.poly_square()

    def gen():
        e = env(ctl.env.attack, ctl.env.decay, ctl.env.sustain, ctl.env.release)
        fe = fenv(ctl.filter.fattack, ctl.filter.fattack, 0, 0)
        f = params.freq
        p1(f)
        p2(f * 2 ** ctl.osc.detune)
        p3(f * 2 ** (-ctl.osc.detune))
        sig = (osc(p1, ctl.osc.pwm) + osc(p2, ctl.osc.pwm) + osc(p3, ctl.osc.pwm)) / 3
        sig = flt(sig * e, ctl.filter.cutoff + ctl.filter.fenv * fe, ctl.filter.resonance)
        return sig

    return gen


def gen(ctl):
    player = ps.Player()
    # player.set_voice(0, mono(ctl, synth, retrigger=True))
    player.set_voice(0, psm.arp(ctl.arp, ctl, ps.poly(ctl, synth)))

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
        # seq(player, taker, ctl.tempo)
        yield delay(player(), ps.sps(ctl.fx.delay), ctl.fx.feedback)
        # yield player()


if __name__ == '__main__':
    ctl = gui.make(Controls)
    app = gui.App(Controls)
    app.bind(ctl)
    app.play(gen(ctl))
