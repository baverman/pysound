# PD tutorial creating Roland SH-101 https://youtu.be/yni40cs8W24

from dataclasses import dataclass

import numpy as np

import pysound as ps
from pysound import (
    dcfilter, line, lowpass, moog, noise, phasor, phasor_apply, poly_saw, poly_square, shold,
    sinsum, square_partials)
from pysound.ngui import VScale, field, pow2

square_t = sinsum(2048, square_partials(20))

RANGES = (0.25, 0.5, 1.0, 2.0)


@dataclass
class Lfo:
    rate: float = field(0, range=(0, 1), widget=VScale)
    form: int = field(0, choices=('tri', 'square', 'shold', 'noise'), label='Type')


@dataclass
class Vco:
    porta: float = field(0, range=(0, 200), widget=VScale)
    mod: float = field(0, range=(0, 1), widget=VScale)
    octave: int = field(2, choices=('16', '8', '4', '2'))
    pw: float = field(0, range=(0, 0.4), widget=VScale)
    angle: float = field(1, range=(1, 1.7), label='Ang', widget=VScale)
    pw_mod: int = field(1, choices=('LFO', 'MAN', 'ENV'), label='PW Mod')


@dataclass
class Mixer:
    square: float = field(1, range=(0, 1), label='Sqr', widget=VScale)
    saw: float = field(0, range=(0, 1), widget=VScale)
    sub: float = field(0, range=(0, 1), widget=VScale)
    sub_type: int = field(0, choices=('1 oct', '2 oct', '2 pulse'), label='Sub Type')
    noise: float = field(0, range=(0, 1), widget=VScale)


@dataclass
class Vcf:
    cutoff: float = field(0.7, range=(0, 1), label='Freq', widget=VScale, map=pow2)
    resonance: float = field(0.7, range=(0, 1), label='Res', widget=VScale)
    env: float = field(0, range=(0, 1), widget=VScale)
    mod: float = field(0, range=(0, 1), widget=VScale)
    kbd: float = field(0, range=(0, 1), label='Kbd', widget=VScale)


@dataclass
class Env:
    attack: float = field(8, range=(1, 1000), map=pow2, label='A', widget=VScale)
    decay: float = field(200, range=(1, 1000), map=pow2, label='D', widget=VScale)
    sustain: float = field(0, range=(0, 1), label='S', widget=VScale)
    release: float = field(0, range=(0, 2000), map=pow2, label='R', widget=VScale)


@dataclass(kw_only=True)
class Sh101:
    volume: float = field(1, range=(0, 1), label='Vol')
    lfo: Lfo
    vco: Vco
    mixer: Mixer
    vcf: Vcf
    env: Env


def vco(ctl):
    saw_g = poly_saw()
    sqr_g = poly_square()
    sub_sqr_g = poly_square()
    pw_line = line(ctl.vco.pw)
    dc1 = dcfilter()
    dc2 = dcfilter()
    p = phasor()
    psub = phasor()

    def gen(freq, lfo_value, env_value):
        ff = freq
        freq = freq * 2 ** (ctl.vco.mod**3 * (lfo_value - 0.5) / 6)

        if ctl.vco.pw_mod == 0:
            pw = ctl.vco.pw * lfo_value
        elif ctl.vco.pw_mod == 1:
            pw = pw_line(ctl.vco.pw, 10)
        else:
            pw = ctl.vco.pw * env_value

        p(freq)
        saw = saw_g(p)
        square = dc1(sqr_g(p, pw))

        if ctl.mixer.sub_type > 0:
            sfreq = ff / 4.0
        else:
            sfreq = ff / 2.0

        psig = psub(sfreq)

        if ctl.mixer.sub_type == 2:
            sub = dc2(sub_sqr_g(psub, 0.166))
        else:
            sub = dc2(phasor_apply(psig, square_t))

        return (square * ctl.mixer.square
                + saw * ctl.mixer.saw
                + sub * ctl.mixer.sub
                + noise() * ctl.mixer.noise / 2) / 4

    return gen


def vcf(ctl):
    lp = moog()

    def process(sig, env_value, lfo_value):
        alpha = (ctl.vcf.cutoff
                 + ctl.vcf.mod * lfo_value
                 + ctl.vcf.env * env_value) ** 2
        return lp(sig, alpha, ctl.vcf.resonance)

    return process


def lfo(ctl):
    p = phasor()
    sh = shold()
    lp = lowpass()

    def sgen():
        f = ctl.lfo.form
        freq = ctl.lfo.rate**3 * 100
        psig = p(freq)
        if f == 0:  # tri
            psig = np.abs(psig - 0.5) * 2
        elif f == 1:  # square
            psig = (psig > 0.5).astype(np.float32)
        elif f == 2:  # s-hold
            n = (noise() + 1) / 2
            psig = lp(sh(n, psig), 0.1)
        elif f == 3:  # noise
            psig = (noise() + 1) / 2

        return psig

    return sgen


def synth(ctl, params):
    env = ps.env_adsr(wait_decay=True)
    params.env = env

    vco_s = vco(ctl)
    vcf_p = vcf(ctl)
    lfo_s = lfo(ctl)
    freq_line = line()

    def gen():
        freq = params.freq * RANGES[int(ctl.vco.octave)]
        if ctl.vco.porta > 0:
            freq = freq_line(freq, ctl.vco.porta)
        lfo_value = lfo_s()
        env_value = env(ctl.env.attack, ctl.env.decay, ctl.env.sustain, ctl.env.release)
        return vcf_p(vco_s(freq, lfo_value, env_value), env_value, lfo_value) * env_value

    return gen
