# PD tutorial creating Roland SH-101 https://youtu.be/yni40cs8W24

import numpy as np
from pysound import (
    saw_t, square_t, osc, noise, Var, lowpass, phasor, phasor_apply, sin_t,
    tri_t, line
)

svars = [
    Var('volume', 0.5, 0, 1, resolution=0.01),
    Var('range', 2, 0, 3),
    Var('porta', 0, 0, 200),
    Var('pw', 0, 0, 0.5, resolution=0.001),
    Var('square', 1, 0, 1, resolution=0.01),
    Var('saw', 0, 0, 1, resolution=0.01),
    Var('sub', 0, 0, 1, resolution=0.01),
    Var('noise', 0, 0, 1, resolution=0.01),
    Var('attack', 8, 1, 100),
    Var('decay', 200, 1, 1000),
    Var('sustain', 0, 0, 1, resolution=0.01),
    Var('release', 0, 1, 2000),
    Var('filter-cutoff', 0.7, 0.0, 1.0, resolution=0.001),
    Var('filter-resonance', 0.7, 0.5, 1, resolution=0.001),
    Var('filter-mod-lfo', 0, 0, 1, resolution=0.01),
    Var('lfo-rate', 0, 0, 1, resolution=0.001)
]


def vco(ctl):
    p1 = phasor()
    p2 = phasor()
    pw_line = line(ctl['pw'])
    def gen(freq):
        p1sig = p1(freq)
        p2sig = p2(freq / 2)
        wp1sig =(p1sig + np.clip(0.5 - pw_line(ctl['pw'], 10), 0.01, 0.5)) % 1.0
        square = phasor_apply(p1sig, saw_t) - phasor_apply(wp1sig, saw_t)
        saw = phasor_apply(p1sig, saw_t)
        sub = phasor_apply(p2sig, tri_t)
        return (square * ctl['square']
                + saw * ctl['saw']
                + sub * ctl['sub'] * 4
                + noise() * ctl['noise'] / 2) / 4

    return gen


def vcf(ctl, params):
    lp = lowpass()
    def process(sig):
        alpha = (ctl['filter-cutoff'] + ctl['filter-mod-lfo']*params['lfo-freq'])**2
        return lp(sig, alpha, ctl['filter-resonance'])
    return process


def lfo(ctl, params):
    p = phasor()
    def sgen():
        psig = p(ctl['lfo-rate']**4 * 500)
        return np.abs(psig - 0.5) * 4 - 1
    return sgen


def synth(ctl, params):
    vco_s = vco(ctl)
    vcf_p = vcf(ctl, params)
    lfo_s = lfo(ctl, params)

    freq_line = line()
    R = [0.25, 0.5, 1, 2]

    while True:
        # params['freq'] = 116
        freq = freq_line(params['freq'] * R[int(ctl['range'])], ctl['porta'])
        params['lfo-freq'] = lfo_s()
        yield vcf_p(vco_s(freq))
