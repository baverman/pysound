# PD tutorial creating Roland SH-101 https://youtu.be/yni40cs8W24

import numpy as np
from pysound import (
    noise, lowpass, phasor, line, HStack, VSlide, Radio, ensure_buf,
    shold, poly_saw, poly_square, dcfilter
)

svars = [HStack(
    HStack(
        VSlide('lfo-rate', 0, 0, 1, label='Rate'),
        Radio('lfo-form', 0, ['tri', 'square', 'shold', 'noise'], label='Type'),
        label='LFO',
    ),
    HStack(
        VSlide('volume', 1, 0, 1, label='Vol'),
        VSlide('porta', 0, 0, 200),
        VSlide('vco-mod', 0, 0, 1, label='Mod'),
        Radio('range', 2, ['16', '8', '4', '2']),
        VSlide('pw', 0, 0, 0.4, label='PW'),
        VSlide('sq-angle', 1, 1, 1.7, label='Ang'),
        # VSlide('pw-mod', 0, 0, 0.4, label='Mod'),
        Radio('pw-mod-type', 1, ['LFO', 'MAN', 'ENV'], label='Type'),
        label='VCO',
    ),
    HStack(
        VSlide('square', 1, 0, 1, label='Sqr'),
        VSlide('saw', 0, 0, 1),
        VSlide('sub', 0, 0, 1),
        Radio('sub-type', 0, ['1 oct', '2 oct', '2 pulse'], label='Type'),
        VSlide('noise', 0, 0, 1),
        label='Source Mixer'
    ),
    HStack(
        VSlide('filter-cutoff', 0.7, 0.0, 1.0, label='Freq'),
        VSlide('filter-resonance', 0.7, 0.0, 1, label='Res'),
        VSlide('filter-env', 0, 0, 1, label='Env'),
        VSlide('filter-mod', 0, 0, 1, label='Mod'),
        VSlide('filter-kbd', 0, 0, 1, label='Kbd'),
        label='VCF'
    ),
    HStack(
        VSlide('attack', 8, 1, 100, label='A'),
        VSlide('decay', 200, 1, 1000, label='D'),
        VSlide('sustain', 0, 0, 1, label='S'),
        VSlide('release', 0, 1, 2000, label='R'),
        label='Env',
    ),
    label='SH-101'
)]


def vco(ctl, params):
    saw_g = poly_saw()
    sqr_g = poly_square()
    sub_sqr_g = poly_square()
    pw_line = line(ctl['pw'])

    dc1 = dcfilter()
    dc2 = dcfilter()
    def gen(freq):
        ff = freq
        freq = freq * 2**(ctl['vco-mod']**3 * (params['lfo-freq'] - 0.5)/6)
        if ctl['pw-mod-type'] == 0:
            pw = ctl['pw'] * params['lfo-freq']
        elif ctl['pw-mod-type'] == 1:
            pw = pw_line(ctl['pw'], 10)
        else:
            pw = ctl['pw'] * params['env']

        saw = saw_g(freq)
        square = dc1(sqr_g(freq, pw))

        if ctl['sub-type'] > 0:
            sfreq = ff / 4.0
        else:
            sfreq = ff / 2.0

        if ctl['sub-type'] == 2:
            sub = dc2(sub_sqr_g(sfreq, 0.166))
        else:
            sub = sub_sqr_g(sfreq, 0.0)

        return (square * ctl['square']
                + saw * ctl['saw']
                + sub * ctl['sub']
                + noise() * ctl['noise'] / 2) / 4

    return gen


def vcf(ctl, params):
    lp = lowpass()
    def process(sig):
        alpha = (ctl['filter-cutoff'] + ctl['filter-mod']*params['lfo-freq'] + ctl['filter-env']*(params['env']**4))**2
        return lp(sig, alpha, ctl['filter-resonance'])
    return process


def lfo(ctl, params):
    p = phasor()
    sh = shold()
    lp = lowpass()
    def sgen():
        f = ctl['lfo-form']
        freq = ctl['lfo-rate']**3 * 100
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
    vco_s = vco(ctl, params)
    vcf_p = vcf(ctl, params)
    lfo_s = lfo(ctl, params)

    freq_line = line()
    R = [0.25, 0.5, 1, 2]

    while True:
        freq = params['freq'] * R[int(ctl['range'])]
        if ctl['porta'] > 0:
            freq = freq_line(freq, ctl['porta'])
        params['lfo-freq'] = lfo_s()
        yield vcf_p(vco_s(freq))
