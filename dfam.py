import numpy as np
import pysound as ps

square_t = ps.sinsum(4096, ps.square_partials(20))
tri_t = ps.sinsum(4096, ps.tri_partials(20))

svars = [ps.HStack(
    ps.HStack(
        ps.VSlide('vco-decay', 0, 0, 1, label='Decay'),
        ps.VSlide('1-env', 0, -1, 1, label='Env'),
        ps.VSlide('1-freq', 0, 0, 1, label='Freq'),
        ps.Radio('1-type', 0, ['square', 'tri'], label='Type'),
        label='OSC 1',
    ),
    ps.HStack(
        ps.VSlide('fm', 0, 0, 1, label='FM'),
        ps.VSlide('2-env', 0, -1, 1, label='Env'),
        ps.VSlide('2-freq', 0, 0, 1, label='Freq'),
        ps.Radio('2-type', 0, ['square', 'tri'], label='Type'),
        label='OSC 2',
    ),
    ps.HStack(
        ps.VSlide('1-volume', 1, 0, 1, label='OSC 1'),
        ps.VSlide('2-volume', 0, 0, 1, label='OSC 2'),
        ps.VSlide('noise-volume', 0, 0, 1, label='Noise'),
        label='Mixer',
    ),
    ps.HStack(
        ps.VSlide('cutoff', 1, 0, 1, label='Freq'),
        ps.VSlide('resonance', 0, 0, 1, label='Res'),
        ps.VSlide('vcf-decay', 0, 0, 1, label='Decay'),
        ps.VSlide('vcf-env', 0, -1, 1, label='Env'),
        ps.VSlide('vcf-mod', 0, 0, 1, label='Mod'),
        label='VCF',
    ),
    ps.HStack(
        ps.VSlide('volume', 0, 0, 1, label='Vol'),
        ps.VSlide('decay', 0, 0, 1, label='Decay'),
        label='VCA',
    ),
    label='DFAM'
)]


def vco(ctl, params):
    p = ps.phasor()
    def gen(freq):
        env = params['env_gen'].vco(5.0, ctl['vco-decay']**2*10000.0, 0.0, 5)**2.0
        f = (ctl['1-freq'] + ctl['1-env']*env)**2*1000.0
        sig = p(f)
        return ps.phasor_apply(sig, tri_t)
    return gen


def vcf(ctl, params):
    lp = ps.moog()
    def process(sig):
        env = params['env_gen'].vcf(5.0, ctl['vcf-decay']**2*10000.0, 0.0, 5)**2.0
        cutoff = (ctl['cutoff'] + ctl['vcf-env']*env)**2.0
        return lp(sig, cutoff, ctl['resonance'])
    return process


def synth(ctl, params):
    vco_s = vco(ctl, params)
    vcf_p = vcf(ctl, params)

    while True:
        freq = params['freq']
        sig = vco_s(freq)
        yield vcf_p(sig)


class env_factory:
    def __init__(self, ctl, params):
        self.ctl = ctl
        self.params = params
        self.amp = ps.env_adsr(stopped=True)
        self.vco = ps.env_adsr(stopped=True)
        self.vcf = ps.env_adsr(stopped=True)

    def __call__(self, attack, decay, sustain, release, hold=0.0):
        return self.amp(2.0, decay**2.0*10000, 0, 5)

    def stop(self, *args, **kwargs):
        self.amp.stop(*args, **kwargs)
        self.vco.stop(*args, **kwargs)
        self.vcf.stop(*args, **kwargs)

    def trigger(self):
        self.amp.trigger()
        self.vco.trigger()
        self.vcf.trigger()
