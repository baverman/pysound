#padsp python
# FM
# https://www.youtube.com/watch?v=mvtN7de6Oko&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=6
from pysound import GUI, Var, mtof, osc, sin_t, fps, choicer, poly, env_ahr
from tonator import Scales

gui = GUI(
    Var('tempo', 330, 50, 600),
    Var('harmonicity', 1.5, 0, 4, resolution=0.01),
    Var('modulation-index', 4, 0, 10, resolution=0.1),
    Var('modulation-release', 200, 1, 1000),
    Var('attack', 10, 1, 100),
    Var('hold', 50, 1, 1000),
    Var('release', 500, 1, 2000),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def fm(ctl):
    o = osc(sin_t)
    m = osc(sin_t)
    def sig(f, mline):
        harm = ctl['harmonicity']
        depth = f * harm * ctl['modulation-index'] * mline
        mf = m(f * harm)
        return o(f + mf * depth)
    return sig


def synth(ctl, f):
    o = fm(ctl)
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    mline = env_ahr(10, 1, ctl['modulation-release'])
    while line.running:
        yield o(f, mline()) * line()**4


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    while True:
        f = mtof(48 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            yield p()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
