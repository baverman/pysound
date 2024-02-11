# python
# Polyphony
# https://www.youtube.com/watch?v=jGgu77pCfHU&list=PLqJgTfn3kSMW3AAAl2liJRKd-7DhZwLlq&index=3
from pysound import GUI, Var, mtof, osc, sin_t, fps, choicer, poly, env_ahr
from tonator import Scales

gui = GUI(
    Var('tempo', 250, 50, 600),
    Var('attack', 8, 1, 100),
    Var('hold', 70, 1, 1000),
    Var('release', 1200, 1, 2000),
    Var('master-volume', 0.2, 0, 1, resolution=0.01),
)


def synth(ctl, f):
    o = osc(sin_t)
    line = env_ahr(ctl['attack'], ctl['hold'], ctl['release'])
    while line.running:
        yield o(f) * line()**4


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    p = poly()
    while True:
        f = mtof(60 + next(notes))
        p.add(synth(ctl, f))
        for _ in range(fps(60/ctl['tempo'])):
            yield p()


if __name__ == '__main__':
    gui.play(gen(gui.ctl))
