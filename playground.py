#padsp python
from functools import partial
from pysound import osc, sin_t, fps, mtof, choicer, GUI, Var, env_ahr, mono
from tonator import Scales

gui = GUI(
    Var('tempo', 250, 50, 600),
    Var('master-volume', 0.5, 0, 1, resolution=0.01),
)


def synth(o, last):
    line = env_ahr(15, 100, 1000, last)
    while True:
        yield o() * line() ** 2, line.last


def gen(ctl):
    notes = choicer(it.value for it in Scales.major.notes)
    m = mono()
    o = osc(sin_t)
    while True:
        f = mtof(60 + next(notes))
        m.set(synth(partial(o, f), m.last))
        for _ in range(fps(60/ctl['tempo'])):
            yield m()


if __name__ == '__main__':
    # from pysound import render_to_file
    # render_to_file('/tmp/boo.wav', gui.ctl, gen(gui.ctl), 1)
    gui.play(gen(gui.ctl))
