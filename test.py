#padsp python
import math
import wave
import matplotlib.pyplot as plt
from itertools import islice
import time
import ossaudiodev
import threading

import keys


_KEYNOTES1 = ['z', 's', 'x', 'd', 'c', 'v', 'g', 'b', 'h', 'n', 'j', 'm',
             'comma', 'l', 'period', 'semicolon', 'slash', 'Shift_R', 'Return']
_KEYNOTES2 = ['q', '2', 'w', '3', 'e', 'r', '5', 't', '6', 'y', '7', 'u',
              'i', '9', 'o', '0', 'p', 'bracketleft', 'equal', 'bracketright', 'BackSpace', 'backslash']
KEYNOTES = {it: i for i, it in enumerate(_KEYNOTES1)}
KEYNOTES.update({it: i + 12 for i, it in enumerate(_KEYNOTES2)})


def fm(ctl, base_freq, n):
    o = osc([sin_t, saw_t, square_t, square_unlim_t][int(ctl['waveform'])-1])
    f = osc(sin_t)
    while True:
        harm = ctl['harm'].val
        depth = base_freq * harm * ctl['mod-index'].val
        mf = f(base_freq * harm, n)
        frame = o(base_freq + mf * depth)
        yield frame


def adsr(scount, g, n, attack, decay, sustain, release):
    acnt = int(attack[0] * FREQ)
    dcnt = int(decay[0] * FREQ)

    if 'start_env' not in g:
        env = np.concatenate([np.linspace(0, attack[1], acnt, dtype=np.float32, endpoint=False),
                              np.linspace(attack[1], decay[1], dcnt, dtype=np.float32),
                              np.full(n, decay[1], dtype=np.float32)])
        g['start_env'] = env

    adelta = scount - g['start']
    if adelta > acnt + dcnt:
        if g['release'] >= 0:
            rcnt = int(release * FREQ)
            if 'release_env' not in g:
                g['release_env'] = np.concatenate([np.linspace(decay[1], 0, rcnt, dtype=np.float32),
                                                   np.full(n, 0, dtype=np.float32)])

            rdelta = scount - g['release']
            if rdelta > rcnt:
                g['stopped'] = True
                return g['release_env'][-n:]

            return g['release_env'][rdelta:rdelta+n]

        return g['start_env'][-n:]

    if g['release'] >= 0:
        g['release'] = scount + n
    return g['start_env'][adelta:adelta+n]


def ins(ctl, n):
    gens = {}
    rgens = {}
    scount = 0
    sustain = -1
    while True:
        attack = ctl['attack'].val, 1
        decay = ctl['decay'].val, 0.7
        release = ctl['release'].val
        pressed = ctl.get('keys', [])
        new_keys = set(it for it in pressed  if it in KEYNOTES) - set(gens)
        depressed = set(gens) - set(pressed)

        for key in new_keys:
            gens[key] = {
                'g': fm(ctl, get_note(12*ctl['octave'].val + KEYNOTES[key]), n),
                'start': scount,
                'release': -1,
                'stopped': False
            }

        for key in depressed:
            g = gens.pop(key)
            g['release'] = scount
            rgens[id(g)] = g

        result = np.full(n, 0, dtype=np.float32)
        for key, g in list(gens.items()) + list(rgens.items()):
            e = adsr(scount, g, n, attack, decay, sustain, release)
            result += next(g['g']) * e
            if g['stopped']:
                rgens.pop(key)

        yield result * ctl['volume'].val
        scount += n


ctl = {
    # 'base': Var('Base', 330, 50, 600),
    'octave': Var('Octave', 4, 0, 6, resolution=1),
    'attack': Var('Attack', 0.05, 0, 0.5, resolution=0.001),
    'decay': Var('Decay', 0.1, 0, 0.5, resolution=0.001),
    'release': Var('Release', 0.5, 0, 5, resolution=0.01),
    'waveform': Var('Wave form', 1, 1, 4, resolution=1),
    'harm': Var('Harmonicity', 0, 0, 4, resolution=0.01),
    'mod-index': Var('Modulation index', 0, 0, 10, resolution=0.1),
    'volume': Var('Volume', 0.2, 0, 1, resolution=0.1),
}

t = threading.Thread(target=play, args=(ins(ctl, 512), 512), daemon=True)
t.start()

gui(ctl, 600)
