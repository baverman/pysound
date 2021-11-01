#padsp python
import math
import wave
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import time
import ossaudiodev
import threading

import keys

FREQ = 44100
ROOT = 60

tau = np.pi * 2

_KEYNOTES1 = ['z', 's', 'x', 'd', 'c', 'v', 'g', 'b', 'h', 'n', 'j', 'm',
             'comma', 'l', 'period', 'semicolon', 'slash', 'Shift_R', 'Return']
_KEYNOTES2 = ['q', '2', 'w', '3', 'e', 'r', '5', 't', '6', 'y', '7', 'u',
              'i', '9', 'o', '0', 'p', 'bracketleft', 'equal', 'bracketright', 'BackSpace', 'backslash']
KEYNOTES = {it: i for i, it in enumerate(_KEYNOTES1)}
KEYNOTES.update({it: i + 12 for i, it in enumerate(_KEYNOTES2)})


def get_note(num):
    return 440 * 2 ** ((num-69)/12)


def sinsum(steps, *partials):
    x = np.linspace(0, 1, steps, endpoint=False, dtype=np.float32)
    y = 0
    for p, v in enumerate(partials, 1):
        y = y + v * np.sin(x*tau*p)
    y /= np.max(y)
    return x, y


class phasor:
    def __init__(self, phase=0, freq=FREQ):
        self.phase = 0
        self.freq = freq

    def __call__(self, freqs, bufsize=None):
        if bufsize is not None:
            freqs = np.full(bufsize, freqs, dtype=np.float32)
        result = np.cumsum(freqs / self.freq)
        if result[-1] != 0:
            result += self.phase
        result %= 1
        self.phase = result[-1]
        return result


class osc:
    def __init__(self, table, phase=0, freq=FREQ):
        self.phasor = phasor(phase, freq)
        self.table = table

    def __call__(self, freqs, bufsize=None):
        return np.interp(self.phasor(freqs, bufsize), *self.table).astype(np.float32)


def fft_plot(signal, n=2048):
    return np.fft.rfftfreq(n, 1/FREQ), 2/n*np.abs(np.fft.rfft(signal, n))


def open_wav_f32(fname):
    wave.WAVE_FORMAT_PCM = 3  # float
    f = wave.open(fname, 'wb')
    f.setnchannels(1)
    f.setsampwidth(4)
    f.setframerate(FREQ)
    return f


def scream(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
    return inner


@scream
def play(gen, n):
    dsp = ossaudiodev.open('w')
    dsp.setparameters(ossaudiodev.AFMT_S16_LE, 1, 44100)
    cnt = 0
    start = time.time()
    while True:
        while cnt < (time.time() - start) * FREQ + n:
            frame = next(gen)
            if np.max(np.abs(frame)) >= 1:
                print('@@@@@@')
            dsp.write((frame * 32767).astype(np.int16))
            cnt += len(frame)
        time.sleep(n/FREQ/2)
    dsp.sync()


sin_t = sinsum(1024, 1)
saw_t = sinsum(1024, *[1/i for i in range(1, 20)])
square_t = sinsum(1024, *[1/i if i%2 else 0 for i in range(1, 20)])
square_unlim_t = square_t[0], np.sign(square_t[1])
plt.plot(*square_unlim_t)
plt.show()

o = osc(square_t)
n = 512
frame = np.concatenate([o(200, n), o(400, n), o(500, n), o(600, n)]) * 0.2
# frame = np.concatenate([o(300, n)])
# plt.plot(frame)
# plt.plot(*fft_plot(frame))
# plt.xscale('log')
# plt.show()


def fm(ctl, base_freq, n):
    o = osc([sin_t, saw_t, square_t, square_unlim_t][int(ctl['waveform'].val)-1])
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


def gui(ctl, width):
    import tkinter as tk

    def kb(event):
        ctl['keys'] = keys.keyboard_state()

    master = tk.Tk()
    master.bind('<KeyPress>', kb)
    master.bind('<KeyRelease>', kb)

    for _, v in ctl.items():
        w = tk.Scale(master, label=v.label, from_=v.min, to=v.max, orient=tk.HORIZONTAL,
                     length=width, command=v, resolution=v.resolution)
        w.set(v.val)
        w.pack()

    tk.mainloop()


class Var:
    def __init__(self, label, value, min, max, resolution=1):
        self.label = label
        self.val = value
        self.min = min
        self.max = max
        self.resolution = resolution

    def __call__(self, event):
        self.val = float(event)


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

# with open_wav_f32('/tmp/boo.wav') as f:
#     ctl['keys'] = ['n']
#     g = ins(ctl, 512)
#     f.writeframes(next(g))
#     ctl['keys'] = []
#     for _ in range(3*FREQ//512):
#         f.writeframes(next(g))


t = threading.Thread(target=play, args=(ins(ctl, 512), 512), daemon=True)
t.start()

gui(ctl, 600)
