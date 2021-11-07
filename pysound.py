import time
import numpy as np
import ossaudiodev
import threading
import wave
import random
import pprint

import filters

FREQ = 44100
BUFSIZE = 512

tau = np.pi * 2


def choicer(values):
    values = list(values)
    last = None
    while True:
        random.shuffle(values)
        if values[0] == last:
            values[0], values[1] = values[1], values[0]
        yield from values
        last = values[-1]


def mtof(num):
    return 440 * 2 ** ((num-69)/12)


def sinsum(steps, *partials):
    x = np.linspace(0, 1, steps, endpoint=False, dtype=np.float32)
    y = 0
    for p, v in enumerate(partials, 1):
        y = y + v * np.sin(x*tau*p)
    y /= np.max(y)
    return x, y


sin_t = sinsum(1024, 1)
saw_t = sinsum(1024, *[1/i for i in range(1, 20)])
square_t = sinsum(1024, *[1/i if i%2 else 0 for i in range(1, 20)])
square_unlim_t = square_t[0], np.sign(square_t[1])


class phasor:
    def __init__(self, phase=0, freq=FREQ):
        self.phase = 0
        self.freq = freq

    def __call__(self, freq):
        if type(freq) in (int, float):
            freq = np.full(BUFSIZE, freq, dtype=np.float32)
        result = np.cumsum(freq / self.freq)
        if result[-1] != 0:
            result += self.phase
        result %= 1
        self.phase = result[-1]
        return result


class osc:
    def __init__(self, table, phase=0, freq=FREQ):
        self.phasor = phasor(phase, freq)
        self.table = table

    def __call__(self, freq):
        return np.interp(self.phasor(freq), *self.table).astype(np.float32)


def env_ahr(attack, hold, release, last=None):
    last = last or 0
    acnt = sps(attack / 1000)
    hcnt = sps(hold / 1000)
    rcnt = sps(release / 1000)
    e = np.concatenate([
        np.linspace(last, 1, acnt, endpoint=False, dtype=np.float32),
        np.full(hcnt, 1, dtype=np.float32),
        np.linspace(1, 0, rcnt, dtype=np.float32),
        np.full(BUFSIZE, 0, dtype=np.float32),
    ])

    samples = 0
    def gen():
        nonlocal samples
        if samples < acnt + hcnt + rcnt:
            result = e[samples:samples+BUFSIZE]
        else:
            result = e[-BUFSIZE:]
            gen.running = False
        samples += BUFSIZE
        gen.last = result[-1]
        return result

    gen.running = True
    return gen


class poly:
    def __init__(self):
        self.gens = set()

    def add(self, gen):
        self.gens.add(gen)

    def __call__(self):
        toremove = []
        result = np.full(BUFSIZE, 0, dtype=np.float32)
        for g in self.gens:
            data = next(g, None)
            if data is None:
                toremove.append(g)
            else:
                result += data
        self.gens.difference_update(toremove)
        return result


def fft_plot(signal, window=2048, crop=0):
    return np.fft.rfftfreq(window, 1/FREQ)[crop:], 2/window*np.abs(np.fft.rfft(signal, window))[crop:]


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


class GUI:
    def __init__(self, *controls):
        self.controls = controls
        self.ctl = {it.name: it.val for it in controls}

    def __iter__(self):
        return iter(self.controls)

    def command(self, control):
        def callback(value):
            self.ctl[control.name] = float(value)
        return callback

    def play(self, gen, width=600):
        t = threading.Thread(target=play, args=(self.ctl, gen), daemon=True)
        t.start()
        show_window(self, width)


class Var:
    def __init__(self, name, value, min, max, resolution=1, label=None):
        self.name = name
        self.label = label or name.replace('-', ' ').title()
        self.val = value
        self.min = min
        self.max = max
        self.resolution = resolution


def show_window(controls, width):
    import tkinter as tk
    import keys

    controls.ctl['keys'] = []
    def kb(_):
        controls.ctl['keys'] = keys.keyboard_state()

    master = tk.Tk()
    master.bind('<KeyPress>', kb)
    master.bind('<KeyRelease>', kb)

    for v in controls:
        w = tk.Scale(master, label=v.label, from_=v.min, to=v.max, orient=tk.HORIZONTAL,
                     length=width, command=controls.command(v), resolution=v.resolution)
        w.set(v.val)
        w.pack()

    tk.mainloop()
    pprint.pprint(controls.ctl)


def fps(duration=1):
    return int(duration * FREQ / BUFSIZE)


def sps(duration=1):
    return int(duration * FREQ)


def lowpass():
    result = np.empty(BUFSIZE, dtype=np.float32)
    state = np.zeros(2, dtype=np.float32)

    def gen(data, cutoff, resonance=0):
        if type(cutoff) in (int, float):
            alpha = np.full(data.shape[0], cutoff/FREQ, dtype=np.float32)
        else:
            alpha = cutoff / FREQ

        filters.lowpass(result, data, alpha, resonance, state)
        return result

    return gen


@scream
def play(ctl, gen):
    dsp = ossaudiodev.open('w')
    dsp.setparameters(ossaudiodev.AFMT_S16_LE, 1, 44100)
    cnt = 0
    start = time.time()
    while True:
        now = time.time()
        need_samples = (now - start) * FREQ + BUFSIZE
        if cnt <= need_samples:
            frame = next(gen, None) * ctl['master-volume']
            if frame is None:
                break
            if np.max(np.abs(frame)) >= 1:
                print('Distortion!!!')
            dsp.write((frame * 32767).astype(np.int16))
            cnt += len(frame)
        else:
            time.sleep((cnt - need_samples)/FREQ)
    dsp.sync()
