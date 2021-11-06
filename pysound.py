import time
import numpy as np
import ossaudiodev
import threading
import wave

import filters

FREQ = 44100
BUFSIZE = 512

tau = np.pi * 2


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


class Var:
    def __init__(self, label, value, min, max, resolution=1):
        self.label = label
        self.val = value
        self.min = min
        self.max = max
        self.resolution = resolution

    def __call__(self, value):
        self.val = float(value)


def gui(ctl, width):
    import tkinter as tk
    import keys

    def kb(_):
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


def fps(duration=1):
    return int(duration * FREQ / BUFSIZE)


def sps(duration=1):
    return int(duration * FREQ)


def lowpass():
    result = np.empty(BUFSIZE, dtype=np.float32)
    result[-1] = 0

    def gen(data, cutoff):
        if type(cutoff) in (int, float):
            alpha = np.full(data.shape[0], cutoff/FREQ, dtype=np.float32)
        else:
            alpha = cutoff / FREQ

        filters.lowpass(result, data, alpha, result[-1])
        return result

    return gen


@scream
def play(ctl, gen):
    dsp = ossaudiodev.open('w')
    dsp.setparameters(ossaudiodev.AFMT_S16_LE, 1, 44100)
    cnt = 0
    start = time.time()
    while True:
        if cnt < (time.time() - start) * FREQ + BUFSIZE:
            frame = next(gen, None) * ctl['master'].val
            if frame is None:
                break
            if np.max(np.abs(frame)) >= 1:
                print('Distortion!!!')
            dsp.write((frame * 32767).astype(np.int16))
            cnt += len(frame)
        else:
            time.sleep(BUFSIZE/FREQ/2)
    dsp.sync()


def gui_play(ctl, gen, width=600):
    t = threading.Thread(target=play, args=(ctl, gen), daemon=True)
    t.start()
    gui(ctl, width)
