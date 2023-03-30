import os
import time
import numpy as np
import ossaudiodev
import threading
import wave
import random
import pprint
import json

import cfilters
from cfilters import addr
from glob import glob

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
        result = np.cumsum(ensure_buf(freq) / self.freq)
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

    def reset(self, phase=0.0):
        self.phasor.phase = phase


def ensure_buf(value, dtype=np.float32):
    if type(value) in (int, float):
        return np.full(BUFSIZE, value, dtype=dtype)
    return value


def square():
    o = phasor()
    def sig(f, duty):
        return (o(f) > ensure_buf(duty)).astype(np.float32) * 2.0 - 1.0
    return sig


def noise():
    return np.random.rand(BUFSIZE).astype(np.float32) * 2.0 - 1.0


def seed_noise(seed=None):
    state = np.random.RandomState(seed)
    def process():
        return state.rand(BUFSIZE).astype(np.float32) * 2.0 - 1.0
    return process

def seed_noise2(seed=None):
    state = np.random.RandomState(seed)
    def process():
        return (state.rand(BUFSIZE).astype(np.float32) > 0.5) * 2.0 - 1.0
    return process


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


class mono:
    def __init__(self):
        self.gen = None
        self.last = 0.0

    def set(self, gen):
        self.gen = gen

    def __call__(self):
        sig = None
        if self.gen:
            data = next(self.gen, None)
            if data is None:
                self.gen = None
            else:
                sig, self.last = data

        if sig is None:
            sig = np.full(BUFSIZE, 0, dtype=np.float32)

        return sig


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
    def __init__(self, *controls, preset_prefix=''):
        self.controls = controls
        self.ctl = {it.name: it.val for it in controls}
        self.midi_channel = None
        self.preset_prefix = preset_prefix

    def __iter__(self):
        return iter(self.controls)

    def command(self, control):
        def callback(value):
            self.ctl[control.name] = float(value)
        return callback

    def play(self, gen, width=600):
        t = threading.Thread(target=scream(play), args=(self.ctl, gen), daemon=True)
        t.start()
        show_window(self, width)


class Var:
    def __init__(self, name, value, min, max, resolution=1, label=None,
                 hidden=False, midi_ctrl=None, midi_channel=None):
        self.name = name
        self.label = label or name.replace('-', ' ').title()
        self.val = value
        self.min = min
        self.max = max
        self.resolution = resolution
        self.hidden = hidden
        self.midi_ctrl = midi_ctrl
        self.midi_channel = midi_channel


class VarGroup:
    def __init__(self, name, controls, label=None, midi_channel=None):
        self.name = name
        self.controls = controls
        self.label = label or name.replace('-', ' ').title()
        self.ctl = {it.name: it.val for it in controls}
        self.val = self.ctl
        self.midi_channel = midi_channel

    def __iter__(self):
        return iter(self.controls)

    def command(self, control):
        def callback(value):
            self.ctl[control.name] = float(value)
        return callback


def get_presets(prefix):
    suffix = '.state.json'
    return [it[len(prefix):-len(suffix)] for it in glob(f'{prefix}*{suffix}')]


def update_state(cmap, dest, src):
    for k, v in src.items():
        if k not in dest:
            continue
        if type(v) is dict:
            update_state(cmap[k], dest[k], v)
        else:
            dest[k] = v
            cmap[k].set(v)


def show_window(controls, width):
    import tkinter as tk
    from tkinter import ttk
    import keys

    controls.ctl['keys'] = []
    def kb(_):
        controls.ctl['keys'] = keys.keyboard_state()
        print(controls.ctl['keys'])

    master = tk.Tk()
    master.bind('<KeyPress>', kb)
    master.bind('<KeyRelease>', kb)

    def save_preset():
        name = preset_cb.get().strip()
        if not name:
            return

        state = controls.ctl.copy()
        state.pop('keys', None)
        with open(controls.preset_prefix + name + '.state.json', 'w') as fd:
            fd.write(json.dumps(state))

        values = sorted(set(list(preset_cb['values']) + [name]))
        preset_cb['values'] = values

    def load_preset(_e):
        name = preset_cb.get().strip()
        if not name:
            return

        with open(controls.preset_prefix + name + '.state.json') as fd:
            data = json.load(fd)
            update_state(ctrl_map, controls.ctl, data)

    save_frame = ttk.Frame(master)
    preset_cb = ttk.Combobox(save_frame, values=sorted(get_presets(controls.preset_prefix)))
    preset_cb.bind('<<ComboboxSelected>>', load_preset)
    preset_cb.pack(side='left')
    ttk.Label(save_frame, text=" ").pack(side='left')
    save_button = ttk.Button(save_frame, text="Save", command=save_preset)
    save_button.pack(side='left')
    save_frame.pack(anchor='e')

    canvas = tk.Canvas(master, width=width+4)
    scrollbar = ttk.Scrollbar(master, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    notebook = None
    midi_ctrl_map = {}
    midi_ctrl = None
    ctrl_map = {}

    def get_label(w):
        if w.midi_ctrl is not None:
            return f'{w.orig_label} ch:{w.midi_channel} ctrl:{w.midi_ctrl}'
        else:
            return w.orig_label

    def midi_cb(etype, value):
        nonlocal midi_ctrl
        if etype == 2:
            ch, param, v = value
            key = (ch, param)
            if midi_ctrl is not None:
                midi_ctrl.midi_channel = ch
                midi_ctrl.midi_ctrl = param
                midi_ctrl_map[key] = midi_ctrl
                midi_ctrl['label'] = get_label(midi_ctrl)
            elif key in midi_ctrl_map:
                w = midi_ctrl_map[(ch, param)]
                value = w['from'] + (w['to'] - w['from']) * v / 127
                w.pysound_set_cb(value)
                w.set(value)

    def set_midi_ctrl(e):
        nonlocal midi_ctrl
        midi_ctrl = e.widget
        e.widget['label'] = e.widget.orig_label + ' (waiting midi...)'

    def release_midi_ctrl(e):
        nonlocal midi_ctrl
        midi_ctrl = None
        e.widget['label'] = get_label(e.widget)

    def pack_controls(parent, controls, cmap):
        nonlocal notebook
        for v in controls:
            if isinstance(v, VarGroup):
                if notebook is None:
                    notebook = ttk.Notebook(parent)
                    notebook.pack()
                nf = ttk.Frame(notebook)
                notebook.add(nf, text=v.label)
                pack_controls(nf, v, cmap.setdefault(v.name, {}))
            else:
                set_cb = controls.command(v)
                w = tk.Scale(parent, label=v.label, from_=v.min, to=v.max, orient=tk.HORIZONTAL,
                             length=width, command=set_cb, resolution=v.resolution)
                cmap[v.name] = w
                w.orig_label = v.label
                w.midi_channel = v.midi_channel or controls.midi_channel
                w.midi_ctrl = v.midi_ctrl
                if w.midi_ctrl is not None:
                    midi_ctrl_map[(w.midi_channel, w.midi_ctrl)] = w
                w['label'] = get_label(w)
                w.pysound_set_cb = set_cb
                w.bind('<ButtonPress-3>', set_midi_ctrl)
                w.bind('<ButtonRelease-3>', release_midi_ctrl)
                w.set(v.val)
                w.pack()

    pack_controls(scrollable_frame, controls, ctrl_map)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")


    source = os.environ.get('MIDI_SOURCE')
    if source:
        import asound
        t = threading.Thread(target=asound.listen, args=(source, midi_cb), daemon=True)
        t.start()

    tk.mainloop()
    # pprint.pprint(controls.ctl)


def fps(duration=1):
    return int(duration * FREQ / BUFSIZE)


def sps(duration=1):
    return int(duration * FREQ)


def lowpass():
    result = np.empty(BUFSIZE, dtype=np.float32)
    state = np.zeros(2, dtype=np.float32)
    ra = addr(result)
    sa = addr(state)

    def sig(data, cutoff, resonance=0):
        alpha = ensure_buf(cutoff) / FREQ
        cfilters.lowpass(ra, addr(data), len(data), addr(alpha), resonance, sa)
        return result

    return sig


def dcfilter(r=0.98):
    result = np.empty(BUFSIZE, dtype=np.float32)
    state = np.zeros(2, dtype=np.float32)
    ra = addr(result)
    sa = addr(state)

    def sig(data):
        # print(state, sa[0], sa[1])
        cfilters.dcfilter(ra, addr(data), len(data), sa, r)
        return result

    return sig


def delay(max_duration=0.5):
    buf = np.full(sps(max_duration), 0, dtype=np.float32)
    ba = addr(buf)
    def process(sig, delay, feedback):
        size = len(sig)
        shift = sps(delay)
        buf[:-size] = buf[size:]
        cfilters.delay_process(ba, len(buf), addr(sig), len(sig), shift, feedback)
        return buf[-size:].copy()
    return process


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
            dsp.write((np.clip(frame, -0.99, 0.99) * 32767).astype(np.int16))
            cnt += len(frame)
        else:
            time.sleep((cnt + BUFSIZE/2 - need_samples)/FREQ)
    dsp.sync()


def render_to_file(fname, ctl, gen, duration):
    with open_wav_f32(fname) as f:
        for _ in range(fps(duration)):
            frame = next(gen, None) * ctl['master-volume']
            if frame is None:
                break
            f.writeframes(frame)
