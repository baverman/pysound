import sys
import argparse
import os
import time
import threading
import wave
import random
import pprint
import json
import math
import collections
import functools

from glob import glob
from ctypes import byref, memmove, c_void_p

import sdl2
import numpy as np

import cfilters
from cfilters import addr


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


class Trigger:
    def __init__(self, value=True):
        self._value = value

    def __bool__(self):
        return self._value

    def set(self, value):
        self._value = value


@functools.lru_cache(50)
def adsr_tail(a, b, cnt):
    return (np.linspace(a, b, cnt, endpoint=False, dtype=np.float32),
            np.full(BUFSIZE, b, dtype=np.float32))


def env_adsr(trig, attack, decay, sustain, release, last=None):
    last = last or 0
    acnt = sps(attack / 1000)
    dcnt = sps(decay / 1000)
    rcnt = sps(release / 1000)

    ae = np.concatenate([
        np.linspace(last, 1, acnt, endpoint=False, dtype=np.float32),
        *adsr_tail(1, sustain, dcnt)
    ])

    re = np.concatenate(adsr_tail(sustain, 0, rcnt))

    samples = 0
    state = 1
    def gen():
        nonlocal samples, state
        if state == 1:
            if samples < acnt + dcnt:
                result = ae[samples:samples+BUFSIZE]
            else:
                result = ae[-BUFSIZE:]
                if not trig:
                    state = 2
                    samples = -BUFSIZE
        elif state == 2:
            if samples < rcnt:
                result = re[samples:samples+BUFSIZE]
            else:
                result = re[-BUFSIZE:]
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

    def __len__(self):
        return len(self.gens)

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
    def __init__(self, ctl, synth, flt=None):
        self.params = {'freq': 1}
        self.synth = synth
        self.env = None
        self.gen = None
        self.ctl = ctl
        self.flt = flt

    def add(self, key, params):
        self.params.update(params)
        ctl = self.ctl
        last = self.env.last if self.env else 0.0
        self.env = env_ahr(ctl['attack'], ctl['hold'], ctl['release'], last)

    def remove(self, key):
        pass

    def __call__(self, result=None):
        if result is None:
            result = np.full(BUFSIZE, 0, dtype=np.float32)

        data = next(self.gen, None)
        if data is not None:
            if self.flt:
                data = self.flt(self.ctl, self.params, data)
            result += data * (self.env() ** 4) * self.params['volume'] * self.ctl.get('volume', 1.0)

        return result


class poly_adsr:
    def __init__(self, ctl, synth, flt=None):
        self.ctl = ctl
        self.synth = synth
        self.flt = flt
        self.gens = {}

    def add(self, key, params):
        ctl = self.ctl
        t = Trigger()
        env = env_adsr(t, ctl['attack'], ctl.get('decay', ctl['attack']), ctl['sustain'], ctl['release'])
        self.gens[key] = params, self.synth(self.ctl, params), env, t

    def remove(self, key):
        if key in self.gens:
            v = self.gens.pop(key)
            v[-1].set(False)
            self.gens[v[1]] = v

    def __len__(self):
        return len(self.gens)

    def __call__(self, result=None):
        toremove = []
        if result is None:
            result = np.full(BUFSIZE, 0, dtype=np.float32)

        for key, (p, g, e, _t) in self.gens.items():
            data = next(g, None)
            if data is not None:
                if self.flt:
                    data = self.flt(self.ctl, p, data)
                result += data * (e() ** 4) * p['volume'] * self.ctl.get('volume', 1.0)

            if data is None or not e.running:
                toremove.append(key)

        for it in toremove:
            self.gens.pop(it)

        return result


class Player:
    def __init__(self):
        self.channels = {}

    def set_voice(self, channel, env_synth):
        self.channels[channel] = env_synth

    def note_on(self, channel, midi_note, volume):
        self.note_off(channel, midi_note)
        es = self.channels[channel]
        es.add(midi_note, {'freq': mtof(midi_note), 'volume': volume})

    def note_off(self, channel, midi_note):
        try:
            self.channels[channel].remove(midi_note)
        except KeyError:
            pass

    def __len__(self):
        return len(self.channels)

    def __call__(self, result=None):
        if result is None:
            result = np.full(BUFSIZE, 0, dtype=np.float32)
        for es in self.channels.values():
            result = es(result)
        return result


def midi_player(ctl, player):
    notes = ctl['midi_notes']
    while notes:
        etype, (ch, note, velocity) = notes.popleft()
        print(time.time(), etype, ch, note)
        if etype == 0:
            player.note_off(ch, note)
        elif etype == 1:
            player.note_on(ch, note, velocity / 127)


def kbd_player(channel=0, octave=4, volume=1.0):
    import keys
    state = set()

    def step(ctl, player):
        nonlocal state
        old_state = set(state)
        pressed = set(it for it in ctl.get('keys', []) if it in keys.KEYNOTES)
        state = pressed

        for key in old_state - pressed:
            player.note_off(channel, 12*octave + keys.KEYNOTES[key])

        for key in pressed - old_state:
            player.note_on(channel, 12*octave + keys.KEYNOTES[key], volume)

    return step


def fft_plot(signal, window=2048, crop=0):
    return np.fft.rfftfreq(window, 1/FREQ)[crop:], 2/window*np.abs(np.fft.rfft(signal, window))[crop:]


def open_wav(fname, fmt=3):
    wave.WAVE_FORMAT_PCM = fmt  # float
    f = wave.open(fname, 'wb')
    f.setnchannels(1)
    f.setsampwidth(4 if fmt==3 else 2)
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
        self.update_dist = None
        self.dist_counter = 0
        self.args = None

    def __iter__(self):
        return iter(self.controls)

    def command(self, control):
        def callback(value):
            self.ctl[control.name] = float(value)
        return callback

    def dist_cb(self):
        self.dist_counter += 1
        if self.update_dist:
            self.update_dist(self.dist_counter)

    def play(self, gen, width=600, output=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--preset')
        parser.add_argument('-o', '--output')
        self.args, _ = parser.parse_known_args()

        master = create_window(self, width)
        output = output or self.args.output
        if output:
            wavfile = open_wav(output, 1)
        else:
            wavfile = None

        stop = threading.Event()
        t = threading.Thread(target=scream(play),
                             args=(self.ctl, gen, self.dist_cb),
                             kwargs={'stop': stop, 'wavfile': wavfile},
                             daemon=True)
        t.start()

        master.tk.mainloop()
        master.pysound_exit()
        stop.set()
        t.join()

        if wavfile:
            wavfile.close()


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


def create_window(controls, width):
    import tkinter as tk
    from tkinter import ttk
    import keys


    internal_ctl_keys = 'keys', 'midi_notes'
    controls.ctl['keys'] = []
    controls.ctl['midi_notes'] = collections.deque()
    def kb(_):
        controls.ctl['keys'] = keys.keyboard_state()
        # print(controls.ctl['keys'])

    master = tk.Tk()
    master.bind('<KeyPress>', kb)
    master.bind('<KeyRelease>', kb)

    def save_preset(name=None):
        explicit_name = name is not None
        name = name or preset_cb.get().strip()
        if not name:
            return

        state = controls.ctl.copy()
        for k in internal_ctl_keys:
            state.pop(k, None)

        with open(controls.preset_prefix + name + '.state.json', 'w') as fd:
            fd.write(json.dumps(state))

        if not explicit_name:
            values = sorted(set(list(preset_cb['values']) + [name]))
            preset_cb['values'] = values

    def load_preset(_e, name=None):
        name = name or preset_cb.get().strip()
        if not name:
            return

        with open(controls.preset_prefix + name + '.state.json') as fd:
            data = json.load(fd)
            update_state(ctrl_map, controls.ctl, data)

    save_frame = ttk.Frame(master)

    preset_cb = ttk.Combobox(save_frame, values=sorted(get_presets(controls.preset_prefix)))
    preset_cb.bind('<<ComboboxSelected>>', load_preset)
    preset_cb.pack(side='right')

    ttk.Label(save_frame, text=" ").pack(side='right')

    save_button = ttk.Button(save_frame, text="Save", command=save_preset)
    save_button.pack(side='right')

    dist_w = ttk.Label(save_frame, text="d.frames: 0")
    dist_w.pack(side='left')
    def update_dist(value):
        dist_w['text'] = f'd.frames: {value}'
    controls.update_dist = update_dist

    save_frame.pack(fill='x')

    canvas = tk.Canvas(master)
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
            if w.midi_wait_value is not None:
                return f'{w.orig_label} ch:{w.midi_channel} ctrl:{w.midi_ctrl} val:{round(w.midi_wait_value, w.round)}'
            else:
                return f'{w.orig_label} ch:{w.midi_channel} ctrl:{w.midi_ctrl}'
        else:
            return w.orig_label

    def midi_cb(etype, value):
        nonlocal midi_ctrl
        print('@@', time.time(), etype, value)
        if etype in (0, 1):
            controls.ctl['midi_notes'].append((etype, value))
        elif etype == 2:
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
                if abs(value - w.get()) / (w['to'] - w['from']) > 0.03:
                    w.midi_wait_value = value
                else:
                    w.midi_wait_value = None
                    w.pysound_set_cb(value)
                    w.set(value)
                w['label'] = get_label(w)

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
                w.midi_wait_value = None
                w.round = int(math.log10(int(1/v.resolution)))
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

    master.update()
    canvas['width'] = scrollable_frame.winfo_reqwidth()
    canvas['height'] = min(900, scrollable_frame.winfo_reqheight())

    if controls.args.preset:
        preset_cb.set(controls.args.preset)
        load_preset(None)
    else:
        if 'tmp' in preset_cb['values']:
            load_preset(None, 'tmp')

    source = os.environ.get('MIDI_SOURCE')
    if source:
        import asound
        t = threading.Thread(target=asound.listen, args=(source, midi_cb), daemon=True)
        t.start()

    def exit():
        save_preset('tmp')

    master.pysound_exit = exit
    return master


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
        alpha = ensure_buf(cutoff)
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


def play(ctl, gen, dist_cb=None, wavfile=None, stop=None):
    cnt = 0
    max_p = 0
    def handle_sound(_userdata, stream, length):
        nonlocal cnt, max_p
        s = time.perf_counter()
        frame = next(gen)

        frame *= ctl['master-volume']
        if np.max(np.abs(frame)) >= 1:
            if dist_cb:
                dist_cb()
            else:
                print('Distortion!!!')

        frame[frame > 0.99] = 0.99
        frame[frame < -0.99] = -0.99
        frame = (frame * 32767).astype(np.int16)

        memmove(stream, frame.ctypes.data, length)

        if wavfile:
            wavfile.writeframesraw(frame)

        dur = time.perf_counter() - s
        if dur > max_p:
            print('@@ max process time', dur)
            max_p = dur

    sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
    spec = sdl2.SDL_AudioSpec(FREQ, sdl2.AUDIO_S16LSB, 1, BUFSIZE, sdl2.SDL_AudioCallback(handle_sound))
    target = sdl2.SDL_AudioSpec(0, 0, 0, 0)
    dev = sdl2.SDL_OpenAudioDevice(None, 0, byref(spec), byref(target), 0)
    assert target.freq == FREQ, target.freq
    assert target.size == BUFSIZE*2, target.size

    sdl2.SDL_PauseAudioDevice(dev, 0);

    if stop:
        stop.wait()
    else:
        while True:
            time.sleep(1000)

    sdl2.SDL_CloseAudioDevice(dev)


def render_to_file(fname, ctl, gen, duration):
    with open_wav(fname) as f:
        for _ in range(fps(duration)):
            frame = next(gen, None) * ctl['master-volume']
            if frame is None:
                break
            f.writeframes(frame)
