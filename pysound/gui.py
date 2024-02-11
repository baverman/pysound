import argparse
import os
import time
import threading
import json
import math
import collections

from glob import glob
from functools import partial

import tkinter as tk
from tkinter import ttk

from .core import scream, open_wav, play


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


class Scale:
    def __init__(self, parent, *args, **kwargs):
        self.root = tk.Frame(parent)
        self.label = ttk.Label(self.root, text=kwargs.pop('label', None))
        anchor = 'w' if kwargs.get('orient') == tk.HORIZONTAL else None
        self.label.pack(anchor=anchor)
        self.scale = tk.Scale(self.root, *args, **kwargs, showvalue=False)
        self.scale.pack(fill=tk.Y, expand=1)
        self.get = self.scale.get
        self.set = self.scale.set

    def bind(self, *args):
        return self.scale.bind(*args)

    def __setitem__(self, key, value):
        if key == 'label':
            self.label['text'] = value
        else:
            self.scale[key] = value


def idfn(value):
    return value


class Var:
    def __init__(self, name, value, min, max, resolution=None, label=None,
                 hidden=False, midi_ctrl=None, midi_channel=None, orient=tk.HORIZONTAL, func=None):

        self.minr = min
        self.maxr = max
        if func == 'exp':
            min, max = 0.0, 1.0
            self.transform = lambda value: self.minr + value ** 2.0 * (self.maxr - self.minr)
            self.rtransform = lambda value: ((value - self.minr) / (self.maxr - self.minr)) ** 0.5
        else:
            self.transform = idfn
            self.rtransform = idfn

        if not resolution:
            resolution = (max-min)/1000;
        self.name = name
        self.label = label or name.replace('-', ' ').title()
        self.val = value
        self.min = min
        self.max = max
        self.resolution = resolution
        self.hidden = hidden
        self.midi_ctrl = midi_ctrl
        self.midi_channel = midi_channel
        self.orient = orient

    def widget(self, parent, ctl, config):
        def cb(val):
            ctl[self.name] = self.transform(float(val))

        def reset(event):
            ctl[self.name] = self.transform(self.val)
            w.set(self.val)

        def incrset(event):
            val = self.rtransform(ctl[self.name])
            delta = abs(self.min - self.max) / 500;
            if event.num == 5:
                delta = -delta
            val += delta
            event.widget.set(val)
            rmax, rmin = self.max, self.min
            if rmin > rmax:
                rmax, rmin = rmin, rmax
            ctl[self.name] = self.transform(min(rmax, max(float(val), rmin)))

        w = Scale(parent, from_=self.min, to=self.max, orient=self.orient,
                  resolution=self.resolution, command=cb, label=self.label, length=200)
        w.pysound_set_cb = cb
        w.pysound_init = lambda val: w.set(self.rtransform(val))
        w.set(self.val)
        w.bind('<ButtonPress-4>', incrset)
        w.bind('<ButtonPress-5>', incrset)
        w.bind('<Control-ButtonRelease-1>', reset)
        config(self, w)
        return w.root


class VSlide(Var):
    def __init__(self, name, value, min, max, resolution=None, **kwargs):
        super().__init__(name, value, max, min, resolution, orient=tk.VERTICAL, **kwargs)


class Radio:
    def __init__(self, name, value, choice, label=None):
        self.name = name
        self.label = label or name.replace('-', ' ').title()
        self.val = value
        self.choice = choice
        self.hidden = False

    def widget(self, parent, ctl, config):
        frm = ttk.LabelFrame(parent, text=self.label, padding=5)
        tkvar = tk.IntVar()

        def cb():
            ctl[self.name] = tkvar.get()

        for i, label in enumerate(self.choice):
            ttk.Radiobutton(frm, text=label, variable=tkvar, value=i, command=cb).pack(anchor='w')

        tkvar.set(self.val)
        config(self, tkvar)
        return frm


class HStack:
    def __init__(self, *children, label=None, height=None, padding=10):
        self.children = children
        self.label = label
        self.height = height
        self.padding = padding

    def widget(self, parent, ctl, config):
        if self.label:
            w = ttk.LabelFrame(parent, text=self.label, height=self.height, padding=self.padding)
        else:
            w = tk.Frame(parent, height=self.height, padding=self.padding)

        for it in self.children:
            it.widget(w, ctl, config).pack(side='left', padx=self.padding//2, fill=tk.Y, expand=1)

        return w

    def __iter__(self):
        return iter(self.children)

    def items(self):
        for it in self.children:
            if hasattr(it, 'items'):
                yield from it.items()
            else:
                yield it


class VarGroup:
    def __init__(self, name, controls, label=None, midi_channel=None):
        self.name = name
        self.controls = controls
        self.label = label or name.replace('-', ' ').title()

        self.ctl = {}
        for it in controls:
            if hasattr(it, 'items'):
                self.ctl.update({it.name: it.val for it in it.items()})
            else:
                self.ctl[it.name] = it.val

        self.val = self.ctl
        self.midi_channel = midi_channel

    def __iter__(self):
        return iter(self.controls)


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
            if hasattr(cmap[k], 'pysound_init'):
                cmap[k].pysound_init(v)
            else:
                cmap[k].set(v)


def create_window(controls, width):
    from . import keys

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

    def setup_midi(cmap, v, w):
        cmap[v.name] = w
        w.orig_label = v.label
        if hasattr(v, 'midi_channel'):
            w.midi_channel = v.midi_channel or controls.midi_channel
            w.midi_ctrl = v.midi_ctrl
            w.midi_wait_value = None
            w.round = int(math.log10(1/v.resolution))
            if w.midi_ctrl is not None:
                midi_ctrl_map[(w.midi_channel, w.midi_ctrl)] = w
            w['label'] = get_label(w)
            w.bind('<ButtonPress-3>', set_midi_ctrl)
            w.bind('<ButtonRelease-3>', release_midi_ctrl)

    def pack_controls(parent, controls, cmap):
        config = partial(setup_midi, cmap)
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
                v.widget(parent, controls.ctl, config).pack(anchor='w', pady=5)

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
        from . import asound
        t = threading.Thread(target=asound.listen, args=(source, midi_cb), daemon=True)
        t.start()

    def exit():
        save_preset('tmp')

    master.pysound_exit = exit
    return master
