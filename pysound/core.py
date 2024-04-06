import time
import wave
import random
import math

import numpy as np

from . import cfilters
from .cfilters import addr


# FREQ = 44100
FREQ = 48000
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


def sinsum(steps, partials):
    x = np.linspace(0, 1, steps, endpoint=False, dtype=np.float32)
    y = 0
    for p, v in enumerate(partials, 1):
        y = y + v * np.sin(x*tau*p)
    y /= np.max(y)
    return x, y

sin_partials = [1]
saw_partials = lambda n: [1/i for i in range(1, n+1)]
tri_partials = lambda n: [((-1)**((i-1)/2)) * 1/i**2 if i%2 else 0 for i in range(1, n+1)]
square_partials = lambda n: [1/i if i%2 else 0 for i in range(1, n+1)]

sin_t = sinsum(2048, sin_partials)


def phasor_apply(sig, table):
    return np.interp(sig, *table).astype(np.float32)


class phasor:
    def __init__(self, phase=0, srate=FREQ):
        self.phase = 0
        self.srate = srate
        self.fdelta = ensure_buf(0)
        self.sig = ensure_buf(0)

    def __call__(self, freq):
        f = self.fdelta
        if type(freq) in (int, float, np.float64):
            f[:] = freq / self.srate
        else:
            np.divide(freq, self.srate, out=f)

        self.phase = cfilters.lib.phasor(addr(self.sig), addr(f), len(f), self.phase)
        return self.sig


def shold(value=0, prev=0):
    def process(sig, phase):
        nonlocal value, prev
        result = cfilters.shold(sig, phase, value, prev)
        value = result[-1]
        prev = phase[-1]
        return result
    return process


class osc:
    def __init__(self, table, phase=0, freq=FREQ, p=None):
        self.phasor = p or phasor(phase, freq)
        self.table = table

    def __call__(self, freq):
        return np.interp(self.phasor(freq), *self.table).astype(np.float32)

    def reset(self, phase=0.0):
        self.phasor.phase = phase


def ensure_buf(value, dtype=np.float32, size=None, out=None):
    if type(value) in (int, float, np.float64):
        if out is not None:
            out[:] = value
            return out
        return np.full(size or BUFSIZE, value, dtype=dtype)
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


def line(last=0):
    e = np.full(BUFSIZE, 0, dtype=np.float32)
    samples = 0
    cnt = 0

    def pgen(value, duration):
        nonlocal last, e, samples, cnt
        if value != last:
            samples = 0
            cnt = sps(duration / 1000)
            e = np.concatenate([
                np.linspace(last, value, cnt, endpoint=False, dtype=np.float32),
                np.full(BUFSIZE, value, dtype=np.float32)])

        if samples < cnt:
            result = e[samples:samples+BUFSIZE]
        else:
            result = e[-BUFSIZE:]

        samples += BUFSIZE
        last = result[-1]
        return result

    return pgen


class mono:
    def __init__(self, ctl, synth, retrigger=True):
        self.ctl = ctl
        self.params = {'freq': 1, 'volume': 0.0}
        self.gen = synth(ctl, self.params)
        self.params['env'].stop(False)
        self.key = None
        self.vol_line = line()
        self.freq_stack = []
        self.retrigger = retrigger

    def add(self, key, params):
        # print('add', key, params, self.freq_stack)
        if self.key is not None:
            self.freq_stack.append((self.key, self.params['freq']))
        self.key = key
        self.params.update(params)
        if not self.freq_stack or self.retrigger:
            self.params['env'].trigger()

    def remove(self, key):
        # print('remove', key, self.freq_stack)
        # if self.key == key:
        #     self.params['env'].stop()

        if self.freq_stack:
            self.key, f = self.freq_stack.pop()
            self.params['freq'] = f
            if self.retrigger:
                self.params['env'].trigger()
        else:
            self.key = None
            self.params['env'].stop()

    def __call__(self, result=None):
        if result is None:
            result = np.full(BUFSIZE, 0, dtype=np.float32)

        data = self.gen()
        result += data * self.vol_line(self.params['volume'], 10) * self.ctl.get('volume', 1.0)
        return result


class menv:
    def __init__(self, *envs):
        self._envs = envs

    def stop(self, wait_decay=None):
        for e in self._envs:
            e.stop(wait_decay)

    @property
    def active(self):
        return any(it.active for it in self._envs)


class poly:
    def __init__(self, ctl, synth):
        self.ctl = ctl
        self.synth = synth
        self.gens = {}

    def add(self, key, params):
        ctl = self.ctl
        gen = self.synth(self.ctl, params)
        self.gens[key] = params, gen

    def remove(self, key):
        if key in self.gens:
            v = self.gens.pop(key)
            v[0]['env'].stop()
            self.gens[v[1]] = v

    def __len__(self):
        return len(self.gens)

    def __call__(self, result=None):
        toremove = []
        if result is None:
            result = np.full(BUFSIZE, 0, dtype=np.float32)

        for key, (p, g) in self.gens.items():
            if p['env'].active:
                data = g()
                result += data * p['volume'] * self.ctl.get('volume', 1.0)
            else:
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
        # self.note_off(channel, midi_note)
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
    from . import keys
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


def fps(duration=1):
    return int(duration * FREQ / BUFSIZE)


def sps(duration=1):
    return int(duration * FREQ)


def poly_saw():
    result = ensure_buf(0)
    def sgen(phasor, *args):
        cfilters.lib.poly_saw(addr(result), addr(phasor.sig), addr(phasor.fdelta), len(result))
        return result
    return sgen


def poly_square():
    result = ensure_buf(0)
    pwbuf = ensure_buf(0)
    def sgen(phasor, pw=0.0):
        pw = ensure_buf(np.clip(pw, 0.0, 0.45), out=pwbuf)
        cfilters.lib.poly_square(addr(result), addr(phasor.sig), addr(phasor.fdelta), addr(pw), len(result))
        return result
    return sgen


def make_filter(fn, scount):
    def filter_constructor():
        result = np.empty(BUFSIZE, dtype=np.float32)
        state = np.zeros(scount + 1, dtype=np.float32)
        state[0] = FREQ
        ra = addr(result)
        sa = addr(state)

        def sig(data, cutoff, resonance=0):
            alpha = ensure_buf(cutoff)
            fn(ra, addr(data), len(data), addr(alpha), resonance, sa)
            return result

        return sig
    return filter_constructor


bqlp = make_filter(cfilters.lib.bqlp, 4)
flt12 = make_filter(cfilters.lib.flt12, 2)
pdvcf = make_filter(cfilters.lib.pdvcf, 2)
moog = make_filter(cfilters.lib.moog, 7)
lowpass_orig = make_filter(cfilters.lib.lowpass, 2)
pole2 = make_filter(cfilters.lib.pole2, 2)

lowpass = lowpass_orig


def dcfilter(cutoff=20):
    r = 1 - (math.pi*2 * cutoff / FREQ)
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
    buf = cfilters.init_ring_buf(sps(max_duration))
    result = np.empty(BUFSIZE, dtype=np.float32)
    def process(sig, delay, feedback):
        shift = ensure_buf(delay, np.int32)
        cfilters.delmix(buf, result, sig, shift, feedback)
        return result
    return process


class Buffer:
    def __init__(self, duration, dtype=np.float32):
        self.length = sps(duration)
        self.buf = ensure_buf(0, dtype=dtype, size=self.length*2)
        self.pos = 0

    def write(self, data):
        dlen = len(data)
        assert dlen <= self.length
        p2 = self.pos + self.length
        self.buf[self.pos:self.pos+dlen] = data
        if p2+dlen >= self.length*2:
            rest = (p2 + dlen + 1) % (self.length*2)
            self.buf[p2:p2+dlen-rest] = data[:-rest]
            self.buf[:rest] = data[-rest:]
        else:
            self.buf[p2:p2+dlen] = data
        self.pos = (self.pos + dlen) % self.length

    def get(self, duration=None):
        if duration:
            size = sps(duration)
            assert size <= self.length
        else:
            size = self.length

        p = (self.pos - size) % self.length
        return self.buf[p:p+size]


def env_adsr_lin(last=0.0, stopped=False, wait_decay=False):
    gwait_decay = wait_decay
    state = cfilters.env_adsr_lin_init(FREQ, last)
    if stopped:
        state.state = 3
    zero = np.zeros(BUFSIZE, dtype=np.float32)
    result = np.zeros(BUFSIZE, dtype=np.float32)

    def sgen(attack, decay, sustain, release, hold=0.0):
        if state.state == 3:
            sgen.active = False
            return zero
        cfilters.lib.env_adsr(addr(result), BUFSIZE, state, attack, hold, decay, sustain, release)
        sgen.last = state.last
        return result

    sgen.active = not stopped

    def stop(wait_decay=None):
        if wait_decay is None:
            wait_decay = gwait_decay

        if state.state == 0:
            if wait_decay:
                state.state = 1
            else:
                state.state = 2
                state.scount = 0
                state.release_level = state.last

    def trigger():
        sgen.active = True
        state.state = 0
        state.scount = 0

    sgen.stop = stop
    sgen.trigger = trigger
    return sgen


def env_adsr(last=0.0, stopped=False, wait_decay=False, rise_th=0.05, fall_th=0.01, lspeed=10):
    gwait_decay = wait_decay
    state = cfilters.env_adsr_exp_init(FREQ, last)
    state.rise_th = rise_th
    state.fall_th = fall_th
    state.lspeed = lspeed
    if stopped:
        state.state = 0
    else:
        state.state = 1

    zero = np.zeros(BUFSIZE, dtype=np.float32)
    result = np.zeros(BUFSIZE, dtype=np.float32)
    next_state_mut = None

    def sgen(attack, decay, sustain, release, hold=0.0):
        nonlocal next_state_mut

        if state.state == 0:
            sgen.active = False
            return zero

        if next_state_mut:
            next_state_mut()
            next_state_mut = None

        cfilters.lib.env_adsr_exp(addr(result), BUFSIZE, state, attack, hold, decay, sustain, release)
        sgen.last = state.last
        return result

    sgen.active = not stopped

    def state_stop_with_decay():
        state.decay_next_state = 5

    def state_stop():
        state.state = 5

    def state_start():
        state.state = 1
        state.decay_next_state = 4

    def stop(wait_decay=None):
        nonlocal next_state_mut
        if wait_decay is None:
            wait_decay = gwait_decay

        if state.state > 0:
            if wait_decay and state.state < 4:
                next_state_mut = state_stop_with_decay
            else:
                next_state_mut = state_stop

    def trigger():
        nonlocal next_state_mut
        sgen.active = True
        next_state_mut = state_start

    sgen.stop = stop
    sgen.trigger = trigger
    return sgen


def play(ctl, gen, dist_cb=None, wavfile=None, stop=None):
    import sounddevice as sd
    s = sd.OutputStream(FREQ, 0, 'default', 1, dtype='float32', latency=0.015)
    dc = dcfilter()

    s.start()
    for frame in gen:
        if stop and stop.is_set():
            break
        frame = dc(frame)
        frame *= ctl['master-volume']

        if np.max(np.abs(frame)) >= 1:
            if dist_cb:
                dist_cb()
            else:
                print('Distortion!!!')

        frame[frame > 0.99] = 0.99
        frame[frame < -0.99] = -0.99

        af = s.write_available
        if s.write(frame):
            print('Underflow!!!', af)
            dist_cb()

    s.stop()


def play_sdl(ctl, gen, dist_cb=None, wavfile=None, stop=None):
    from ctypes import byref, memmove
    import sdl2

    cnt = 0
    max_p = 0
    dc = dcfilter()
    lastbuf = Buffer(3.0, dtype=np.int16)

    def handle_sound(_userdata, stream, length):
        nonlocal cnt, max_p
        s = time.perf_counter()

        frame = next(gen, None)
        if frame is None:
            return

        frame = dc(frame)
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

        lastbuf.write(frame)
        if wavfile:
            wavfile.writeframesraw(frame)

        dur = time.perf_counter() - s
        if dur > max_p:
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

    print('@@ max process time', max_p)
    sdl2.SDL_CloseAudioDevice(dev)

    with open_wav('/tmp/debug.wav', 1) as dw:
        dw.writeframesraw(lastbuf.get())


def render_to_file(fname, ctl, gen, duration):
    with open_wav(fname) as f:
        for _ in range(fps(duration)):
            frame = next(gen, None) * ctl['master-volume']
            if frame is None:
                break
            f.writeframes(frame)
