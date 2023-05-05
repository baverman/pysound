import re
import time
import itertools

from collections import namedtuple
from functools import lru_cache
from fractions import Fraction as F

import tonator

LIT_NOTES = {n: i for i, n in enumerate('C.D.EF.G.A.B') if n != '.'}
OFFSETS = LIT_NOTES.copy()
OFFSETS.update((f'{k}#', v+1) for k, v in LIT_NOTES.items())
OFFSETS.update((f'{k}b', v-1) for k, v in LIT_NOTES.items())

NOTE_RE = re.compile(r'^([<>#b!\']*)(-|_|\d+|[A-G])(.*)')
LENGTH_RE = re.compile(r'(-|\.|:|\*|\d+)')
CMD_RE = re.compile(r'(amul|o|g|v|k|c)=(.*)')

NOTE = 2
REST = 3
TIE = 4
CMD = 5
Note = namedtuple('Note', 'offset step accent')


class Event(namedtuple('Event', 'type length value')):
    def tie(self, length):
        return self._replace(length=self.length + length)


class Bar(list):
    def unwrap(self, ctx):
        bar_length = sum(it.length for it in self if it.length)
        factor = ctx.tsig / bar_length
        for it in self:
            if it.length:
                it = it._replace(length=it.length*factor)
            yield it


class Line(list):
    def unwrap(self, ctx):
        result = []
        for bar in self:
            for e in bar.unwrap(ctx):
                if e.type == TIE:
                    result[-1] = result[-1].tie(e.length)
                else:
                    result.append(e)
        return result


def make_context(**kwargs):
    kwargs.setdefault('channel', 0)
    kwargs.setdefault('octave', 3)
    kwargs.setdefault('bpm', 120)
    kwargs.setdefault('tsig', F(4, 4))
    kwargs.setdefault('key', tonator.Scales.major)
    kwargs.setdefault('volume', 80)
    kwargs.setdefault('accmul', 1.1)
    kwargs.setdefault('gate', 100)
    return ChainMap(None, kwargs)


class ChainMap(dict):
    __getattr__ = dict.__getitem__

    def __init__(self, parent, data):
        self.parent = parent
        super().__init__(data)

    def __missing__(self, key):
        if not self.parent:
            raise KeyError(key)
        return self.parent[key]

    def child(self):
        return ChainMap(self, {})


def get_key(name):
    (name, salias), = re.findall('([A-G][#b]?)([m]?)', name)
    scale = {'': tonator.Scales.major, 'm': tonator.Scales.minor}[salias]
    return scale.to(OFFSETS[name])


@lru_cache(None)
def parse_length(l):
    k = F(1)
    pending = None
    for it in LENGTH_RE.findall(l):
        if it == '.':
            if pending:
                k *= pending
                pending = None
            k *= F(3, 2)
        elif it in '-_':
            if pending:
                k *= pending
                pending = None
            k += F(1)
        elif it == '*':
            if pending:
                k *= pending
                pending = None
            pending = F(2)
        elif it == ':':
            if pending:
                k *= pending
                pending = None
            pending = F(1, 2)
        else:
            if pending is None:
                k = F(1, int(it))
            elif pending < 1:
                pending = F(1, int(it))
            else:
                pending = F(int(it))

    if pending:
        k *= pending
    return k


@lru_cache(None)
def parse_cmd(cmd, value):
    if cmd == 'o':
        key = 'octave'
        value = int(value)
    elif cmd == 'g':
        key = 'gate'
        value = int(value)
    elif cmd == 'v':
        key = 'volume'
        value = int(value)
    elif cmd == 'c':
        key = 'channel'
        value = int(value)
    elif cmd == 'amul':
        key = 'accmul'
        value = float(value)
    elif cmd == 'k':
        key = 'key'
        value = get_key(value)

    def do_cmd(ctx):
        ctx[key] =  value

    return do_cmd


def sym_inc_dec(value, sinc, sdec):
    result = value.count(sinc) - value.count(sdec)
    return result, value.replace(sinc, '').replace(sdec, '')


class AbsStep(int):
    def resolve(self, scale):
        return self


class RelStep(int):
    def resolve(self, scale):
        soct, step = divmod(self-1, 7)
        return scale[step+1].value + soct*12


def parse_bar(line):
    result = Bar()

    parts = line.split()
    for p in parts:
        m = NOTE_RE.match(p)
        cmd = CMD_RE.match(p)
        if m:
            o, s, rest = m.group(1, 2, 3)
            off = sym_inc_dec(o, '>', '<')[0] * 12
            aoff = sym_inc_dec(rest, '#', 'b')[0]
            accent = sym_inc_dec(o, '!', '\'')[0]
            l = parse_length(rest)
            if s == '-':
                e = Event(TIE, l, None)
            elif s == '_':
                e = Event(REST, l, None)
            elif s in LIT_NOTES:
                e = Event(NOTE, l, Note(off+aoff, AbsStep(LIT_NOTES[s]), accent))
            else:
                e = Event(NOTE, l, Note(off+aoff, RelStep(int(s)), accent))
            result.append(e)
        elif cmd:
            result.append(Event(CMD, F(0), parse_cmd(*cmd.group(1, 2))))

    return result


def parse_line(line):
    parts = line.strip(' |').split('|')
    return Line([parse_bar(it) for it in parts])


class EventList(list):
    def __init__(self, data, duration):
        super().__init__(data)
        self.duration = duration

    def loop(self):
        start = F(0)
        while True:
            for it in self:
                yield (it[0] + start,) + it[1:]
            start += self.duration


NOTE_ON = 20
NOTE_OFF = 10


def eval_events(ctx, events):
    pos = F(0)
    result = []
    ctx = ctx.child()
    for ev in events:
        if ev.type == NOTE:
            note = ctx.octave, ev.value.offset + ev.value.step.resolve(ctx.key)
            result.append((pos, NOTE_ON, (ctx.channel, note, ctx.volume * (ctx.accmul ** ev.value.accent))))
            result.append((pos + ev.length * F(ctx.gate, 100), NOTE_OFF, (ctx.channel, note, 0)))
        elif ev.type == CMD:
            ev.value(ctx)
        pos += ev.length
    return EventList(result, pos)


def make_events(line, **kwargs):
    ctx = make_context(**kwargs)
    l = parse_line(line)
    events = l.unwrap(ctx)
    return eval_events(ctx, events)


def mix_events(event_lists):
    result = EventList([], 0)
    for it in event_lists:
        result.extend(it)

    result.sort()
    result.duration = max(it.duration for it in event_lists)
    return result


def take_until(loop):
    last = None
    loop = iter(loop)
    def take(until):
        nonlocal last
        if last is not None:
            if last[0] > until:
                return
            l = itertools.chain([last], loop)
        else:
            l = loop
        for last in l:
            if last[0] <= until:
                yield last
            else:
                break
        take.running = False

    take.running = True
    return take


def player_event_adapter(freq, bufsize):
    pos = F(0)
    tmul = F(freq) / bufsize * 60 * 4

    def step(player, taker, tempo):
        nonlocal pos, tmul
        for _, etype, evalue in taker(pos):
            if etype == NOTE_ON:
                ch, (o, offset), volume = evalue
                note = 12 + o*12 + offset
                player.note_on(ch, note, volume / 100)
            elif etype == NOTE_OFF:
                ch, (o, offset), _ = evalue
                note = 12 + o*12 + offset
                player.note_off(ch, note)

        pos += int(tempo) / tmul

    return step


def play_midi(events):
    import rtmidi
    import heapq
    import signal
    from rtmidi.midiconstants import ALL_SOUND_OFF, CONTROL_CHANGE, ALL_NOTES_OFF

    def stop_play(*args):
        flag[0] = False

    signal.signal(15, stop_play)

    o = rtmidi.MidiOut()
    o.open_port(1)

    events = list(events)
    heapq.heapify(events)

    start = time.monotonic()
    flag = [True]
    while flag[0] and events:
        now = time.monotonic()
        nt = heapq.heappop(events)
        if now - start < nt[0]:
            time.sleep(nt[0] - (now - start))
        print(nt[1])
        o.send_message(nt[1])

    o.send_message([CONTROL_CHANGE, ALL_NOTES_OFF, 0])
    o.send_message([CONTROL_CHANGE, ALL_SOUND_OFF, 0])
