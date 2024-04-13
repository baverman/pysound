from pysound import core
from pysound.gui import HStack, VSlide


class EventSlice:
    def __init__(self, events):
        self.events = events
        self.pos = 0

    def iter_until(self, ts):
        e = self.events
        i = self.pos
        el = len(e)
        while i < el:
            if e[i][0] > ts:
                break
            yield e[i][1]
            i += 1
        self.pos = i


class arp:
    def __init__(self, ctl, player):
        self.ctl = ctl
        self.player = player
        self.scnt = 0
        self.pcnt = 0
        self.notes = []
        self.es = None
        self.nchanged = False
        self.oldtempo = 0

    def note_on(self, _channel, mnote, volume):
        self.notes.append((mnote, volume))
        self.nchanged = True

    def note_off(self, _channel, mnote):
        self.notes = [it for it in self.notes if it[0] != mnote]
        self.nchanged = True

    def make_events(self, notes):
        bl = self.ctl.get('arp_length', 7)
        bd = self.ctl.get('arp_div', 4)
        ol = self.ctl.get('arp_octaves', 1)
        tempo = self.ctl.get('tempo', 120)
        bscnt = int(60 / tempo * core.FREQ * 4 / bd)
        result = []
        s = 0
        bi = 0
        while bi < bl:
            for o in range(ol + 1):
                for n, v in notes:
                    if bi >= bl:
                        break
                    n = n + o*12
                    result.extend(((s, (self.player.note_on, (None, n, v))),
                                   (s + bscnt-core.BUFSIZE, (self.player.note_off, (None, n)))))
                    s += bscnt
                    bi += 1
        return result, s, tempo

    def __call__(self, result=None):
        result = core.pass_buf(result)

        if (not self.es or self.scnt < core.BUFSIZE) and (self.nchanged or self.oldtempo != self.ctl.get('tempo', 120)):
            n = self.notes[:]
            if n:
                self.nchanged = False
                self.scnt = 0
                events, self.pcnt, self.oldtempo = self.make_events(n)
                self.es = EventSlice(events)

        if self.es:
            for action, args in self.es.iter_until(self.scnt + core.BUFSIZE // 3):
                action(*args)

        result += self.player(result)

        self.scnt += core.BUFSIZE
        if self.scnt > self.pcnt:
            self.scnt -= self.pcnt
            n = self.notes[:]
            if n:
                if self.es:
                    self.es.pos = 0
            else:
                self.es = None

        return result
