# python

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]

PATTERNS = {
    (3, 4): 'm',
    (3, 4, 3): 'm7',
    (3, 4, 4): 'm/M7',

    (4, 3): '',
    (4, 3, 3): '7',
    (4, 3, 4): 'maj7',

    (3, 3): 'dim',
    (3, 3, 3): 'dim7',
    (3, 3, 4): 'm7b5',

    (4, 4): 'aug',
    (4, 4, 2): 'aug7',
    (4, 4, 3): '5#7',
}

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class Formattable:
    def __format__(self, fmt):
        return self.__str__().__format__(fmt)

    def __repr__(self):
        return self.__str__()


class Note(Formattable):
    def __init__(self, value, label, degree=None):
        self.value = value
        self.label = label
        self.degree = degree
        self.norm_value = value % 12

    def __str__(self):
        return self.label  # + str(self.degree)

    def __add__(self, amount):
        value = (self.value + amount) % 12
        return Note(value, NOTES[value], self.degree)

    def __sub__(self, note):
        return (self.value - note.value) % 12

    def diff(self, note):
        a = self.value - note.value
        b = self.value - (note.value + 12)
        return min((abs(a), a), (abs(b), b))[1]

    def __hash__(self):
        return hash(self.norm_value)

    def __eq__(self, other):
        return self.norm_value == other.norm_value


class Chord(Formattable):
    def __init__(self, notes):
        self.notes = notes

    def guess_type(self):
        pattern = tuple((b.value - a.value) % 12
                        for a, b in zip(self.notes[:-1], self.notes[1:]))
        return PATTERNS[pattern]

    def __str__(self):
        return '{}{}'.format(self.notes[0], self.guess_type())


def degree_to_step(degree, steps=MAJOR_STEPS):
    diff = degree.count('#') + degree.count('♯') - degree.count('b') - degree.count('♭')
    return steps[int(degree.strip('♯#b♭')) - 1] + diff


class Scale:
    def __init__(self, notes, relabel=True):
        if relabel:
            self.notes = [Note(n.value, n.label, str(i))
                          for i, n in enumerate(notes, 1)]
        else:
            self.notes = notes

    def shift(self, amount=1):
        notes = self.notes[amount:] + self.notes[:amount]
        return Scale(notes)

    def to(self, amount=1, relabel=False):
        if type(amount) is str:
            amount = NOTES.index(amount)
        return Scale([it + amount for it in self.notes], relabel=relabel)

    def __getitem__(self, degree):
        degree = (degree - 1) % 7
        return self.notes[degree]

    def __iter__(self):
        return iter(self.notes)

    def steps(self, *steps):
        return [self[it] for it in steps]

    def chord(self, *steps):
        return Chord([self[it] for it in steps])

    def triad(self, degree):
        return Chord([self[it] for it in (degree, degree+2, degree+4)])

    def c7th(self, degree):
        return Chord([self[it] for it in (degree, degree+2, degree+4, degree+6)])

    def triads(self):
        return [self.triad(s) for s in range(1, len(self.notes)+1)]

    def c7ths(self):
        return [self.c7th(s) for s in range(1, len(self.notes)+1)]

    def diff(self, scale):
        return [a.diff(b) for a, b in zip(self.notes, scale.notes)]

    def note_diff(self, scale):
        return map(note_diff, zip(self.notes, self.diff(scale)))

    @staticmethod
    def make(degrees, start=0, steps=MAJOR_STEPS):
        degrees = degrees.split()
        notes = [(start + degree_to_step(it, steps)) % 12 for it in degrees]
        return Scale([Note(n, NOTES[n], d) for n, d in zip(notes, degrees)], relabel=False)


class Scales:
    major = ionian = Scale.make('1 2 3 4 5 6 7')
    minor = aeolian = Scale.make('1 2 ♭3 4 5 ♭6 ♭7')
    harmonic_minor = Scale.make('1 2 ♭3 4 5 ♭6 7')
    dorian = Scale.make('1 2 ♭3 4 5 6 ♭7')
    locrian = Scale.make('1 ♭2 ♭3 4 ♭5 ♭6 ♭7')
    lydian = Scale.make('1 2 3 ♯4 5 6 7')
    mixolydian = Scale.make('1 2 3 4 5 6 ♭7')
    phrygian = Scale.make('1 ♭2 ♭3 4 5 ♭6 ♭7')
    minor_pentatonic = Scale.make('1 ♭3 4 5 ♭7')
    major_pentatonic = Scale.make('1  2 3 5  6')
    blues = Scale.make('1 ♭3 4 ♭5 5 ♭7')
    major_blues = Scale.make('1 2 ♭3 3 5 6')


def plist(values, width=8):
    print(''.join(f'{str(it):<{width}}' for it in values))


DIFFS = {
    -2: '♭♭',
    -1: '♭',
    0: '',
    1: '#',
    2: '##'
}


def note_diff(item):
    note, diff = item
    d = DIFFS[diff]
    if d:
        d = f'({d})'
    return '{}{}'.format(note, d)


def list_modes(start):
    major = Scale.major(start)
    plist(major.notes)
    plist(major.triads())
    plist(major.c7ths())
    print()

    for s in range(1, 7):
        mode = major.shift(s)
        trans = major.transpose(mode[1] - major[1])
        plist(mode.note_diff(trans))
        # plist(mode.notes, 8)
        # plist(trans.notes, 8)
        # plist(mode.diff(trans), 8)
        # print()
        # plist(mode.triads(), 8)
        # plist(mode.c7ths(), 8)
        # print()


def print_chords(start, m):
    major = Scale.major(start)
    mode = major.shift(m)
    plist(mode.note_diff(major.transpose(mode[1] - major[1])))
    plist(mode.triads())
    plist(mode.c7ths())


def print_neck(notes, length=19):
    result = []
    result.append('   ' + ' '.join(f'{str(it+1).center(3)}' for it in range(length)))
    values = [4, 11, 7, 2, 9, 4]

    pv = {}
    for n in notes:
        pv[n.value] = n.degree
        pv[n.value+12] = n.degree
        pv[n.value+24] = n.degree

    for l in range(6):
        result.append(f'{pv.get(values[l], "║"):>3}' + '|'.join(
            f'{pv.get(values[l]+it + 1, " ").center(3)}' for it in range(length)) + '|')

    print('\n'.join(result))


if __name__ == '__main__':
    plist(Scales.blues.to('E'))
    print_neck(Scales.blues.to('E'))

    print()
    plist(Scales.major_blues.to('E'))
    print_neck(Scales.major_blues.to('E'))
