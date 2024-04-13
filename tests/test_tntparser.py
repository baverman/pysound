from unittest.mock import ANY
import itertools

from pysound.tntparser import parse_line, make_context, F, eval_events, mix_events, take_until
from pysound.tonator import Scales


def I(v):
    return F(1, v)


def get_events(line, **kwargs):
    l = parse_line(line)
    # print(l)
    return list(l.unwrap(make_context(**kwargs)))


def evalevents(line, **kwargs):
    ctx = make_context(**kwargs)
    l = parse_line(line)
    events = l.unwrap(ctx)
    return eval_events(ctx, events)


def assert_lengths(events, *expected):
    value = [it.length for it in events]
    assert value == list(expected)


def assert_offsets(events, scale, *expected):
    value = [it.value.offset + it.value.step.resolve(scale) for it in events if it.type == 2]
    assert value == list(expected)


def test_simple():
    rv = get_events('1')
    assert_lengths(rv, I(1))

    rv = get_events('1 1 | 6 6 5 -')
    assert_lengths(rv, I(2), I(2), I(4), I(4), I(2))

    rv = get_events('1 1 1 | 6 6 5 -')
    assert_lengths(rv, I(3), I(3), I(3), I(4), I(4), I(2))

    rv = get_events('1 1--- 5--- _ _ 5- 3- _')
    assert_lengths(rv, I(16), I(4), I(4), I(16), I(16), I(8), I(8), I(16))

    rv = get_events('1 1** 5** _* 5* 3* _')
    assert_lengths(rv, I(16), I(4), I(4), I(8), I(8), I(8), I(16))

    rv = get_events('1:: 1 5 _: 5: 3: _::')
    assert_lengths(rv, I(16), I(4), I(4), I(8), I(8), I(8), I(16))

    rv = get_events('1: _. 5*')
    assert_lengths(rv, I(8), F(3/8), I(2))

    rv = get_events('1:: _:. 5')
    assert_lengths(rv, I(8), F(3/8), I(2))


def test_time_signature():
    rv = get_events('1', tsig=F(3, 4))
    assert_lengths(rv, F(3, 4))

    rv = get_events('1 1', tsig=F(3, 4))
    assert_lengths(rv, F(3, 8), F(3, 8))

    rv = get_events('1 1 1', tsig=F(3, 4))
    assert_lengths(rv, I(4), I(4), I(4))


def test_abs_step():
    rv = get_events('C D Eb >F')
    assert_offsets(rv, Scales.major, 0, 2, 3, 17)


def test_rel_step():
    rv = get_events('1 2 3b >4')
    assert_offsets(rv, Scales.major, 0, 2, 3, 17)


def test_eval_events():
    rv = evalevents("1 !1 '1")
    assert rv.duration == F(1)
    assert rv == [
        (F(0, 1), 20, (0, (3, 0), 80.0)),
        (F(1, 3), 10, (0, (3, 0), 0)),
        (F(1, 3), 20, (0, (3, 0), 88.0)),
        (F(2, 3), 10, (0, (3, 0), 0)),
        (F(2, 3), 20, (0, (3, 0), 72.72727272727272)),
        (F(1, 1), 10, (0, (3, 0), 0))]


def test_cmd():
    rv = evalevents('o=2 k=Dm v=70 c=1 1')
    assert rv == [
        (F(0), 20, (1, (2, 2), 70.0)),
        (F(1), 10, (1, (2, 2), 0))]


def test_mix_events():
    rv1 = evalevents('_ 1')
    rv2 = evalevents('2 _')
    rv = mix_events([rv1, rv2])
    assert rv.duration == 1
    assert rv == [
        (F(0), 20, (0, (3, 2), 80.0)),
        (F(1, 2), 10, (0, (3, 2), 0)),
        (F(1, 2), 20, (0, (3, 0), 80.0)),
        (F(1, 1), 10, (0, (3, 0), 0))]


def test_loop_events():
    loop = evalevents('1').loop()
    taker = take_until(loop)

    rv = list(taker(F(1)))
    assert rv == [
        (F(0), 20, ANY),
        (F(1), 10, ANY),
        (F(1), 20, ANY)]

    rv = list(taker(F(1)))
    assert rv == []

    rv = list(taker(F(2)))
    assert rv == [
        (F(2), 10, ANY),
        (F(2), 20, ANY)]
