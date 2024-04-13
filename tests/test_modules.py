from pysound.modules import EventSlice


def test_event_slice():
    es = EventSlice([(0, 'boo'), (2, 'foo')])
    assert list(es.iter_until(-1)) == []
    assert list(es.iter_until(0)) == ['boo']
    assert list(es.iter_until(0)) == []
    assert list(es.iter_until(1)) == []
    assert list(es.iter_until(3)) == ['foo']
    assert list(es.iter_until(4)) == []
