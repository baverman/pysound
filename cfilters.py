import numpy as np
from _cfilters import ffi, lib

def addr(arr, t="float[]"):
    return ffi.from_buffer(t, arr)

dcfilter = lib.dcfilter
lowpass = lib.lowpass
delay_process = lib.delay_process


def init_ring_buf(length):
    bits = length.bit_length()
    length = 1 << bits

    data = np.zeros(length, dtype=np.float32)

    buf = ffi.new('struct ring_buf *')
    buf.data = addr(data)
    buf.length = length
    buf.length_mask = length - 1
    buf.start = 0
    return buf, data


def delwrite(buf, data):
    lib.delwrite(buf[0], addr(data), len(data))


def delmix(buf, dst, src, delay):
    lib.delmix(buf[0], addr(dst), addr(src), len(src), addr(delay, 'int32_t[]'))
    return dst


def shold(dst, src, value, prev):
    lib.shold(addr(dst), addr(src), len(src), value, prev)
    return dst
