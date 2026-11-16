import numpy as np
from . _cfilters import ffi, lib

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


def delmix(buf, dst, src, delay, feedback):
    lib.delmix(buf[0], addr(dst), addr(src), len(src), addr(delay, 'int32_t[]'), feedback)
    return dst


def shold(dst, src, value, prev):
    lib.shold(addr(dst), addr(src), len(src), value, prev)
    return dst


STAGE_OFF = 0
STAGE_ATTACK = 1
STAGE_HOLD = 2
STAGE_DECAY = 3
STAGE_SUSTAIN = 4
STAGE_RELEASE = 5


def env_init(srate, last=0.0, rise_k=-2.0, fall_k=4.0, glide=10.0):
    state = ffi.new('env_state *')
    state.srate = srate
    state.stage = STAGE_OFF
    state.stop = 0
    state.elapsed = 0
    state.last = last
    state.rise_k = rise_k
    state.fall_k = fall_k
    state.glide = glide
    return state
