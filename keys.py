#python
import numpy as np
from ctypes import cdll, c_char, c_int, c_void_p, c_uint8, c_char_p, c_uint32

x11 = cdll.LoadLibrary('libX11.so')
x11.XOpenDisplay.restype = c_void_p

x11.XQueryKeymap.argtypes = c_void_p, c_char * 32

x11.XKeycodeToKeysym.argtypes = c_void_p, c_uint8, c_int
x11.XKeycodeToKeysym.restype = c_uint32

x11.XKeysymToString.argtypes = c_uint32,
x11.XKeysymToString.restype = c_char_p

_dpy = None
_state = (c_char * 32)()


def keyboard_state():
    global _dpy
    if not _dpy:
        _dpy = x11.XOpenDisplay(None)

    x11.XQueryKeymap(_dpy, _state)
    s = np.frombuffer(_state[:32], np.uint8)
    keys = np.nonzero(np.unpackbits(s, bitorder='little'))[0]
    syms = [x11.XKeysymToString(x11.XKeycodeToKeysym(_dpy, it, 0)).decode() for it in keys]
    return syms
