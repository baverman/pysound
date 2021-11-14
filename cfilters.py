from _cfilters import ffi, lib

def addr(arr):
    return ffi.from_buffer("float[]", arr)

dcfilter = lib.dcfilter
lowpass = lib.lowpass
delay_process = lib.delay_process
