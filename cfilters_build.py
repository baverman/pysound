from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(open("cfilters.h").read())

ffibuilder.set_source("_cfilters",
"""
#include "cfilters.c"
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
