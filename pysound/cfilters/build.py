import os.path
from cffi import FFI
ffibuilder = FFI()

src_dir = os.path.dirname(__file__)
ffibuilder.cdef(open(os.path.join(src_dir, "cfilters.h")).read())

ffibuilder.set_source("pysound.cfilters._cfilters",
"""
#include "cfilters.c"
""", include_dirs=[src_dir])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
