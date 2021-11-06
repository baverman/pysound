import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def lowpass(float[:] dst, float[:] src, float[:] alpha, float acc):
    for i in range(len(src)):
        acc += alpha[i] * (src[i] - acc)
        dst[i] = acc
