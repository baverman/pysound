import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def lowpass(float[:] dst, float[:] src, float[:] alpha, float q, float[:] state):
    cdef float fb
    cdef float s0 = state[0]
    cdef float s1 = state[1]
    cdef float f
    for i in range(len(src)):
        f = alpha[i]
        if f < 0.9999:
            fb = q + q/(1.0 - f)
        else:
            fb = 0.0
        s0 = s0 + f * (src[i] - s0 + fb * (s0 - s1))
        s1 = s1 + f * (s0 - s1);
        dst[i] = s1

    state[0] = s0
    state[1] = s1
