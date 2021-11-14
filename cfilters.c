// gcc -Wall -Wextra -Werror -O3 -fpic -shared -o _cfilters.so cfilters.c
#include <stddef.h>
#include <stdint.h>

void dcfilter(float dst[], const float src[], size_t n, float state[], float R) {
    float px = state[0];
    float py = state[1];
    for(size_t i=0; i < n; i++) {
        dst[i] = py = src[i] - px + R*py;
        px = src[i];
    }
    state[0] = px;
    state[1] = py;
}

void lowpass(float dst[], const float src[], size_t n,
             const float alpha[], float q, float state[]) {
    float fb;
    float s0 = state[0];
    float s1 = state[1];
    float f;
    for(size_t i=0; i<n; i++) {
        f = alpha[i];
        if (f < 0.9999) {
            fb = q + q/(1.0 - f);
        } else {
            fb = 0.0;
        }
        s0 = s0 + f * (src[i] - s0 + fb * (s0 - s1));
        s1 = s1 + f * (s0 - s1);
        dst[i] = s1;
    }

    state[0] = s0;
    state[1] = s1;
}

void delay_process(float buf[], size_t buf_size, const float src[], size_t src_size, size_t shift, float feedback) {
    size_t i = 0;
    size_t start = buf_size - src_size;
    for(; i < src_size; i++, start++) {
        buf[start] = buf[start-shift] * feedback + src[i];
    }
}
