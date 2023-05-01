// gcc -Wall -Wextra -Werror -O3 -fpic -shared -o _cfilters.so cfilters.c
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "cfilters.h"

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
        if (f > 0.9999) {
            f = 0.9999;
        }

        fb = q + q/(1.0 - f);
        s0 = s0 + f * (src[i] - s0 + fb * (s0 - s1));
        s1 = s1 + f * (s0 - s1);
        dst[i] = s1;
    }
    state[0] = s0;
    state[1] = s1;
}

void moog(float dst[], const float src[], size_t n,
             const float alpha[], float q, float state[]) {
    float x = 0;
    float oldx = state[0], y4=state[1],
          oldy1=state[2], y1=state[3],
          oldy2=state[4], y2=state[5],
          oldy3=state[6], y3=state[7];

    float kPi2 = 3.14f / 2.0f;

    for(size_t i=0; i<n; i++) {
        float p = alpha[i] * (1.8f - 0.8f * alpha[i]);
        float k = 2.0f * sin(alpha[i] * kPi2) - 1.0f;

        float t1 = (1.0f - p) * 1.386249f;
        float t2 = 12.0f + t1 * t1;
        float r = q * (t2 + 6.0f * t1) / (t2 - 6.0f * t1);

        /* float f = alpha[i]; */
        /* float p=f*(1.8f-0.8f*f); */
        /* float k=p+p-1.f; */
        /*  */
        /* float t=(1.f-p)*1.386249f; */
        /* float t2=12.f+t*t; */
        /* float r = q*(t2+6.f*t)/(t2-6.f*t); */

        x = src[i] - r*y4;

        //Four cascaded onepole filters (bilinear transform)
        y1 = x*p + oldx*p - k*y1;
        y2 = y1*p + oldy1*p - k*y2;
        y3 = y2*p + oldy2*p - k*y3;
        y4 = y3*p + oldy3*p - k*y4;

        //Clipper band limited sigmoid
        y4 = y4 - (y4*y4*y4)/6.0f;
        dst[i] = y4;

        oldx = x;
        oldy1 = y1;
        oldy2 = y2;
        oldy3 = y3;
    }
    state[0] = oldx; state[1] = y4;
    state[2] = oldy1; state[3] = y1;
    state[4] = oldy2; state[5] = y2;
    state[6] = oldy3; state[7] = y3;
}


void delay_process(float buf[], size_t buf_size, const float src[], size_t src_size, size_t shift, float feedback) {
    size_t i = 0;
    size_t start = buf_size - src_size;
    for(; i < src_size; i++, start++) {
        buf[start] = buf[start-shift] * feedback + src[i];
    }
}

void delwrite(struct ring_buf *buf, const float src[], size_t length) {
    size_t i, j;
    for(i=0, j=buf->start; i < length; i++, j++) {
        buf->data[j & buf->length_mask] = src[i];
    }
    buf->start = j & buf->length_mask;
}

/*
d = 3
--------s------
-----*--s------  i = 0
------*-s------  i = 1
-------*s------  i = 2
--------S------  i = 3

*/

void delmix(struct ring_buf *buf, float dst[], const float src[], int length,
            const int32_t delay_samples[]) {
    size_t k;
    int i, j;
    float v;
    for(i=0, j=buf->start; i < length; i++, j++) {
        if (delay_samples[i] <= i) {
            v = dst[i - delay_samples[i]];
        } else {
            k = (j - delay_samples[i]) & buf->length_mask;
            v = buf->data[k];
        }
        dst[i] = src[i] + v;
    }
}

void shold(float dst[], const float phase[], size_t n, float value, float prev) {
    for(size_t i=0; i<n; i++) {
        if (phase[i] < prev) {
            value = dst[i];
        } else {
            dst[i] = value;
        }
    }
}


float poly_blep(float t, float dt) {
  // 0 <= t < 1
  if (t < dt) {
    t /= dt;
    // 2 * (t - t^2/2 - 0.5)
    return t+t - t*t - 1.;
  } else if (t > 1. - dt) { // -1 < t < 0
    t = (t - 1.) / dt;
    // 2 * (t^2/2 + t + 0.5)
    return t*t + t+t + 1.;
  } else { // 0 otherwise
    return 0.;
  }
}


float poly_saw(float dst[], float dt[], size_t n, float t) {
  for(size_t i=0; i<n; i++) {
      if (t >= 1.) t -= 1.;
      dst[i] = 2.*t - 1. - poly_blep(t, dt[i]);
      t += dt[i];
  }
  return t;
}


float poly_square(float dst[], float dt[], float pw[], size_t n, float t) {
  for(size_t i=0; i<n; i++) {
      if (t >= 1.) t -= 1.;

      float t2 = t + 0.5;
      if (t2 >= 1.) t2 -= 1.;

      float out = 1.0;
      if (t > 0.5 + pw[i]) out = -1.0;

      dst[i] = out + poly_blep(t, dt[i]) - poly_blep(t2, dt[i]);
      t += dt[i];
  }
  return t;
}
