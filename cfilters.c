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


#define MUUG_PI 3.14159265358979323846

void moog(float dst[], const float src[], size_t n,
          const float alpha[], float q, float state[]) {
    float v   = 2.0;
    float ya1 = state[0];
    float wa1 = state[1];
    float yb1 = state[2];
    float wb1 = state[3];
    float yc1 = state[4];
    float wc1 = state[5];
    float yd1 = state[6];
    for (size_t i = 0; i < n; ++i) {
        float g = 1 - expf(-2. * MUUG_PI * alpha[i] * 20000. / state[7]);
        float ya = ya1 + v * g * tanhf((src[i] - 4 * q * yd1) / v - wa1);
        float wa = tanhf(ya / v);
        float yb = yb1 + v * g * (wa - wb1);
        float wb = tanhf(yb / v);
        float yc = yc1 + v * g * (wb - wc1);
        float wc = tanhf(yc / v);
        float yd = yd1 + v * g * (wc - tanhf(yd1 / v));
        float y = yd;
        ya1 = ya;
        wa1 = wa;
        yb1 = yb;
        wb1 = wb;
        yc1 = yc;
        wc1 = wc;
        yd1 = yd;
        dst[i] = y;
    }
    state[0] = ya1;
    state[1] = wa1;
    state[2] = yb1;
    state[3] = wb1;
    state[4] = yc1;
    state[5] = wc1;
    state[6] = yd1;
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

      float t2 = t + 0.5 - pw[i];
      if (t2 >= 1.) t2 -= 1.;

      float out = 1.0;
      if (t > 0.5 + pw[i]) out = -1.0;

      dst[i] = out + poly_blep(t, dt[i]) - poly_blep(t2, dt[i]);
      t += dt[i];
  }
  return t;
}


void env_ahdsr(float dst[], size_t n, env_ahdsr_state *state, float a, float h, float d, float s, float r) {
    size_t acnt = a / 1000.0 * state->srate;
    size_t hcnt = h / 1000.0 * state->srate;
    size_t dcnt = d / 1000.0 * state->srate;
    size_t rcnt = r / 1000.0 * state->srate;
    float speed = state->speed;

    float val = state->last;
    float target = 0.0;
    size_t i=state->scount, j=0;

    if (state->state == 0) {
        for(; i<acnt && j<n; i++, j++) {
            target = i/(float)acnt;
            dst[j] = val = val + (target - val)/speed;
        }

        for(; i<acnt+hcnt && j<n; i++, j++) {
            target = 1.0;
            dst[j] = val = val + (target - val)/speed;
        }

        for(; i<acnt+hcnt+dcnt && j<n; i++, j++) {
            target = 1.0 - (i - acnt - hcnt) / (float)dcnt * (1.0 - s);
            dst[j] = val = val + (target - val)/speed;
        }

        for(;j<n; j++) {
            target = s;
            dst[j] = val = val + (target - val)/speed;
        }
    } else {
        for(; i<rcnt && j<n; i++, j++) {
            target = s - i/(float)rcnt*s;
            dst[j] = val = val + (target - val)/speed;
        }
        for(;j<n; j++) {
            target = 0.0;
            dst[j] = val = val + (target - val)/speed;
        }
        if (i >= rcnt) {
            state->state = 2;
        }
    }

    state->last = val;
    state->scount = i;
}


void pdvcf(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]) {

    float maxf = 10000. * 2. * MUUG_PI / state[0];
    float re = state[1], re2;
    float im = state[2];
    q = q * 10;
    float qinv = (q > 0? 1.0f/q : 0);
    float ampcorrect = 2. - 2. / (q + 2.);
    float coefr, coefi;

    for(size_t i = 0; i < n; i++) {
        float cf, r, oneminusr;
        cf = (alpha[i] > 1. ? 1.0 : alpha[i]) * maxf;
        if (cf < 0) cf = 0;
        r = (qinv > 0 ? 1 - cf * qinv : 0);
        if (r < 0) r = 0;
        oneminusr = 1.0f - r;
        coefr = r * cos(cf);
        /* coefi = r * cos(cf - MUUG_PI/2.); */
        coefi = r * sin(cf);

        re2 = re;
        re = ampcorrect * oneminusr * src[i] + coefr * re2 - coefi * im;
        im = coefi * re2 + coefr * im;
        dst[i] = re;
    }

    state[1] = re;
    state[2] = im;
}

void flt12(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]) {

    float vibrapos = state[1];
    float vibraspeed = state[2];
    float maxf = 10000. * 2. * MUUG_PI / state[0];

    float amp = 1. + q*4.;

    for(size_t i = 0; i < n; i++) {
        float w = (alpha[i] > 1. ? 1.0 : alpha[i]) * maxf;
        if (w < 0) w = 0;
        float pm = 1.0-w/(2.0*(amp+0.5/(1.0+w))+w-2.0); // Pole magnitude
        float r = pm*pm;
        float c = r+1.0-2.0*cos(w)*pm;

        vibraspeed += (src[i] - vibrapos) * c;
        vibrapos += vibraspeed;
        vibraspeed *= r;

        float temp = vibrapos;
        if (temp > 32767.)
            temp = 32767.;
        else if (temp < -32768.)
            temp = -32768.;

        dst[i] = temp;
    }
    state[1] = vibrapos;
    state[2] = vibraspeed;
}
