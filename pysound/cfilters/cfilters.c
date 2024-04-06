// gcc -Wall -Wextra -Werror -O3 -fpic -shared -o _cfilters.so cfilters.c
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "cfilters.h"

#define MUUG_PI 3.14159265358979323846


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
    float s0 = state[1];
    float s1 = state[2];
    float f;
    float maxf = 10000. * 2. * MUUG_PI / state[0];

    for(size_t i=0; i<n; i++) {
        f = (alpha[i] > 1. ? 1.0 : alpha[i]) * maxf;
        if (f < 0.) f = 0.;
        fb = q + q/(1.0 - f/1.5);
        f = sin(f);
        s0 = s0 + f * (src[i] - s0 + fb * (s0 - s1));
        s1 = s1 + f * (s0 - s1);
        dst[i] = s1;
    }
    state[1] = s0;
    state[2] = s1;
}


void moog(float dst[], const float src[], size_t n,
          const float alpha[], float q, float state[]) {
    float v   = 2.;
    float ya1 = state[1];
    float wa1 = state[2];
    float yb1 = state[3];
    float wb1 = state[4];
    float yc1 = state[5];
    float wc1 = state[6];
    float yd1 = state[7];
    for (size_t i = 0; i < n; ++i) {
        float g = 1 - expf(-2. * MUUG_PI * alpha[i] * 10000. / state[0]);
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
        dst[i] = y*(1+q*4);
    }
    state[1] = ya1;
    state[2] = wa1;
    state[3] = yb1;
    state[4] = wb1;
    state[5] = yc1;
    state[6] = wc1;
    state[7] = yd1;
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
            const int32_t delay_samples[], float feedback) {
    int start = buf->start;
    size_t mask = buf->length_mask;
    for(int i=0; i < length; i++) {
        size_t k = (start + i - delay_samples[i]) & mask;
        float v = buf->data[k];
        dst[i] = src[i] + v * feedback;
        buf->data[(start + i) & mask] = dst[i];
    }
    buf->start = (buf->start + length) & mask;
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
    return t+t - t*t - 1.f;
  } else if (t > 1.f - dt) { // -1 < t < 0
    t = (t - 1.f) / dt;
    // 2 * (t^2/2 + t + 0.5)
    return t*t + t+t + 1.f;
  } else { // 0 otherwise
    return 0.f;
  }
}


void poly_saw(float *restrict dst, float *restrict phase, float *restrict fdelta, size_t n) {
  for(size_t i=0; i<n; i++) {
      float t = phase[i];
      dst[i] = 2.f*t - 1.f - poly_blep(t, fdelta[i]);
  }
}


void poly_square(float dst[], float phase[], float fdelta[], float pw[], size_t n) {
  for(size_t i=0; i<n; i++) {
      float t = phase[i];
      float t2 = t + 0.5 - pw[i];
      if (t2 >= 1.) t2 -= 1.;

      float out = 1.0;
      if (t > 0.5 + pw[i]) out = -1.0;

      dst[i] = out + poly_blep(t, fdelta[i]) - poly_blep(t2, fdelta[i]);
  }
}

static inline
float calc_step(float start, float end, size_t i, size_t iend) {
    if (i >= iend) {
        return end;
    }
    return (end - start) / (float)(iend - i);
}

void env_adsr(float dst[], size_t n, env_adsr_state *state, float a, float h, float d, float s, float r) {
    size_t acnt = a / 1000.0 * state->srate;
    size_t hcnt = h / 1000.0 * state->srate;
    size_t dcnt = d / 1000.0 * state->srate;
    size_t rcnt = r / 1000.0 * state->srate;

    float val = state->last;
    size_t i = state->scount, j=0;

    // 0 trigger active
    // 1 trigger released, should continue to play a/h/d and then state 2
    // 2 trigger released, should stop anyway and then release
    // 3 play finished

    if (state->state < 2) {
        float step = calc_step(val, 1.0, i, acnt);
        for(; i<acnt && j<n; i++, j++) {
            val += step;
            dst[j] = val;
        }

        for(; i<acnt+hcnt && j<n; i++, j++) {
            dst[j] = 1.0;
        }

        step = calc_step(val, s, i, acnt+hcnt+dcnt);
        for(; i<acnt+hcnt+dcnt && j<n; i++, j++) {
            val += step;
            dst[j] = val;
        }
    }

    if (state->state == 1 && j < n) {
        state->state = 2;
        state->scount = 0;
        state->release_level = val;
        i = 0;
    }

    if (state->state == 0) {
        for(;j<n; j++) {
            dst[j] = s;
        }
    }

    if (state->state == 2) {
        float step = calc_step(val, 0.0, i, rcnt);
        for(; i<rcnt && j<n; i++, j++) {
            val += step;
            dst[j] = val;
        }
        for(;j<n; j++) {
            dst[j] = 0.0;
        }
        if (i >= rcnt) {
            state->state = 3;
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

void pole2(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]) {
    float s1 = state[1];
    float s2 = state[2];
    float R = 1.0f - q;
    float maxf = MUUG_PI / 3.0f;
    for(size_t i = 0; i < n; i++) {
        float g = (alpha[i] > 1. ? 1.0 : alpha[i]) * maxf;
        float g1 = 2.0f*R + g;
        float d = 1.0f / (1.0f + 2.0f*R*g + g*g);
        float HP = (tanh(src[i]) - g1*s1 - s2) * d;
        float v1 = g*HP; float BP = v1 + s1; s1 = BP + v1;
        float v2 = g*BP; float LP = v2 + s2; s2 = LP + v2;
        dst[i] = LP*1.33f;
    }
    state[1] = s1;
    state[2] = s2;
}


// https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
void bqlp(float dst[], const float src[], size_t n,
          const float cutoff[], float res, float state[]) {

    float x1 = state[1];
    float x2 = state[2];
    float y1 = state[3];
    float y2 = state[4];
    float maxf = 10000. * 2. * MUUG_PI / state[0];
    float q = 0.7 + res*7.;

    for(size_t i = 0; i < n; i++) {
        float co = cutoff[i];
        float f = maxf*(co < 0. ? 0. : (co > 1. ? 1. : co));
        float sf = sin(f);
        float cf = cos(f);
        float alpha = sf/2./q;
        float a0 = 1. + alpha;
        float a1 = -2*cf;
        float a2 = 1. - alpha;
        float b0 = (1.-cf)/2.;
        float b1 = 1.-cf;
        float b2 = (1.-cf)/2.;
        float y = dst[i] = (b0*src[i] + b1*x1 + b2*x2 - a1*y1 - a2*y2)/a0;
        x2 = x1;
        x1 = src[i];
        y2 = y1;
        y1 = y;
    }

    state[1] = x1;
    state[2] = x2;
    state[3] = y1;
    state[4] = y2;
}


float phasor(float *restrict dst, float *restrict delta, size_t n, float phase) {
    for(size_t i=0; i<n; ++i) {
        dst[i] = phase;
        phase += delta[i];
        if (phase > 1.f) {
            phase -= 1.f;
        }

    }
    return phase;
}


typedef struct {
    size_t i;
    float v;
    bool done;
} exp_result;


exp_result exp_fill(float dst[], size_t n, size_t i, float last, float dur, float th, float d, float o) {
    float p = fmax(0, (last - o) / d);
    if (dur < 1 || p <= th) {
        return (exp_result){i, last, true};
    }
    float r = powf(th, 1.0f/dur);
    size_t oi = i;
    while ((p > th) && (i < n)) {
        p = p*r;
        dst[i] = p*d + o;
        i += 1;
    }
    return (exp_result){i, i == oi ? last : dst[i-1], p <= th};
}

exp_result exp_rise(float dst[], size_t n, size_t i, float last, float dur, float th) {
    float d = 1.0f / (th - 1.0f);
    float o = -d;
    return exp_fill(dst, n, i, last, dur, th, d, o);
}

exp_result exp_fall(float dst[], size_t n, size_t i, float last, float dur, float th, float start, float stop) {
    float d = start - stop;
    if ( fabs(d) < 0.00001 ) {
        return (exp_result){i, last, true};
    }
    float o = stop - th * d;
    return exp_fill(dst, n, i, last, dur, th, d, o);
}

exp_result line(float dst[], size_t n, size_t i, float value, float target, float speed, bool fill) {
    float step = 1.0/speed;
    size_t count, j;

    if (target > value) {
        count = (target - value) * speed;
    } else {
        count = (value - target) * speed;
        step = -step;
    }

    for(j=0; (j < count) && (i < n); ++i, ++j) {
        value += step;
        dst[i] = value;
    }

    if ((j >= count) && (i < n)) {
        dst[i] = value = target;
        i++;
        if (fill) {
            for(; i < n; ++i) {
                dst[i] = target;
            }
        }
    }

    return (exp_result){i, value, !fill && j >= count};
}

/*
* state:
* 0 - idle
* 1 - attack
* 2 - hold
* 3 - decay
* 4 - sustain
* 5 - release
* 6 - bring to 0
*/
void env_adsr_exp(float dst[], size_t n, env_adsr_exp_state *state, float a, float h, float d, float s, float r) {
    #define spms(dur) (dur / 1000.0f * state->srate)

    exp_result er = {0, state->last};
    if (state->state == 1) { // attack
        er = exp_rise(dst, n, er.i, er.v, spms(a), state->rise_th);
        if (er.done) {
            state->state = 2;
            state->hcount = 0;
        }
    }

    if (state->state == 2) { // hold
        size_t hcnt = spms(h);
        size_t j = state->hcount;
        for(; (er.i < n) && (j < hcnt) ; ++er.i, ++j) {
            dst[er.i] = er.v;
        }
        state->hcount = j;
        if (j >= hcnt) {
            state->state = 3;
        }
    }

    if (state->state == 3) { // decay
        er = exp_fall(dst, n, er.i, er.v, spms(d), state->fall_th, 1.0f, s);
        if (er.done) {
            state->state = state->decay_next_state;
        }
    }

    if (state->state == 4) { // sustain
        er = line(dst, n, er.i, er.v, s, spms(state->lspeed), true);
    }

    if (state->state == 5) { // release
        er = exp_fall(dst, n, er.i, er.v, spms(r), state->fall_th, s, 0.0f);
        if (er.done) {
            state->state = 6;
        }
    }

    if (state->state == 6) { // bring to 0
        er = line(dst, n, er.i, er.v, 0.0f, spms(state->lspeed), false);
        if (er.done) {
            state->state = 0;
        }
    }

    if (state->state == 0) { // idle
        er.v = 0.0f;
        for(; er.i < n; ++er.i) {
            dst[er.i] = 0.0f;
        }
    }

    state->last = er.v;
}
