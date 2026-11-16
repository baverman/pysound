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


/*
* stage:
* 0 - off
* 1 - attack
* 2 - hold
* 3 - decay
* 4 - sustain
* 5 - release
*
* stop:
* 0 - sustain is enabled
* 1 - complete the decay, then release
* 2 - skip the decay, release after the hold
*/
#define ENV_OFF 0
#define ENV_ATTACK 1
#define ENV_HOLD 2
#define ENV_DECAY 3
#define ENV_SUSTAIN 4
#define ENV_RELEASE 5

static size_t env_samples(const env_state *state, float ms) {
    float count = ms / 1000.0f * (float)state->srate;
    return count > 0.0f ? (size_t)count : 0;
}

static void env_enter(env_state *state, int stage) {
    state->stage = stage;
    state->elapsed = 0;
}


typedef struct {
    size_t count;
    bool done;
} env_result;


/* Move last toward target over dur samples.
* Writes at most avail samples and reports how many and whether dur was reached.
*/
static env_result env_seg(float dst[], size_t avail, env_state *state,
                          float target, size_t dur, float k) {
    size_t left = state->elapsed < dur ? dur - state->elapsed : 0;

    if (left == 0) {
        state->last = target;
        return (env_result){0, true};
    }

    size_t count = avail < left ? avail : left;
    float v = state->last;
    float r = 1.0f;
    float add = (target - v) / (float)left;

    if (k != 0.0f) {
        float step = -k / (float)dur;
        float q = expf(step * (float)left);
        r = expf(step);
        add = (target - q * v) * -expm1f(step) / -expm1f(step * (float)left);
    }

    float lo = fminf(v, target);
    float hi = fmaxf(v, target);

    for (size_t j = 0; j < count; ++j) {
        v = fminf(fmaxf(v * r + add, lo), hi);
        dst[j] = v;
    }

    state->elapsed += count;
    bool done = state->elapsed >= dur;
    state->last = done ? target : v;
    return (env_result){count, done};
}


/* Hold the current level for dur samples. */
static env_result env_hold(float dst[], size_t avail, env_state *state, size_t dur) {
    size_t left = state->elapsed < dur ? dur - state->elapsed : 0;
    size_t count = avail < left ? avail : left;
    float last = state->last;

    for (size_t j = 0; j < count; ++j) {
        dst[j] = last;
    }

    state->elapsed += count;
    return (env_result){count, state->elapsed >= dur};
}


/* Sustain holds the live level. When it is changed, last glides to it. */
static size_t env_sustain(float dst[], size_t avail, env_state *state, float s) {
    float v = state->last;
    float span = s - v;
    size_t dur = (size_t)(fabsf(span) * (float)env_samples(state, state->glide));

    if (dur == 0) {
        state->last = s;
        for (size_t j = 0; j < avail; ++j) {
            dst[j] = s;
        }
        return avail;
    }

    size_t count = avail < dur ? avail : dur;
    float k = span > 0.0f ? state->rise_k : state->fall_k;
    float step = k != 0.0f ? -k / (float)dur : 0.0f;
    float r = k != 0.0f ? expf(step) : 1.0f;
    float add = (s - v) / (float)dur;

    if (k != 0.0f) {
        float q = expf(step * (float)dur);
        add = (s - q * v) * -expm1f(step) / -expm1f(step * (float)dur);
    }

    float lo = fminf(v, s);
    float hi = fmaxf(v, s);

    for (size_t j = 0; j < count; ++j) {
        v = fminf(fmaxf(v * r + add, lo), hi);
        dst[j] = v;
    }

    state->last = count == dur ? s : v;
    return count;
}


void env_adsr_exp(float dst[], size_t n, env_state *state,
                  float a, float h, float d, float s, float r) {
    size_t i = 0;

    while (i < n) {
        switch (state->stage) {
        case ENV_ATTACK: {
            env_result res = env_seg(dst + i, n - i, state, 1.0f,
                                     env_samples(state, a), state->rise_k);
            i += res.count;
            if (res.done) {
                env_enter(state, ENV_HOLD);
            }
            break;
        }

        case ENV_HOLD: {
            env_result res = env_hold(dst + i, n - i, state, env_samples(state, h));
            i += res.count;
            if (res.done) {
                env_enter(state, ENV_DECAY);
            }
            break;
        }

        case ENV_DECAY: {
            if (state->stop >= 2) {
                env_enter(state, ENV_SUSTAIN);
                break;
            }
            env_result res = env_seg(dst + i, n - i, state, s,
                                     env_samples(state, d), state->fall_k);
            i += res.count;
            if (res.done) {
                env_enter(state, ENV_SUSTAIN);
            }
            break;
        }

        case ENV_SUSTAIN:
            if (state->stop != 0) {
                env_enter(state, ENV_RELEASE);
            } else {
                i += env_sustain(dst + i, n - i, state, s);
            }
            break;

        case ENV_RELEASE: {
            env_result res = env_seg(dst + i, n - i, state, 0.0f,
                                     env_samples(state, r), state->fall_k);
            i += res.count;
            if (res.done) {
                env_enter(state, ENV_OFF);
            }
            break;
        }

        default:
            state->last = 0.0f;
            for (; i < n; ++i) {
                dst[i] = 0.0f;
            }
            break;
        }
    }

}
