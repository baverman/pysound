void dcfilter(float dst[], const float src[], size_t n, float state[], float R);

void lowpass(float dst[], const float src[], size_t n,
             const float alpha[], float q, float state[]);

void moog(float dst[], const float src[], size_t n,
          const float alpha[], float q, float state[]);

void pdvcf(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]);

void flt12(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]);

void bqlp(float dst[], const float src[], size_t n,
          const float cutoff[], float res, float state[]);

void delay_process(float buf[], size_t buf_size, const float src[],
                   size_t src_size, size_t shift, float feedback);

void pole2(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]);

struct ring_buf {
    float *data;
    size_t length;
    size_t length_mask;
    size_t start;
};

void delwrite(struct ring_buf *buf, const float src[], size_t length);

void delmix(struct ring_buf *buf, float dst[], const float src[], int length,
            const int32_t delay_samples[], float feedback);

void shold(float dst[], const float phase[], size_t n, float value, float prev);

void poly_saw(float *restrict dst, float *restrict phase, float *restrict delta, size_t n);

void poly_square(float dst[], float phase[], float fdelta[], float pw[], size_t n);


typedef struct {
    size_t scount;
    float last;
    int state;
    size_t srate;
    float speed;
    float release_level;
} env_adsr_state;

void env_adsr(float dst[], size_t n, env_adsr_state *state, float a, float h, float d, float s, float r);

float phasor(float *restrict dst, float *restrict delta, size_t n, float phase);


typedef struct {
    size_t srate;
    float last;
    int state;
    int decay_next_state;
    float lspeed;
    float rise_th;
    float fall_th;
    size_t hcount;
} env_adsr_exp_state;


void env_adsr_exp(float dst[], size_t n, env_adsr_exp_state *state, float a, float h, float d, float s, float r);
