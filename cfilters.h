void dcfilter(float dst[], const float src[], size_t n, float state[], float R);

void lowpass(float dst[], const float src[], size_t n,
             const float alpha[], float q, float state[]);

void moog(float dst[], const float src[], size_t n,
          const float alpha[], float q, float state[]);

void pdvcf(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]);

void flt12(float dst[], const float src[], size_t n,
           const float alpha[], float q, float state[]);

void delay_process(float buf[], size_t buf_size, const float src[],
                   size_t src_size, size_t shift, float feedback);

struct ring_buf {
    float *data;
    size_t length;
    size_t length_mask;
    size_t start;
};

void delwrite(struct ring_buf *buf, const float src[], size_t length);

void delmix(struct ring_buf *buf, float dst[], const float src[], int length,
            const int32_t delay_samples[]);

void shold(float dst[], const float phase[], size_t n, float value, float prev);

float poly_saw(float dst[], float dt[], size_t n, float t);

float poly_square(float dst[], float dt[], float pw[], size_t n, float t);


typedef struct {
    size_t scount;
    float last;
    int state;
    size_t srate;
    float speed;
} env_ahdsr_state;

void env_ahdsr(float dst[], size_t n, env_ahdsr_state *state, float a, float h, float d, float s, float r);
