void dcfilter(float dst[], const float src[], size_t n, float state[], float R);

void lowpass(float dst[], const float src[], size_t n,
             const float alpha[], float q, float state[]);

void delay_process(float buf[], size_t buf_size, const float src[],
                   size_t src_size, size_t shift, float feedback);
