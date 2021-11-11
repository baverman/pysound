#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pysound import FREQ, square_unlim_t, osc, fft_plot, env_ahr, fps, dcfilter
from timeit_helper import timeit

sig = np.random.rand(512).astype(np.float32)
sig = dcfilter(0.9)(sig)

plt.plot(sig)
plt.show()
