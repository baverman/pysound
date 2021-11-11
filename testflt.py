#PYTHONPATH=/home/bobrov/work python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pysound import FREQ, square_unlim_t, osc, fft_plot, env_ahr, fps, dcfilter
from timeit_helper import timeit

o = osc(square_unlim_t)
sig = (o(440) + 1.0) / 2.0
sig = dcfilter()(sig)

plt.plot(sig)
plt.show()
