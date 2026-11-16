import math
from collections.abc import Iterator

from pysound import core

core.FREQ = 1000
core.BUFSIZE = 64


# def test_env_adsr():
#     env = core.env_ahr()
#     result = env(5, 10, 5)
#     print([float(it) for it in result], env.active)
