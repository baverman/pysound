import math
from collections.abc import Iterator

from pysound import core

core.FREQ = 1000
core.BUFSIZE = 64


def test_env_adsr_hold():
    env = core.env_adsr(stop=2)
    result = list(env(0, 0, 0, 0, 10))
    assert result[:11] == ([1] * 10 + [0])
