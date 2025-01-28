#!env python3

import pytest
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn, mockstim

pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
series1 = stimulus.Series(train1, traincount=2, trainperiod=600*ms)
stim = stimulus.Parametrized(series1)

data = mockstim.mockstim(stim, 10*kHz, 1500*ms)
plt.figure(1)
plt.clf()
plt.plot(data)

