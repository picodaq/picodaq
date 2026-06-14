import pytest
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn, mockstim

pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
with AnalogOut(rate=10*kHz) as ao:
    with AnalogIn(channels=[2]) as ai:
        ao[2].stimulus(train1)
        t0 = time.time()
        ao.run()
        dt1 = time.time() - t0
        data = ai.readall()
        mockdata = mockstim.mock(ao[2], len(data) / (10*kHz))
        dt2 = time.time() - t0

        
plt.figure(1)
plt.clf()
plt.title("test_sawtooth")
plt.plot(data)
plt.legend([f'{k}' for k in range(data.shape[1])])
