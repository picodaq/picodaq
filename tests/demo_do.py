#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, DigitalOut, AnalogIn, mockstim

plt.close('all')
f, ax = plt.subplots(2,2)
ax = ax.flatten()
for line in range(4):
    pulse1 = stimulus.TTL(20*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with DigitalOut(rate=10*kHz) as do:
        with AnalogIn(channel=0) as ai:
            do[line].stimulus(train1)
            t0 = time.time()
            do.run()
            dt1 = time.time() - t0
            data = ai.readall()
    ax[line].plot(data)
    ax[line].set_ylim(-1, 6)
    ax[line].set_title(f"DO{line} to AI0")
