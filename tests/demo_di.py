#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../software")

from picodaq import *
from picodaq import stimulus

pulse1 = stimulus.TTL(20*ms)
train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)

plt.close('all')
f, ax = plt.subplots(2,2)
ax = ax.flatten()
for line in range(4):
    with DigitalOut(rate=10*kHz) as do:
        with DigitalIn(line=line) as di:
            do[0].stimulus(train1)
            t0 = time.time()
            do.run()
            dt1 = time.time() - t0
            data = di.readall()
    ax[line].plot(data)
    ax[line].set_ylim(-1, 6)
    ax[line].set_title(f"DO0 to DI{line} indiv")


f, ax = plt.subplots(2,2)
ax = ax.flatten()
with DigitalOut(rate=10*kHz) as do:
    with DigitalIn(lines=[0,1,2,3]) as di:
        do[0].stimulus(train1)
        t0 = time.time()
        do.run()
        dt1 = time.time() - t0
        data = di.readall()
for line in range(4):
    ax[line].plot(data[:,line])
    ax[line].set_ylim(-1, 6)
    ax[line].set_title(f"DO0 to DI{line} simult")

