# Calibrating the input of the picodaq lite, by attaching 
# AO1 to AI0 or AI1.
# The output must already have been calibrated, with results
# stored in firmware.

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pytest
from physfit import fit


sys.path.append("../software")

from picodaq import stimulus, V, s, ms, kHz, AnalogOut, AnalogIn, mockstim, DigitalOut

######################################################################

vv = np.random.random(50) * 9.5
dat = []
for k, v in enumerate(vv):
    pulse1 = stimulus.Square(v*V, .5*s)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=0) as ai:
            print(k, len(vv))
            ao.dev.igain = 10/32767.5
            ai.dev.ioffset = 0
            ao[0].stimulus(pulse1)
            ao[1].stimulus(pulse1)
            ao.run()
            dat.append(ai.readall())

#%%
vcmd = np.array([vv, -vv])
vread = []
for dat1 in dat:
    vread.append([np.mean(dat1[100:4900]), np.mean(dat1[5100:9900])])
vread = np.array(vread).T

plt.figure(1)
plt.clf()
plt.plot(vcmd.flatten(), vread.flatten(), '.')
f = fit('linear', vcmd.flatten(), vread.flatten())

plt.figure(2)
plt.clf()
plt.plot(vcmd.flatten(), 1e3*(vread.flatten() - f(vcmd.flatten())), '.')
plt.xlabel('V_command (V)')
plt.ylabel('Residual (mV)')
print(f)
print(1e3 * (1 - f.p[0]), 1e3 * f.p[1])

######################################################################

dat = []
for k, v in enumerate(vv):
    pulse1 = stimulus.Square(v*V, .5*s)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=0) as ai:
            print(k, len(vv))
            #ao.dev.igain = 10/32767.5
            #ai.dev.ioffset = 0
            ao[0].stimulus(pulse1)
            ao[1].stimulus(pulse1)
            ao.run()
            dat.append(ai.readall())

#%%
vcmd = np.array([vv, -vv])
vread = []
for dat1 in dat:
    vread.append([np.mean(dat1[100:4900]), np.mean(dat1[5100:9900])])
vread = np.array(vread).T

plt.figure(11)
plt.clf()
plt.plot(vcmd.flatten(), vread.flatten(), '.')
f = fit('linear', vcmd.flatten(), vread.flatten())

plt.figure(12)
plt.clf()
plt.plot(vcmd.flatten(), 1e3*(vread.flatten() - f(vcmd.flatten())), '.')
plt.xlabel('V_command (V)')
plt.ylabel('Residual (mV)')
print(f)

######################################################################
