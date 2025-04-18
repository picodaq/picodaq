# Calibrating the output of the picodaq lite, with a multimeter attached to
# AO0 or AO1

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pytest
from physfit import fit


sys.path.append("../software")

from picodaq import stimulus, V, s, ms, kHz, AnalogOut, AnalogIn, mockstim, DigitalOut

for v in range(1, 10):
    pulse1 = stimulus.Square(v*V, 4*s)
    with AnalogOut(rate=10*kHz) as ao:
        ao.dev.ogain = (3276.7, 3276.7) # disable slope
        ao[0].stimulus(pulse1)
        ao[1].stimulus(pulse1)
        ao.run()

#%%

def plot(ch):
    vv = np.array([np.arange(10), -np.arange(10)]).T.flatten()
    ch1 = np.array(ch)
    plt.figure(1)
    plt.clf()
    plt.plot(vv, ch, '.')

    f = fit('linear', vv, ch)
    print(f)
    plt.plot(np.arange(-10, 11), f(np.arange(-10, 11)))

    plt.figure(2)
    plt.clf()
    plt.plot(vv, 1e3 * (ch - f(vv)), '.')
    plt.xlabel('Command (V)')
    plt.ylabel('Residual (mV)')
    
#%%
# This is AO1
ch1 = [-.215, -.215,
       .881, -1.310,
       1.977, -2.405,
       3.073, -3.501,
       4.168, -4.596,
       5.264, -5.692,
       6.359, -6.79,
       7.46, -7.88,
       8.55, -8.98,
       9.65, -10.07]
plot(ch1)
######################################################################
# This is AO0
ch0 = [-.193, -.193,
       0.879,-1.303,
       1.964,-2.392,
       3.055,-3.480,
       4.148,-4.571,
       5.232,-5.661,
       6.327,-6.75,
       7.41,-7.84,
       8.50,-8.93,
       9.59,-10.02]

plot(ch0)
######################################################################

ch = [-.007, -.007, .013, .013,
      1.089,-1.102,1.084,-1.094,
      2.185,-2.198,2.173,-2.185,
      3.280,-3.294,3.266,-3.275,
      4.376,-4.389,4.351,-4.363,
      5.471,-5.485,5.443,-5.455,
      6.567,-6.580,6.535,-6.543,
      7.66,-7.67,7.62,-7.63,
      8.76,-8.77,8.71,-8.72,
      9.85,-9.87,9.80,-9.81]
ch = np.array(ch).reshape(10,4)
plot(ch[:,:2].flatten())
#%%
plot(ch[:,2:].flatten())
#%%
plot((ch[:,:2].flatten() + ch[:,2:].flatten())/2)

######################################################################

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pytest
from physfit import fit


sys.path.append("../software")

from picodaq import stimulus, V, s, ms, kHz, AnalogOut, AnalogIn, mockstim, DigitalOut

for v in range(1, 10):
    pulse1 = stimulus.Square(v*V, 1*s)
    with AnalogOut(rate=10*kHz) as ao:
        ao[0].stimulus(pulse1)
        ao[1].stimulus(pulse1)
        ao.run()
