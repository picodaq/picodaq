#!env python3

"""Pipe output from this to

       pdserver ACM0 10000 ai0 ai2 ao0 ao3 > /tmp/x.dat

   Then load the data with:

       import numpy as np
       import matplotlib.pyplot as plt
       plt.ion()
       with open("/tmp/x.dat", "rb") as fd:
           bts = fd.read()
           dat = np.frombuffer(bts, np.float32)
       plt.clf()
       plt.plot(dat)

"""

import numpy as np
import sys
import time
import os
import io

rate = 10000
f = 200
tt = np.arange(300).astype(float) / rate
data = np.sin(2*np.pi*tt*f).astype(np.float32)

with io.FileIO(sys.stdout.fileno(), "wb") as stdout:
    for k in range(400):
        stdout.write(((0.2+0.2*k)*data).astype(np.float32).tobytes())
        time.sleep(1)
    
