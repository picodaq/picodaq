#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append("../software")

from picodaq import AnalogIn, AnalogOut, V, kHz, ms, stimulus


plot = False


def test_ai_combos():
    for msk in range(1, 16):
        chns = [c for c in range(4) if msk & (1<<c)]
        with AnalogIn(channels=chns, rate=10*kHz) as ai:
            t0 = time.time()
            dat = ai.read(1000)
            dt = time.time() - t0
            assert dat.shape[0] == 1000
            assert dat.shape[1] == len(chns)
            assert 0.095 < dt  < .3



def test_ao_ai_combos():
    if plot:
        plt.figure()
        plt.clf()

    pulses = [stimulus.Pulse(5*V, (20+10*k)*ms)
              for k in range(4)]
    trains = [stimulus.Train(pulses[k], 5, pulseperiod=(100+20*k)*ms)
              for k in range(4)]
        
    for imsk in range(1, 16):
        for omsk in range(1, 16):
            ichns = [c for c in range(4) if imsk & (1<<c)]
            ochns = [c for c in range(4) if omsk & (1<<c)]
            with AnalogIn(channels=ichns, rate=10*kHz) as ai:
                with AnalogOut() as ao:
                    for c in ochns:
                        ao[c].stimulus(trains[c])
                    ao.run()
                    data = ai.readall()
            if plot:
                if 3 in ochns and 2 in ichns:
                    plt.plot(data[:,ichns.index(2)])
             
            
if __name__ == "__main__":
    plt.ion()
    plt.close('all')
    plot = True
    vv = {k:v for k,v in vars().items()}
    for name, obj in vv.items():
        if name.startswith("test_") and callable(obj):
            obj()

    input("Press Enter to close")
    plt.close('all')
    
