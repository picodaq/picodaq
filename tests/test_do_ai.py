#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, DigitalOut, AnalogIn, mockstim


assertdata = False
plot = False


def assertApprox(V, Vtarget):
    assert abs(V - Vtarget) < 0.1

    
def test_ttl():
    pulse1 = stimulus.TTL(80*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with DigitalOut(rate=10*kHz) as do:
        with AnalogIn(channel=2) as ai:
            do[1].stimulus(train1)
            t0 = time.time()
            do.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(do[1], len(data) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.480
            assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_ttl")
        plt.plot(mockdata)
        plt.plot(data)
    if assertdata:
        assertApprox(data[799], 5)
        assertApprox(data[810], 0)
        assertApprox(data[1020], 5)
        assertApprox(data[4799], 5)


def test_lowttl():
    pulse1 = stimulus.TTL(80*ms, active_low=True)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with DigitalOut(rate=10*kHz) as do:
        with AnalogIn(channel=2) as ai:
            do[1].stimulus(train1)
            t0 = time.time()
            do.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(do[1], len(data) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.480
            assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_ttl")
        plt.plot(mockdata)
        plt.plot(data)
    if assertdata:
        assertApprox(data[799], 0)
        assertApprox(data[810], 5)
        assertApprox(data[1020], 0)
        assertApprox(data[4799], 0)
        
        

if __name__ == "__main__":
    print("""Make sure there is a BNC cable between DO1 and AI2
    before running this test""")
    assertdata = True
    plot = True
    plt.ion()
    plt.close('all')
    
    vv = {k:v for k,v in vars().items()}
    for name, obj in vv.items():
        if name.startswith("test_") and callable(obj):
            obj()


    input("Press Enter to close")
    plt.close('all')
    
    
