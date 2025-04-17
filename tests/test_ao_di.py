#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, DigitalIn, mockstim


assertdata = False
plot = False


def assertTrue(V):
    assert V == True

    
def assertFalse(V):
    assert V == False

    
def test_pulse():
    pulse1 = stimulus.Pulse(5*V, 7*ms)
    train1 = stimulus.Train(pulse1, 50, pulseperiod=10*ms)
    with AnalogOut(rate=10*kHz) as ao:
        with DigitalIn(lines=[0,1]) as di:
            ao[3].stimulus(train1, delay=0.5*ms)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = di.readall()
            dt2 = time.time() - t0
            mockdata = mockstim.mock(ao[3], len(data) / (10*kHz))
    assert dt1 > 0.480
    assert dt2 < 0.800
    if plot:
        plt.figure()
        plt.title("test_pulse")
        plt.plot(mockdata)
        plt.plot(data)
        plt.legend(["AO3", "DI0", "DI1"])
    if assertdata:
        for k in range(50):
            assertFalse(data[100*k + 3,1])
            assertTrue(data[100*k + 8,1])
            assertTrue(data[100*k + 73,1])
            assertFalse(data[100*k + 78,1])
                   

def test_pulse_episodic():
    pulse1 = stimulus.Pulse(5*V, 7*ms)
    train1 = stimulus.Train(pulse1, 2, pulseperiod=10*ms)
    series1 = stimulus.Series(train1, 3)
    with AnalogOut(rate=10*kHz) as ao:
        with DigitalIn(line=1) as di:
            ao[3].stimulus(series1, delay=0.5*ms)
            di.episodic(duration=120*ms, period=150*ms)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = di.readall()
            dt2 = time.time() - t0
            mockdata = mockstim.mock(ao[3], len(data.T) / (10*kHz))
    assert dt1 > 0.420
    assert dt2 < 0.500
    assert len(data.shape) == 2
    assert data.shape[0] == 3
    assert data.shape[1] >= 180
    if plot:
        plt.figure()
        plt.title("test_pulse_episodic")
        plt.plot(mockdata[0])
        plt.plot(data.T)
        plt.legend(["AO3", "DI1 #0"])
    if assertdata:
        for n in range(3):
           for k in range(2):
               assertFalse(data[n, 100*k + 3])
               assertTrue(data[n, 100*k + 8])
               assertTrue(data[n, 100*k + 73])
               assertFalse(data[n, 100*k + 78])
                

if __name__ == "__main__":
    print("""Make sure there is a BNC cable between AO3 and DI1
    before running this test""")
    input()
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
    


    
