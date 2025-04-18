#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn, mockstim, DigitalOut

assertdata = False
plot = False


import logging

logging.basicConfig(level=logging.DEBUG)

def assertApprox(V, Vtarget, absdelta=.1):
    assert abs(V - Vtarget) < absdelta

    
def assertVecApprox(vv, vvtgt, absdelta=.1, tdelta=2):
    dv = []
    T = len(vv)
    for dt in range(-tdelta, tdelta+1):
        dv.append(vv[tdelta+dt:T-tdelta+dt] - vvtgt[tdelta:T-tdelta])
    dv = np.min(np.abs(dv), 0)
    assert np.max(dv) < absdelta


def xtest_sine():
    wave = np.sin(np.arange(10000)*2*np.pi/1000)*9.99
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=0) as ai:
            ao[1].sampled(wave, 1*V)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[1], len(data) / (10*kHz))
            dt2 = time.time() - t0
    if plot:
        plt.figure()
        plt.title("test_sine")
        plt.plot(mockdata)
        plt.plot(data)

    
def test_sawtooth():
    pulse1 = stimulus.Sawtooth(-2*V, 2*V, 80*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=0) as ai:
            ao[0].stimulus(train1)
            ao[1].stimulus(train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()#(0.5*s)
            mockdata = mockstim.mock(ao[1], len(data) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.480
            assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_sawtooth")
        plt.plot(mockdata)
        plt.plot(data)
        delta = data[1:] - mockdata[:-1]
        plt.figure()
        for k in range(5):
            plt.plot(delta[1001*k:1000*k+800])
        plt.legend([f"Delta {k}" for k in range(5)])
    if assertdata:
        assertVecApprox(data, mockdata)

def test_square():
    pulse1 = stimulus.Square(2*V, 1000*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=3000*ms)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=0) as ai:
            ao[0].stimulus(train1)
            ao[1].stimulus(train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()#(0.5*s)
            mockdata = mockstim.mock(ao[1], len(data) / (10*kHz))
            dt2 = time.time() - t0
            #assert dt1 > 0.480
            #assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_square")
        plt.plot(mockdata)
        plt.plot(data)
        delta = data[1:] - mockdata[:-1]
        plt.figure()
        for k in range(5):
            plt.plot(delta[1001*k:1000*k+800])
        plt.legend([f"Delta {k}" for k in range(5)])
    if assertdata:
        assertVecApprox(data, mockdata)


    
def xtest_ttl():
    pulse1 = stimulus.TTL(30*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with DigitalOut(rate=10*kHz) as do:
        with AnalogIn(channel=0) as ai:
            do[0].stimulus(train1)
            do[1].stimulus(train1)
            t0 = time.time()
            do.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(do[0], len(data) / (10*kHz))
            dt2 = time.time() - t0

    if plot:
        plt.figure()
        plt.title("test_ttl")
        plt.plot(mockdata)
        plt.plot(data)
        
if __name__ == "__main__":
    print("""Make sure there is a BNC cable between AO1 and AI0
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
