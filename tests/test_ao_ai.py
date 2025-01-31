#!env python3

import pytest
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn, mockstim


import logging
#logging.basicConfig(level=logging.DEBUG)


assertdata = False
plot = False


def assertApprox(V, Vtarget, absdelta=.1):
    assert abs(V - Vtarget) < absdelta

    
def assertVecApprox(vv, vvtgt, absdelta=.1, tdelta=2):
    dv = []
    T = len(vv)
    for dt in range(-tdelta, tdelta+1):
        dv.append(vv[tdelta+dt:T-tdelta+dt] - vvtgt[tdelta:T-tdelta])
    dv = np.min(np.abs(dv), 0)
    assert np.max(dv) < absdelta


    
def test_sawtooth():
    pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=2) as ai:
            ao[3].stimulus(train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.480
            assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_sawtooth")
        plt.plot(mockdata)
        plt.plot(data)
        plt.legend(["AO3", "AI2"])
        delta = data[1:] - mockdata[:-1]
        plt.figure()
        for k in range(5):
            plt.plot(delta[1001*k:1000*k+800])
        plt.legend([f"Delta {k}" for k in range(5)])
    if assertdata:
        assertVecApprox(data, mockdata)

def test_pulse():
    pulse1 = stimulus.Pulse(5*V, 70*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=2) as ai:
            ao[3].stimulus(train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.480
            assert dt2 < 0.600
    if plot:
        plt.figure()
        plt.title("test_pulse")
        plt.plot(mockdata)
        plt.plot(data)
        plt.legend(["AO3", "AI2"])
    if assertdata:
        assertVecApprox(data, mockdata)


def test_sawtooth_episodic():
    pulse1 = stimulus.Sawtooth(-3*V, 3*V, 8*ms)
    train1 = stimulus.Train(pulse1, 2, pulseperiod=10*ms)
    series1 = stimulus.Series(train1, 3)
    with AnalogOut(rate=10*kHz) as ao:
        with AnalogIn(channel=2) as ai:
            ao[3].stimulus(series1)
            ao.episodic(duration=120*ms, period=150*ms)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data.T) / (10*kHz))
            dt2 = time.time() - t0
            assert dt1 > 0.420
            assert dt2 < 0.500
            assert len(data.shape) == 2
            assert data.shape[0] == 3
            assert data.shape[1] >= 180
    if plot:
        plt.figure()
        plt.title("test_sawtooth_episodic")
        plt.plot(mockdata[0])
        plt.plot(data.T)
        plt.legend(["AO3", "AI2"])
    if assertdata:
        for dat in data:
            assertVecApprox(dat, mockdata[0])


def test_pulseshapes():
    pulse1 = stimulus.Pulse(amplitude=2*V, duration=12*ms)
    pulse2 = stimulus.Square(amplitude=2*V, duration=12*ms)
    pulse3 = stimulus.Triangle(amplitude=2*V, duration=12*ms)
    perpulse = stimulus.Deltas(amplitude=0.1*V, duration=1*ms)
    train1 = stimulus.Train(pulse1, pulsecount=5, pulseperiod=20*ms,
                            perpulse=perpulse)
    with AnalogIn(channel=2, rate=10*kHz) as ai:
        with AnalogOut() as ao:
            ao[3].stimulus(train1)
            ao[0].stimulus(pulse2)
            ao[1].stimulus(pulse3)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            dt2 = time.time() - t0
    assert dt1 > 4*0.020 + 0.012 + 4*0.001
    assert dt2 < 0.300

def test_wave():
    wavedata = np.sin(2*np.pi*np.arange(0, 0.1, 1e-4) / 0.1)
    pulse1 = stimulus.Wave(wavedata)
    train1 = stimulus.Train(pulse1, pulsecount=5, pulseperiod=100*ms)
    with AnalogIn(channel=2, rate=10*kHz) as ai:
        with AnalogOut() as ao:
            ao[3].stimulus(stim=train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data.T) / (10*kHz))
            dt2 = time.time() - t0
    if plot:
        plt.figure()
        plt.plot(mockdata)
        plt.plot(data)
        plt.legend(["AO3", "AI2"])
    if assertdata:
        assertVecApprox(data, mockdata)
                   
        

if __name__ == "__main__":
    print("""Make sure there is a BNC cable between AO3 and AI2
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
