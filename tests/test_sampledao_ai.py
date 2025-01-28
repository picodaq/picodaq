#!env python3

import pytest
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn, mockstim


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

    
def test_prefab():
    wavedata = np.sin(2*np.pi*np.arange(0, 0.2, 1e-4) / 0.02)
    pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    pulse2 = stimulus.Sawtooth(3*V, -3*V, 80*ms)
    train2 = stimulus.Train(pulse2, 5, pulseperiod=100*ms)
    with AnalogIn(channels=[2], rate=10*kHz) as ai:
        with AnalogOut() as ao:
            ao[3].sampled(wavedata, 1*V)
            
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data) / (10*kHz))
            dt2 = time.time() - t0
            
    if plot:
        plt.figure()
        plt.title("test_prefab")
        plt.plot(mockdata)
        plt.plot(data[:,0])
        plt.legend(["AO3", "AI2"])
    if assertdata:
        assertVecApprox(data[:,0], mockdata)

def test_longcontinuous():
    def gen():
        wavedata = np.sin(2*np.pi*np.arange(0, 0.02, 1e-4) / 0.02)
        while True:
            print("gen")
            yield wavedata
    with AnalogIn(channels=[2], rate=10*kHz) as ai:
        with AnalogOut() as ao:
            ao[3].sampled(gen, 1*V)
            for k in range(1000):
                dat = ai.read()
                ao.poll()
                print("got >> ", k, dat.shape, np.mean(dat), np.std(dat))


def test_twincontinuous():
    def gen():
        wavedata = np.sin(2*np.pi*np.arange(0, 0.02, 1e-4) / 0.02)
        while True:
            yield wavedata
    def gen2():
        wavedata = np.sin(2*np.pi*np.arange(0, 0.02, 1e-4) / 0.01)
        while True:
            yield wavedata
    with AnalogIn(channels=[0, 2], rate=10*kHz) as ai:
        with AnalogOut() as ao:
            ao[0].sampled(gen, 1*V)
            ao[3].sampled(gen2, 1*V)
            data = []
            for k in range(100):
                dat = ai.read()
                data.append(dat)
                ao.poll()
    if plot:
        data = np.concatenate(data, 0)
        plt.figure()
        plt.clf()
        plt.plot(data)
                



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
    
