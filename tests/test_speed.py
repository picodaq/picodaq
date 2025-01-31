#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append("../software")

from picodaq import AnalogIn, kHz, ms, Frequency, Time, s, stimulus, AnalogOut, V, mockstim

def run_speed(nchannels: int, rate: Frequency, duration: Time):
    chans = [k for k in range(nchannels)]
    t0 = time.time()
    nextdt = 0
    dur_s = int(duration.as_("s"))
    with AnalogIn(channels=chans, rate=rate) as ai:
        while True:
            dt = time.time() - t0
            if dt >= nextdt:
                print(f"t = {nextdt} / {dur_s} s")
                nextdt += 1
            if dt > dur_s:
                break
            dat = ai.read()
    print()
    print("ok")


def test_4_50():
    run_speed(4, 50*kHz, 60*s)

def test_4_75():
    run_speed(4, 75*kHz, 60*s)

def test_4_77():
    run_speed(4, 77*kHz, 60*s)

def test_4_120():
    with pytest.raises(RuntimeError):
        run_speed(4, 120*kHz, 5*s)
    

def test_1_1_300():
    pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
    train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
    rate = 300 * kHz
    with AnalogOut(rate=rate) as ao:
        with AnalogIn(channel=2) as ai:
            ao[3].stimulus(train1)
            t0 = time.time()
            ao.run()
            dt1 = time.time() - t0
            data = ai.readall()
            mockdata = mockstim.mock(ao[3], len(data) / rate)
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
    #if assertdata:
    #    assertVecApprox(data, mockdata)    
        

if __name__ == "__main__":
    vv = {k:v for k,v in vars().items()}
    for name, obj in vv.items():
        if name.startswith("test_") and callable(obj):
            print(name)
            obj()
