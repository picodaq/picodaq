#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append("../software")

from picodaq import AnalogIn, kHz, ms


plot = False


def timingtest():
    t0 = time.time()
    with AnalogIn(channel=0, rate=10*kHz) as ai:
        dt1 = time.time() - t0
        data = ai.read(1000)
        dt2 = time.time() - t0
    dt3 = time.time() - t0
    print(f"{dt1:.3f}")
    print(f"{dt2:.3f}")
    print(f"{dt3:.3f}")



def test_mustopen():
    ai = AnalogIn(channel=0, rate=10*kHz)
    with pytest.raises(ValueError):
        ai.read(1000)

    
def test_vector():
    with AnalogIn(channel=0, rate=10*kHz) as ai:
        data = ai.read(1000)
    assert len(data.shape) == 1
    assert len(data) == 1000
    if plot:
        plt.figure()
        plt.plot(data)
        plt.title("test_vector")
        plt.show()

def test_onearray():
    with AnalogIn(channels=[0], rate=10*kHz) as ai:
        data = ai.read(1000)
    assert len(data.shape) == 2
    assert len(data) == 1000
    assert data.shape[1] == 1
    if plot:
        plt.figure()
        plt.plot(data[:,0])
        plt.title("test_onearray")
        plt.show()

def test_raw():
    with AnalogIn(channel=0, rate=10*kHz) as ai:
        d1 = ai.read()
        d2 = ai.read(raw=True)
    assert d1.dtype == np.float32
    assert d2.dtype ==  np.int16

def test_twoarray():
    with AnalogIn(channels=[1, 3], rate=10*kHz) as ai:
        data = ai.read(1000)
    assert len(data.shape) == 2
    assert len(data) >= 1000
    assert data.shape[1] == 2
    if plot:
        plt.figure()
        plt.plot(data)
        plt.legend(['AI1', 'AI3'])
        plt.title("test_twoarray")
        plt.show()

    
def test_episodic_before_open():
    ai = AnalogIn(channel=0, rate=10*kHz)
    ai.episodic(duration=100*ms, period=200*ms)
    ai.open()
    t0 = time.time()
    data1 = ai.read()
    dt1 = time.time() - t0
    data2 = ai.read()
    dt2 = time.time() - t0
    ai.close()
    assert len(data1.shape) == 1
    assert data1.shape[0] == data2.shape[0]
    assert data1.shape[0] >= 1000
    assert 0.05 < dt1 < 0.15
    assert 0.25 < dt2 < 0.35

    
def test_episodic_after_open():
    with AnalogIn(channel=0, rate=10*kHz) as ai:
        ai.episodic(duration=100*ms, period=200*ms)
        t0 = time.time()
        data1 = ai.read()
        dt1 = time.time() - t0
        data2 = ai.read()
        dt2 = time.time() - t0
    assert len(data1.shape) == 1
    assert data1.shape[0] == data2.shape[0]
    assert data1.shape[0] >= 1000
    assert 0.05 < dt1 < 0.15
    assert 0.25 < dt2 < 0.35

def test_empty():
    with AnalogIn(channels=[], rate=10*kHz) as ai:
        t0 = time.time()
        dat = ai.read(10000)
        dt = time.time() - t0
        assert dat.shape[0] == 10000
        assert dat.shape[1] == 0
        assert 0.95 < dt  < 1.2


        

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
    
