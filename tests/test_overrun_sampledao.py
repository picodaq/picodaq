#!env python3



import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

import logging

logging.basicConfig(level=logging.INFO)

sys.path.append("../software")

from picodaq import AnalogIn, AnalogOut, V, kHz, ms, stimulus, DeviceError
import picodaq.dac

t0 = time.time()
def rtime(k):
    return f"{time.time() - t0:.3f} {(time.time() - t0)/max(k,1):.3f}"

def gen1():
    tt = np.arange(1000) / 1e4
    wavedata = np.sin(2*np.pi*tt * 200).astype(np.float32)
    while True:
        print("gen1")
        yield wavedata

def gen2():
    tt = np.arange(1000) / 1e4
    wavedata = (0*tt).astype(np.float32)
    while True:
        print("gen2")
        yield wavedata


        

def test_sampledao_overrun():
    with AnalogIn(channels=[0], rate=5*kHz) as ai:
        with AnalogOut(maxahead=500*ms) as ao:
            ao[3].sampled(gen1)
            #ao[1].sampled(gen2)
            for k in range(3000):
                dat = ai.read()
                print(rtime(k), k, ai.dev.reader.laststatus, np.mean(dat, 0), np.std(dat, 0))
                picodaq.dac._poll(ai.dev, 2)


def test_sampledao_overrun_fast():
    with AnalogIn(channels=[0], rate=25*kHz) as ai:
        with AnalogOut(maxahead=500*ms) as ao:
            ao[3].sampled(gen1)
            #ao[1].sampled(gen2)
            for k in range(3000):
                dat = ai.read()
                print(rtime(k), k, ai.dev.reader.laststatus, np.mean(dat, 0), np.std(dat, 0))
                picodaq.dac._poll(ai.dev, 2)
                                
test_sampledao_overrun()
