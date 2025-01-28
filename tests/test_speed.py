#!env python3

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytest

sys.path.append("../software")

from picodaq import AnalogIn, kHz, ms, Frequency, Time, s

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
    


if __name__ == "__main__":
    vv = {k:v for k,v in vars().items()}
    for name, obj in vv.items():
        if name.startswith("test_") and callable(obj):
            print(name)
            obj()
