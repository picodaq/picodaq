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


def test_ai_slow():
    with AnalogIn(channel=0, rate=5*kHz) as ai:
        dat1 = ai.read(1000, raw=True)
        print("sleeping too long")
        for k in range(20):
            print(f"{k} / 20", end="\r")
            time.sleep(1)
        with pytest.raises(DeviceError):
            dat2 = ai.read(1000, raw=True)
            ai.stop()

def test_ai_timeout():
    for t in [0.1, 0.2, 0.4, 0.5, 2]:
        with AnalogIn(channels=[0,1], rate=20*kHz) as ai:
            t0 = time.time()
            print("  ( pre acquiring )")
            for k in range(10):
                dat = ai.read(1000)
            dt1 = time.time() - t0
            print("  ( sleeping )")
            time.sleep(t)
            dt2 = time.time() - t0
            print("  ( post acquiring )")
            try:
                if t > 1:
                    with pytest.raises(DeviceError):
                        for k in range(10):
                            dat = ai.read(1000)
                else:
                    for k in range(10):
                        dat = ai.read(1000)
            finally:
                print("  ( finally )")
                ai.stop()
                ai.dev.command("nop")
                print(ai.dev.params)
            dt3 = time.time() - t0
            print(f"sleep {t} ({dt2-dt1:.3f}) pre {dt1:.3f} post {dt3-dt2:.3f}")
            print()


            

if __name__ == "__main__":
    vv = {k:v for k,v in vars().items()}
    for name, obj in vv.items():
        if name.startswith("test_") and callable(obj):
            print(name)
            obj()
    
