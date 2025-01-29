#!env python3

import unittest
import sys
import time
import numpy as np

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, AnalogOut, AnalogIn


class TestAO(unittest.TestCase):
    def test_sawtooth(self):
        pulse1 = stimulus.Sawtooth(-3*V, 3*V, 8*ms)
        train1 = stimulus.Train(pulse1, 2, pulseperiod=10*ms)
        series1 = stimulus.Series(train1, 3, trainperiod=100*ms)
        ao = AnalogOut(rate=10*kHz)
        ao.open()
        ao[0].stimulus(series1)
        t0 = time.time()
        ao.run()
        dt1 = time.time() - t0
        ao.close()
        self.assertGreaterEqual(dt1, 0.218)
        self.assertLess(dt1, 0.300)
      

if __name__ == "__main__":
    unittest.main()
    
