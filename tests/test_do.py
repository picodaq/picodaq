#!env python3

import unittest
import sys
import time
import numpy as np

sys.path.append("../software")

from picodaq import stimulus, V, ms, kHz, DigitalOut, AnalogIn


class TestDO(unittest.TestCase):
    def test_do_with_ai(self):
        pulse1 = stimulus.TTL(80*ms)
        train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
        with DigitalOut(rate=10*kHz) as do:
            with AnalogIn(channel=0) as ai:
                do[2].stimulus(train1)
                t0 = time.time()
                do.run()
                dt1 = time.time() - t0
                data = ai.readall()
                dt2 = time.time() - t0
                self.assertGreaterEqual(dt1, 0.480)
                self.assertLess(dt2, 0.600)

                
    def test_do_with_ai_episodic(self):
        pulse1 = stimulus.TTL(8*ms)
        train1 = stimulus.Train(pulse1, 2, pulseperiod=10*ms)
        series1 = stimulus.Series(train1, traincount=6)
        with DigitalOut(rate=10*kHz) as do:
            with AnalogIn(channel=0) as ai:
                do[0].stimulus(series1)
                ai.episodic(duration=120*ms, period=150*ms)
                t0 = time.time()
                do.run()
                dt1 = time.time() - t0
                data = ai.readall()
                dt2 = time.time() - t0
                self.assertGreaterEqual(dt1, 0.420)
                self.assertLess(dt2, 0.500)
                self.assertEqual(len(data.shape), 2)
                self.assertEqual(data.shape[0], 3)
                self.assertGreaterEqual(data.shape[1], 180)
                
    def test_do_without_ai(self):
        pulse1 = stimulus.TTL(8*ms)
        train1 = stimulus.Train(pulse1, 2, pulseperiod=10*ms)
        series1 = stimulus.Series(train1, 3, trainperiod=100*ms)
        do = DigitalOut(rate=10*kHz)
        do.open()
        do[2].stimulus(series1)
        t0 = time.time()
        do.run()
        dt1 = time.time() - t0
        do.close()
        self.assertGreaterEqual(dt1, 0.218)
        self.assertLess(dt1, 0.300)
        

if __name__ == "__main__":
    unittest.main()
    
