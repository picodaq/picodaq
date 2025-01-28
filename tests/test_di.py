#!env python3

import unittest
import sys
import time
import numpy as np

sys.path.append("../software")

from picodaq import DigitalIn, kHz, ms

    
class TestDI(unittest.TestCase):
    def test_mustopen(self):
        di = DigitalIn(line=0, rate=10*kHz)
        self.assertRaises(ValueError, di.read, 1000)

    def test_vector(self):
        with DigitalIn(line=0, rate=10*kHz) as di:
            data = di.read(1000)
        self.assertEqual(len(data.shape), 1)
        self.assertGreaterEqual(len(data), 1000)

    def test_onearray(self):
        with DigitalIn(lines=[0], rate=10*kHz) as di:
            data = di.read(1000)
        self.assertEqual(len(data.shape), 2)
        self.assertGreaterEqual(len(data), 1000)
        self.assertEqual(data.shape[1], 1)

    def test_gap(self):
        def foo():
            with DigitalIn(lines=[0,3], rate=10*kHz) as di:
                data = di.read(1000)
            return data
        self.assertRaises(ValueError, foo)

    def test_rawsize(self):
        with DigitalIn(lines=[0], rate=10*kHz) as di:
            data1 = di.read(1000)
            data2 = di.read(1000, raw=True)
        self.assertEqual(data2.dtype, np.uint8)
        self.assertEqual(len(data1), 8*len(data2))

    def test_twoarray(self):
        with DigitalIn(lines=[1, 2], rate=10*kHz) as di:
            data = di.read(1000)
        self.assertEqual(len(data.shape), 2)
        self.assertGreaterEqual(len(data), 1000)
        self.assertEqual(data.shape[1], 2)

    def test_episodic_before_open(self):
        di = DigitalIn(line=0, rate=10*kHz)
        di.episodic(duration=100*ms, period=200*ms)
        di.open()
        t0 = time.time()
        data1 = di.read()
        t1 = time.time()
        data2 = di.read()
        t2 = time.time()
        di.close()
        self.assertEqual(len(data1.shape), 1)
        self.assertEqual(data1.shape[0], data2.shape[0])
        self.assertGreaterEqual(data1.shape[0], 1000)
        self.assertGreater(t1 - t0, 0.05)
        self.assertLess(t1 - t0, 0.15)
        self.assertGreater(t2 - t0, 0.25)
        self.assertLess(t2 - t0, 0.35)
        
    def test_episodic_after_open(self):
        with DigitalIn(line=0, rate=10*kHz) as di:
            di.episodic(duration=100*ms, period=200*ms)
            t0 = time.time()
            data1 = di.read()
            t1 = time.time()
            data2 = di.read()
            t2 = time.time()
        self.assertEqual(len(data1.shape), 1)
        self.assertEqual(data1.shape[0], data2.shape[0])
        self.assertGreaterEqual(data1.shape[0], 1000)
        self.assertGreater(t1 - t0, 0.05)
        self.assertLess(t1 - t0, 0.15)
        self.assertGreater(t2 - t0, 0.25)
        self.assertLess(t2 - t0, 0.35)



def test_empty():
    with DigitalIn(lines=[], rate=10*kHz) as di:
        t0 = time.time()
        dat = di.read(10000)
        dt = time.time() - t0
        assert dat.shape[0] == 10000
        assert dat.shape[1] == 0
        assert 0.95 < dt  < 1.2

        
if __name__ == "__main__":
    unittest.main()
    
