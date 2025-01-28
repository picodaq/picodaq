from picodaq.stimulus import Pulse, Train, Series
from picodaq.dac import AnalogOut
from picodaq.units import V, mV, s, ms, Hz, kHz 

pulse = Pulse(amplitude=5*V, duration=50*ms)
train = Train(pulse, pulsecount=100, pulseperiod=100*ms)
with AnalogOut(rate=10*kHz) as ao:
     ao[2].stimulus(train)
     ao.run()
