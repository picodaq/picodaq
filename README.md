# picodaq — Python library for picoDAQ data acquisition

## Introduction

Data acquisition is a central aspect of many types of scientific
experiments. Commonly, some measurement device outputs a time-varying
voltage which you would like to capture on a computer. With picodaq,
this can be as easy as

    with AnalogIn(channel=0, rate=50*kHz) as ai:
        data = ai.read(10*s)

(See below for fully worked examples.)

At present, the picodaq library supports our soon-to-be-released
picoDAQ hardware. Support for other popular hardware (including National
Instruments and Measurement Computing) is planned as well.
    
### Example of use



## Installation

As easy as

    pip install picodaq
    

## Fully worked examples

### Single-channel data acquisition

We weren't kidding! After

    from picodaq import *
    import matplotlib.pyplot as plt
    
you really can acquire 10 seconds of data, sampled at 50 kHz from
channel “0” of your data acquisition board simply by

    with AnalogIn(channel=0, rate=50*kHz) as ai:
        data = ai.read(10*s)

and plot the results with

    plt.plot(data)

### Multi-channel data acquisition

Of course, you can also capture several channels at once, and the
“read” method can be made to return time stamps so you can have time
in seconds rather than sample numbers on your x-axis:

    with AnalogIn(channels=[0,2], rate=50*kHz) as ai:
        data, time_s = ai.read(10*s, times=True)
    plt.plot(time_s, data)


## Development

Development of picodaq is on [github](https://github.com/wagenadl/picodaq).
