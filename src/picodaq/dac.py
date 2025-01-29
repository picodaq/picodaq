from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, Iterable
import logging

from .device import PicoDAQ
from .stream import Stream
from .stimulus import Pulse, Train, Series, Parametrized, Sampled
from .units import V, mV, s, ms, Hz, kHz, Time, Voltage, Frequency, Quantity
from .decorators import with_doc
from .utils import makemask, NScanCalc
from .binreader import FLAGS_STIMACTIVE

log = logging.getLogger(__name__)
debug = False

def _poll(dev, _forceqty=None):
    if not dev.reader:
        raise RuntimeError("Cannot poll if not started")
    dev.reader.read()
    if debug:
        log.debug(f"_poll {dev.reader.lastflags}")
    if (dev.reader.lastflags & FLAGS_STIMACTIVE) == 0:
        return True
    if not dev.writer:
        return False
    outcnt = dev.reader.laststatus
    if dev.aheadchunks < 128:
        N = min(4, dev.aheadchunks - outcnt)
    else:
        N = min(4, 200 - outcnt)
    # We must send multiple chunks if the writechunk is smaller than
    # the readchunk. The limit 4 is arbitrary.
    if _forceqty is not None:
        N = _forceqty
    if debug:
        log.debug(f"poll...{N} {outcnt}")
    for k in range(N):
        if debug:
            log.debug("sendchunk")
        dev.writer.sendchunk(False)
        dev.reader.laststatus += 1
        if debug:
            log.debug("sentchunk")
        if dev.writer.finished:
            break
    return False


class OutRef:
    """Helper class to allow access to stimuli for a single channel or line.

    """

    def __init__(self, stream: Stream, idx: int):
        self.stream = stream
        self.idx = idx

    @with_doc(Parametrized)
    def stimulus(self,
                 stim: Pulse | Train | Series,
                 delay: Time = 0*s,
                 repeat: Time | None = None,
                 offset: Voltage = 0*V):
        self.stream._stimulus(self.idx, stim, delay, repeat, offset)

    @with_doc(Sampled)
    def sampled(self,
                data: ArrayLike | Callable,
                scale: Voltage = 1*V, offset: Voltage = 0*V,
                raw: bool = False):
        self.stream._sampled(self.idx, data, scale, offset, raw)

        
class AnalogOut(Stream):
    """Main interface for stimulating through analog channels

    Parameters:
        rate: Sampling frequency for output
        port: Serial port to open
        maxahead: Max. number of samples to preload

    The `rate` may be specified in Hz or kHz. When using multiple
    streams, the rates must all be the same and only need to be
    specified on the first-opened stream.

    The `port` specifies which serial port to open. Use
    ``picodaq.devices()`` to retrieve the list of available
    ports.  If you do not specify a port, the most recently opened
    device is used, or the first device on the system if none was
    opened before.

    The `maxahead` parameter specifies the maximum number of samples
    preloaded to the device during open() and through poll(). If
    omitted, this is optimized to reduce risk of underruns. The most
    common reason to set it explicitly is to reduce latency for
    dynamically generated "sampled" outputs.

    The stimuli themselves are added by calling the `stimulus()` or
    `sampled()` methods on the individual channels, which may be
    accessed using indexing syntax, as in the following example::

        pulse1 = stimulus.Sawtooth(-3*V, 3*V, 80*ms)
        train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
        with AnalogOut(rate=30*kHz) as ao:
            ao[2].stimulus(train1)
            ao.run()

    """

    def __init__(self,
                 rate: Frequency | None = None,
                 port: str | None = None,
                 maxahead: int | Time | None = None):
        super().__init__(port, rate)
        self.stimuli = {} # dict of channel to Parametrized or Sampled
        if isinstance(maxahead, Quantity):
            maxahead = int((self.dev.rate * maxahead).plain())
        self.maxahead = maxahead
        self.committed = False

    def __getitem__(self, channel: int):
        """Access to a single channel.

        This provides a convenient syntax for specifying stimulation
        sequences.
        
        Example::

            pulse = Monophasic(1*V, 10*ms)
            train = Train(pulse, pulsecount=5, pulseperiod=20*ms)
            with AnalogOut(rate=10*kHz) as ao:
                ao[1].stimulus(train)

        """
        return OutRef(self, channel)

    def _stimulus(self,
                 channel: int,
                 stim: Pulse | Train | Series,
                 delay: Time = 0*s,
                 repeat: Time | None = None,
                 offset: Voltage = 0*V):
        self.stimuli[channel] = Parametrized(stim, delay, repeat, offset)
        self.committed = False

    def _sampled(self, channel: int,
                 data: ArrayLike | Callable,
                 scale: Voltage = 1*V, offset: Voltage = 0*V,
                 raw: bool = True):
        self.stimuli[channel] = Sampled(data, scale, offset, raw)
        self.committed = False

    def _Vtodigital(self, y: Voltage) -> int:
        """Convert a voltage to a digital value.

        The current implementation assumes S10 range.
        """
        y = y.as_(V)
        return int(32767.99999*y/10)

    def _Ttosamples(self, t: Time) -> int:
        """Convert a time to a digital value.

        This uses the device's sampling rate.
        """
        return int((t * self.dev.rate).plain())

    def _configwave(self, chan, data, scale, pd_scale, td_scale):
        bindata = np.round(data * scale)
        bindata[bindata < -32768] = -32768
        bindata[bindata > 32767] = 32767
        bindata = bindata.astype(np.int16)
        pdA1 = int(np.round(pd_scale/scale * 256))
        tdA1 = int(np.round(td_scale/scale * 256))
        self.dev.sendwave(chan, bindata)
        self.dev.command(f"pulse A{chan} wave {chan}")
        return pdA1, tdA1

    def _configstim(self, chan: int, stim: Parametrized):
        def sendcmd(cmd, *args):
            pfx = f"{cmd} A{chan}"
            self.dev.command(pfx + "".join([f" {a}" for a in args]))
            
        name = stim.series.train.pulse.name
        A1 = self._Vtodigital(stim.series.train.pulse.amplitude1)
        A2 = self._Vtodigital(stim.series.train.pulse.amplitude2)
        T1 = self._Ttosamples(stim.series.train.pulse.duration1)
        T2 = self._Ttosamples(stim.series.train.pulse.duration2)
        npulse = stim.series.train.pulsecount
        pulseival = self._Ttosamples(stim.series.train.pulseperiod)
        pdA1 = self._Vtodigital(stim.series.train.perpulse.amplitude1)
        pdA2 = self._Vtodigital(stim.series.train.perpulse.amplitude2)
        pdT1 = self._Ttosamples(stim.series.train.perpulse.duration1)
        pdT2 = self._Ttosamples(stim.series.train.perpulse.duration2)
        pdpival = self._Ttosamples(stim.series.train.perpulse.pulseperiod)

        delay = self._Ttosamples(stim.delay)
        trainival = self._Ttosamples(stim.series.trainperiod)
        ntrain = stim.series.traincount

        tdA1 = self._Vtodigital(stim.series.pertrain.amplitude1)
        tdA2 = self._Vtodigital(stim.series.pertrain.amplitude2)
        tdT1 = self._Ttosamples(stim.series.pertrain.duration1)
        tdT2 = self._Ttosamples(stim.series.pertrain.duration2)
        tdpival = self._Ttosamples(stim.series.train.perpulse.pulseperiod)
        tdn = stim.series.pertrain.pulsecount
        tdtival = self._Ttosamples(stim.series.train.perpulse.trainperiod)

        trepeat = self._Ttosamples(stim.repeat) if stim.repeat else None

        offset = self._Vtodigital(stim.offset)

        sendcmd("aorange", "S10")
        if name == "wave":
            pdA1, tdA1 = self._configwave(chan,
                                          stim.series.train.pulse.data,
                                          A1, pdA1, tdA1)
        else:
            sendcmd("pulse", name, A1, T1, A2, T2)
        sendcmd("train", npulse, pulseival)
        sendcmd("perpulse", pdpival, pdA1, pdT1, pdA2, pdT2)
        sendcmd("series", delay, ntrain, trainival, tdtival, tdn)
        sendcmd("pertrain", tdpival, tdA1, tdT1, tdA2, tdT2)
        sendcmd("offset", offset)
        if trepeat:
            sendcmd("repeat", trepeat)
        else:
            sendcmd("once")   

            
    @with_doc(Stream.open)
    def open(self):
        self.committed = False
        super().open()

    @with_doc(Stream.close)
    def close(self):
        super().close()
        
    def commit(self):
        """Send all defined stimulus sequences to the device.

        In most cases, you do not need to use this method, as it is
        called automatically by `start()` and `run()` as needed.
        
        That said, committing takes several milliseconds, or more if
        long waves need to be transmitted. By calling `commit()`
        explicitly before calling `start()` or `run()`, you can
        control the timing of the start of the stimulation more
        precisely.

        If you make any changes to stimulation parameters, you will
        have to re-commit. The system understands that and will do it
        for you upon `start()` or `run()`. However, relying on this
        behavior is not recommended, if only because it negates any
        timing advantage of calling `commit()` ahead of time.

        """
        if not self.isopen:
            raise ValueError("Not open")
        if self.committed:
            return

        sampchans = []
        for chan in range(4):
            if chan in self.stimuli:
                if isinstance(self.stimuli[chan], Parametrized):
                    self._configstim(chan, self.stimuli[chan])
                elif isinstance(self.stimuli[chan], Sampled):
                    sampchans.append(chan)
                else:
                    raise ValueError("Confusion about stimulus")
            else:
                self.dev.command(f"off A{chan}")
        if not self.dev.verify(True):
            raise ValueError("Not verified")
        self.committed = True
        self.dev.commit(adata={c: self.stimuli[c] for c in sampchans},
                        maxahead=self.maxahead)

    @with_doc(Stream.start)
    def start(self):
        """Commits stimulation sequences if still needed."""
        if not self.committed:
            self.commit()
        super().start()

    def run(self):
        """Convenience function to run through an entire stimulus
        sequence.

        This starts the device running. Once stimulation is complete,
        concurrently recorded data may be retrieved using the
        `readall()` methods of AnalogIn and DigitalIn.

        This method may be called whether or not the output stream was
        opened and returns the stream to the original state at the end
        of the stimulation sequence. That means that the device is
        left running at the end of the sequence if the output stream
        was previously opened, and stopped if it was not.

        If using both AnalogOut and DigitalOut, calling `run()` on
        either has the same effect.

        Example::

            ao = AnalogOut(...)
            ao[0].stimulus(...)
            ...
            with AnalogIn(channels=..., rate=...) as ai:
                ao.run()
                # The device is now in "stopped" state.
                data = ai.alldata()

        """
        wasopen = self.isopen
        if not wasopen:
            self.open()
        self.start()
        while not _poll(self.dev):
            pass
        self.stop()
        if not wasopen:
            self.close()

    def poll(self) -> bool:
        """Send some data, if space available in buffer

        Returns:

            True if the last chunk has been sent

        You must call start and stop yourself.
        """
        return _poll(self.dev)
        
        
class DigitalOut(Stream):
    """Main interface for stimulating through digital lines.

    The rate may be specified in Hz or kHz. When using multiple
    streams, the rates must all be the same and only need to be
    specified on the first-opened stream.

    The `port` specifies which serial port to open. Use
    ``picodaq.devices()`` to retrieve the list of available
    ports.

    If you do not specify a port, the most recently opened device is
    used, or the first device on the system if none was opened before.

    The stimuli themselves are added by calling the `stimulus()`
    method on the individual lines, which may be accessed using
    indexing syntax, as in the following example::

        pulse1 = stimulus.TTL(40*ms)
        train1 = stimulus.Train(pulse1, 5, pulseperiod=100*ms)
        with DigitalOut(rate=30*kHz) as do:
            do[2].stimulus(train1)
            do.run()

    The current firmware imposes the restriction that digital lines
    used for output must be consecutive and their number may not be
    3. Lines are automatically added to the specified set as necessary
    to satisfy this constraint. If that creates a conflict with lines
    used for digital input, `commit()` or `start()` will raise an
    exception.

    """
    
    def __init__(self,
                 rate: Frequency | None = None,
                 port: str | None = None,
                 maxahead: int | Time | None = None):
        super().__init__(port, rate)
        self.stimuli = {}
        if isinstance(maxahead, Quantity):
            maxahead = (self.dev.rate * maxahead).plain()
        self.maxahead = maxahead # always store as samples
        self.committed = False

    def __getitem__(self, line: int):
        """Access to a single line.

        This provides a convenient syntax for specifying stimulation
        sequences.
        
        Example::

            pulse = TTL(10*ms)
            train = Train(pulse, pulsecount=5, pulseperiod=20*ms)
            with DigitalOut(rate=10*kHz) as do:
                do[1].stimulus(train)

        """
        return OutRef(self, line)

    @with_doc(OutRef.stimulus)
    def _stimulus(self,
                 line: int,
                 stim: Pulse | Train | Series,
                 delay: Time = 0*s,
                 repeat: Time | None = None,
                 offset: Voltage = 0*V):
        """The syntax `do[line].stimulus(stim, ...)` is generally
        preferred over `do._stimulus(line, stim, ...)`.

        """
        
        self.stimuli[line] = Parametrized(stim, delay, repeat, offset)
        self.committed = False

    @with_doc(AnalogOut._Ttosamples)
    def _Ttosamples(self, t: Time):
        return int((t * self.dev.rate).plain())

    def _confignostim(self, line):
        """Define a non-stimulus for given line.

        We achieve this through a zero-length TTL pulse that never gets
        sent because we define an empty train.
        """
        self.dev.command(f"ttl D{line} 0")
        self.dev.command(f"train D{line} 0")
        self.dev.command(f"offset D{line} 0")
        self.dev.command(f"once D{line}")

    def _configstim(self, line, stim):
        def sendcmd(cmd, *args):
            pfx = f"{cmd} D{line}"
            self.dev.command(pfx + "".join([f" {a}" for a in args]))
            
        name = stim.series.train.pulse.name
        if name != "ttl":
            self.dev.command(f"off D{line}")
            return
        
        activelow = stim.series.train.pulse.amplitude1
        T1 = self._Ttosamples(stim.series.train.pulse.duration1)
        npulse = stim.series.train.pulsecount
        pulseival = self._Ttosamples(stim.series.train.pulseperiod)
        pdT1 = self._Ttosamples(stim.series.train.perpulse.duration1)
        pdpival = self._Ttosamples(stim.series.train.perpulse.pulseperiod)

        delay = self._Ttosamples(stim.delay)
        trainival = self._Ttosamples(stim.series.trainperiod)
        ntrain = stim.series.traincount

        tdT1 = self._Ttosamples(stim.series.pertrain.duration1)
        tdpival = self._Ttosamples(stim.series.train.perpulse.pulseperiod)
        tdn = stim.series.pertrain.pulsecount
        tdtival = self._Ttosamples(stim.series.train.perpulse.trainperiod)

        trepeat = self._Ttosamples(stim.repeat) if stim.repeat else None

        sendcmd("ttl", T1)
        sendcmd("train", npulse, pulseival)
        sendcmd("perpulse", pdpival, 0, pdT1, 0, 0)
        sendcmd("series", delay, ntrain, trainival, tdtival, tdn)
        sendcmd("pertrain", tdpival, 0, tdT1, 0, 0)
        if activelow:
            sendcmd("offset", 1)
        else:
            sendcmd("offset", 0)
        if trepeat:
            sendcmd("repeat", trepeat)
        else:
            sendcmd("once")
        
    
    def commit(self):
        """Send all defined stimulus sequences to the device.

        In most cases, you do not need to use this method, as it is
        called automatically by `start()` and `run()` as needed.
        
        That said, committing takes several milliseconds. By calling
        `commit()` explicitly before calling `start()` or `run()`, you
        can control the timing of the start of the stimulation more
        precisely.

        """
        if not self.isopen:
            raise ValueError("Not open")
        if self.committed:
            return

        firstline = min(self.stimuli.keys())
        lastline = max(self.stimuli.keys())
        if lastline + 1 - firstline == 3:
            # we cannot have three lines total, so use all four
            firstline = 0
            lastline = 3
        samplines = []
        for line in range(4):
            if line in self.stimuli:
                if isinstance(self.stimuli[line], Parametrized):
                    self._configstim(line, self.stimuli[line])
                elif isinstance(self.stimuli[line], Sampled):
                    samplines.append(line)
                else:
                    raise ValueError("Confusion about stimulus")
            elif line >= firstline and line <= lastline:
                self._confignostim(line)
            else:
                self.dev.command(f"off D{line}")
        if not self.dev.verify(True):
            raise ValueError("Not verified")
        self.committed = True
        self.dev.commit(ddata={c: self.stimuli[c] for c in samplines},
                        maxahead=self.maxahead)

    @with_doc(Stream.open)
    def open(self):
        self.committed = False
        super().open()

    @with_doc(Stream.close)
    def close(self):
        super().close()
        
    @with_doc(Stream.start)
    def start(self):
        """Commits stimulation sequences if still needed."""
        if not self.committed:
            self.commit()
        super().start()

    @with_doc(AnalogOut.run)
    def run(self):
        wasopen = self.isopen
        if not wasopen:
            self.open()
        self.start()
        while not _poll(self.dev):
            pass
        self.stop()
        if not wasopen:
            self.close()
            
