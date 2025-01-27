from __future__ import annotations
import numpy as np
import logging
from numpy.typing import ArrayLike

from .device import PicoDAQ
from .stream import Stream
from .units import s, ms, Frequency, Time
from .decorators import with_doc

debug = False

log = logging.getLogger(__name__)


class AnalogIn(Stream):
    """Main interface for acquiring analog data

    Parameters:

        channel: A single channel to record from
        channels: A list of channels to record from
        rate: Sampling frequency for the recording
        port: Device to connect to

    You must specify either a single `channel` or a list of `channels`
    to record from, but not both. Any combination of analog inputs 0,
    1, 2, and 3 may be used. The order in which you specify the
    channels in the constructor becomes the order in which they will
    appear in the output from `read()`.

    The `rate` may be specified in Hz or kHz. When using multiple
    streams, the rates must all be the same and only need to be
    specified on the first-opened stream.

    The `port` specifies which serial port to open. Use
    ``picodaq.devices()`` to retrieve the list of available
    ports.

    If you do not specify a port, the most recently opened device is
    used, or the first device on the system if none was opened before.

    Example::

        with AnalogIn(channel=2, rate=30*kHz) as ai:
            data = ai.read(10*s)

    This reads 10 seconds worth of data from channel AI2 at 30 kS/s.

    """
    
    def __init__(self, channel: int | None = None,
                 channels: ArrayLike | None = None,
                 rate: Frequency = None,
                 port: str | None = None):
        super().__init__(port, rate)

        if channel is None:
            if channels is None:
                raise ValueError("Must specify either channel or channels")
            else:
                self.asvector = False
        else:
            if channels is None:
                channels = [channel]
                self.asvector = True
            else:
                raise ValueError("May not specify both channel and channels")
        for c in channels:
            if c!=int(c) or c<0 or c>3:
                raise ValueError("Unsupported channel")
        self.channels = channels
        self.partial = {}
        self.partialcount = None

    @with_doc(Stream.open)
    def open(self):
        self.dev.setaichannels(self.channels)
        super().open()
        
    @with_doc(Stream.close)
    def close(self):
        super().close()
        self.dev.setaichannels([])

    def verify(self, force=False):
        """Confirm whether recording parameters are OK

        Parameters:

            force: Re-verify unconditionally

        You typically don't have to call this, as `start()` and
        `read()` call it for you if you don't. To reduce the latency
        between when you first call `read()` and when the first sample
        is acquired, you can call this ahead of time, but the
        difference is unlikely to be more than a millisecond.

        The device keeps track of parameter changes since last call to
        verify and returns without doing work if there have been
        none. The `force` parameter causes unconditional re-verification.

        """
        self.dev.verify(force)

    def readchunk(self, maxn=None) -> np.ndarray:
        """Read a single chunk

        The shape of the result depends on whether the `channel` or
        `channels` parameter was used at construction time. If
        `channel` was used, the result is a T-vector, where T is the
        number of samples read. Otherwise, the result is a T×C array,
        even if only one channel is in use.

        Typically, `read()` is more convenient.

        """
        if not self.dev.reader:
            self.start()
        if not self.dev.reader.hasadata():
            self.dev.reader.read()
        if not self.dev.reader.hasadata():
            return None
        data = self.dev.reader.fetchadata(maxn)
        if debug:
            log.debug(f"readchunk {data.shape}")
        if self.asvector:
            return data[:,0]
        else:
            return data

    @with_doc(Stream.read)
    def read(self, amount: Time | int | None = None,
             raw: bool = False,
             times: bool = False) -> np.ndarray:
        """The shape of the result depends on whether the `channel` or
        `channels` parameter was used at construction time. If
        `channel` was used, the result is a T-vector, where T is the
        number of samples read. Otherwise, the result is a T×C array,
        even if only one channel is in use.

        If `raw` is true, signed 16-bit values from the DAC are
        returned.  Otherwise, readings are converted to volts and
        returned as 32-bit floats.

        """
        if times:
            data, times1 = super().read(amount, times=True)
        else:
            data = super().read(amount)
        if not raw:
            data = data.astype(np.float32) * 10 / 32768
        if times:
            return data, times1
        else:
            return data

    def readall(self, raw: bool = False,
             times: bool = False) -> np.ndarray:
        """Read all data accumulated during run().

        Used after calling run() on AnalogOut or DigitalOut to
        retrieve all the data recorded during the run. In continuous
        mode, returns a T-vector or T×C array. In episodic mode,
        returns an N×L or N×L×C array, where N is the number of
        episodes and L is the number of scans per episode.

        The meaning of `raw` and `times` is as for `read()`.

        """
        if not self.dev.reader:
            self.start()
        data = []
        times1 = []
        while self.dev.reader.hasadata():
            if times:
                dat, tms = self.read(raw=raw, times=True)
                data.append(dat)
                times1.append(tms)
            else:
                data.append(self.read(raw=raw))
                
        if not data:
            if times:
                return np.array([]), np.array([])
            else:
                return np.array([])
        
        M = self.dev.params.get('nchunks', 0)
        if M:
            data = np.stack(data, 0)
            if times:
                times1 = np.stack(times1, 0)
        else:
            data = np.concatenate(data, 0)
            if times:
                times1 = times1[0]
        if times:
            return data, times1
        else:
            return data
            
        
class DigitalIn(Stream):
    """Main interface for acquiring digital data

    You must specify either a single `line` or a list
    of `lines` to record from. There are some restrictions on
    the selection. The lines must be consecutive and specified in
    order. The total number of lines must be 1, 2, or 4 (i.e., not 3).

    The `rate` may be specified in Hz or kHz. When using multiple
    streams, the rates must all be the same and only need to be
    specified on the first-opened stream.

    The `port` specifies which serial port to open. The
    `picodaq.devicelist()` function retrieves the list of available
    ports.

    If you do not specify a port, the most recently opened device
    is used, or the first device on the system if none was opened before.

    """

    def __init__(self, line: int | None = None,
                 lines: ArrayLike | None = None,
                 rate: Frequency | None = None,
                 port: str | None = None):
        super().__init__(port, rate)
 
        if line is None:
            if lines is None:
                raise ValueError("Must specify either line or lines")
            else:
                self.asvector = False
        else:
            if lines is None:
                lines = [line]
                self.asvector = True
            else:
                raise ValueError("May not specify both line and lines")
        line0 = None
        if len(lines)==3:
            raise ValueError("Unsupported line combination")
        for line in lines:
            if line!=int(line) or line<0 or line>3:
                raise ValueError("Unsupported line")
            if line0 is not None and line != line0+1:
                raise ValueError("Unsupported line combination")
            line0 = line
        self.lines = lines
        self.partial = {}
        self.partialcount = None
        if len(lines):
            self.scanspersample = 8 // len(lines)
        
    @with_doc(Stream.open)
    def open(self):
        self.dev.setdilines(self.lines)
        super().open()

    @with_doc(AnalogIn.verify)
    def verify(self, force=False):
        self.dev.verify(force)

    def readchunk(self, maxn=None) -> np.ndarray:
        """Read a single chunk of data

        The output is always a vector of bytes comprising interleaved
        data.
        
        Typically, `read()` is more convenient.

        """
        if not self.dev.reader:
            self.start()
        if not self.dev.reader.hasddata():
            self.dev.reader.read()
        if not self.dev.reader.hasddata():
            return None
        data = self.dev.reader.fetchddata(maxn)
        return data

    @with_doc(Stream.read)
    def read(self, amount: Time | int | None = None,
             raw: bool = False, times: bool = False) -> np.ndarray:
        """If `raw` is given, the result is a vector of bytes with the
        lines interleaved. Otherwise, the result is a T-vector or
        T×C array of zeros and ones, one value per sample.

        Digital data must always be read in multiples of 8 scans
        divided by the number of lines. Requested amounts are rounded
        down to meet this criterion.

        """
        if times:
            data, times1 = super().read(amount, times=True)
        else:
            data = super().read(amount)
        if not raw:
            L = len(data)
            C = len(self.lines)
            if C:
                NSCANS = L * self.scanspersample
                data = np.unpackbits(data, bitorder='little').reshape(NSCANS, C)
                if self.asvector:
                    data = data[:,0]
            else:
                data = np.zeros((L,0), dtype=np.uint8)
        if times:
            return data, times1
        else:
            return data


    def readall(self, raw: bool = False, times: bool = False) -> np.ndarray:
        """Read all data accumulated during run().

        Used after calling run() on AnalogOut or DigitalOut to
        retrieve all the data recorded during the run. In continuous
        mode, returns a T-vector or T×C array. In episodic mode,
        returns an N×L or N×L×C array, where N is the number of
        episodes and L is the number of scans per episode.

        The meaning of `raw` is as for `read()`.

        """
        if not self.dev.reader:
            self.start()
        data = []
        times1 = []
        while self.dev.reader.hasddata():
            if times:
                dat, tms = self.read(raw=raw, times=True)
                data.append(dat)
                times1.append(tms)
            else:
                data.append(self.read(raw=raw))

        if not data:
            if times:
                return np.array([]), np.array([])
            else:
                return np.array([])

        M = self.dev.params.get('nchunks', 0)
        if M:
            data = np.stack(data, 0)
            if times:
                times1 = np.stack(times1, 0)
        else:
             data = np.concatenate(data, 0)
             if times:
                times1 = times1[0]

        if times:
            return data, times1
        else:
            return data
                

    @with_doc(Stream.close)
    def close(self):
        super().close()
        self.dev.setdilines([])
