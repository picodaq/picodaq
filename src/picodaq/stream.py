from __future__ import annotations
import numpy as np
import logging

from .device import PicoDAQ
from .units import Hz, kHz, Time, Frequency, Quantity
from .decorators import with_doc

MINRATE = 100 * Hz
MAXRATE = 330 * kHz

log = logging.getLogger(__name__)

class Stream:
    """Parent class for AnalogIn and friends

    This is a low-level class not intended for typical users.
    """
    def __init__(self, port: str | None = None,
                 rate: Frequency = None):
        self.dev = PicoDAQ.finddevice(port)
        if rate is None:
            if self.dev.rate is None:
                raise ValueError("Must specify sample rate")
        else:
            if rate < MINRATE or rate > MAXRATE:
                raise ValueError("Sample rate out of supported range")
            if self.dev.rate is not None:
                if rate != self.dev.rate:
                    raise ValueError("Cannot have different sample rates")
            self.dev.rate = rate
        self.isopen = False
        self.isstarted = False
        self.scanspersample = 1 # This is >1 for digital

    @with_doc(PicoDAQ.episodic)
    def episodic(self, duration: Time,
                 period: Time | None = None,
                 count: int | None = None):
        self.dev.episodic(duration, period, count)

    @with_doc(PicoDAQ.continuous)
    def continuous(self):
        self.dev.continuous()

    @with_doc(PicoDAQ.trigger)
    def trigger(self, source: int, polarity: int):
        self.dev.trigger(source, polarity)

    @with_doc(PicoDAQ.immediate)
    def immediate(self):
        self.dev.immediate()
        
    def __enter__(self):
        """Support `with ... as` syntax

        This is equivalent to `open() ... close()` but ensures that
        `close()` gets called even when errors occur.
        """
        self.open()
        return self

    def open(self):
        """Open the stream

        This automatically opens the underlying PicoDAQ device. If
        more than one stream is associated with a single device, each
        stream must be individually opened and the device remains
        open as long as any streams are open.
        
        Often, ``with ... as`` syntax is more convenient than calling
        `open()` yourself. If you do call `open()` yourself, you must
        match it with `close()`.
        """
        if self.isopen:
            raise ValueError("Already open")
        self.dev.open(self)
        self.isopen = True
        self.offset = 0

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the stream

        The underlying device is automatically closed once the last
        stream is closed.

        """
        if not self.isopen:
            raise ValueError("Not open")
        self.isopen = False
        try:
            self.stop()
        finally:
            self.dev.close(self)        

    def start(self):
        """Start data acquisition

        You typically do not have to call this directly, as `read()`
        calls it for you. If more than one stream is associated with a
        single PicoDAQ device, calling `start()` on one stream
        suffices to start all of them.

        The streams must be open before being started.
        """
        if self.isstarted:
            return
        if not self.isopen:
            raise ValueError("Not open")
        self.isstarted = True
        self.dev.start()

    def stop(self):
        """Stop data acquisition
        
        You typically do not have to call this directly, as `close()`
        (or the end of a `with ... as` block) calls it for you.

        If more than one stream is associated with a single PicoDAQ
        device, calling `stop()` on one stream suffices to stop all of
        them.
        """
        if not self.isstarted:
            return
        self.isstarted = False
        self.dev.stop()

    def readchunk(self):
        raise ValueError("Stream does not support reading")

    def chunkscans(self, maxn=None):
        """Quantum of data transfer

        This returns the number of scans that would be returned by the
        low-level `readchunk()` method. The high-level `read()` method
        is most efficient if requested quantities are an integer
        multiple of this number.

        This number is only available when the stream is open.

        """

        if not self.isopen:
            raise ValueError("Not open")
        return self.dev.nscans

        
    def read(self,
             amount: Time | int | None = None,
             raw: bool = False,
             times: bool = False) -> np.ndarray:
        """High-level method for reading data

        Parameters
        ----------
        amount
            Amount of data to be read, either in units of time,
            or as an integer number of samples.
        raw
            Whether to return raw data from the device or convert
            them to more convenient units.
        times
            Whether to return a vector of time stamps

        Returns
        -------
        data
            A numpy array containing the data, either a vector
            or a T×C array.
        times:
            A corresponding vector of time stamps, in seconds
            since start of run; only if the `times` flag is set
            in the function call.

        Notes
        -----
        If no `amount` is specified at all, a single chunk is read in
        continuous mode, or a full episode in episodic mode.

        """
        if not self.isopen:
            raise ValueError("Not open")
        if not self.dev.reader:
            self.start()
        if isinstance(amount, Quantity):
            amount = int((amount * self.dev.rate).plain())
        elif amount is None:
            amount = self.dev.nscans
            epichunks = self.dev.params.get('nchunks', 0)
            if epichunks:
                amount *= epichunks
        data = []
        got = 0
        while got < amount:
            dat = self.readchunk(amount - got)
            if dat is None or len(dat)==0:
                break
            data.append(dat)
            got += len(dat) * self.scanspersample
        if data:
            data = np.concatenate(data, 0)
        else:
            if times:
                return np.array([]), np.array([])
            else:
                return np.array([])
        if times:
            times = self.offset + np.arange(len(data) * self.scanspersample,
                                            dtype=np.float32)
            if epichunks:
                times %= epichunks * self.dev.nscans
            times /= self.dev.rate.as_("Hz")
            self.offset += len(data) * self.scanspersample
            return data, times
        else:
            self.offset += len(data) * self.scanspersample
            return data

    def commit(self):
        pass
