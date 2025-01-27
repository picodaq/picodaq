import serial
import serial.tools.list_ports
import time
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Optional, List, Iterable
import logging

from .units import Hz, kHz, Time
from .utils import NScanCalc, makemask, checksum
from .binreader import BinaryReader
from .binwriter import BinaryWriter
from .errors import DeviceError

log = logging.getLogger(__name__)
debug = True#False # set true for more debug info
if debug:
    log.setLevel(logging.DEBUG)
t0 = time.time()
Serial = serial.Serial


def picos() -> Dict[str, str]:
    """Enumerate the Raspberry Pi Pico boards connected to the computer

    Returns

        A dictionary mapping ports to serial numbers
    """
    vidpid="2E8A:000A"
    devs = {}
    for p in serial.tools.list_ports.comports():
        info = p.usb_info().split()
        if info[0]=='USB':
            res = {}
            for x in info[1:]:
                k, v = x.split("=")
                res[k] = v
            if res['VID:PID'] == vidpid:
                devs[p.device] = res['SER']
    return devs


def _getfirstlines(ser):
    lines = []
    while True:
        line = ser.readline()
        if not line: # Timeout
            return lines
        try:
            lines.append(str(line, "utf8"))
        except UnicodeDecodeError:
            return []

        
def picodaqheader(port: str) -> str:
    """Test whether device is a picoDAQ

    Parameters

        port - Serial port name of device to test

    Returns

       The "+picodaq" line reported by the device 

    Raises exception if not a picodaq
    """

    ser = serial.Serial(port, timeout=1)

    ser.write(b"picodaq\n")
    while True:
        line = ser.readline()
        if b"+picodaq" in line:
            ser.close()
            txt = str(line, "utf8").strip()
            return txt[txt.index("+picodaq"):]
        elif not line: # Timeout
            break
    ser.close()
    raise DeviceError("Not a picoDAQ")
        
        
def isapicodaq(port: str) -> bool:
    """Test whether device is a picoDAQ

    Parameters

        port - Serial port name of device to test

    Returns

        True if the device is a picoDAQ
    """
    try:
        header = picodaqheader(port)
        return True
    except (serial.SerialException, ValueError):
        return False

    
def devices() -> Dict[str, str]:
    """Enumerate the PicoDAQs connected to the computer

    Returns

        A dictionary mapping serial port names to serial numbers

    See also `find`.
    """
    return {port: serno
            for port, serno in picos().items()
            if isapicodaq(port)}

def find(serno: str) -> str:
    """Find a PicoDAQ by serial number

    Parameters

        serno - Serial number of device to find

    Returns

        Serial port name of found device

    Raises exception if device is not found. See also `picodaqs`.
    """

    for port, serno1 in picos().items():
        if serno1 == serno:
            return port
    raise DeviceError(f"No PicoDAQ found with serial number {serno}")

class PicoDAQ:
    _opendevs: List["PicoDAQ"] = []

    @staticmethod
    def finddevice(port: str) -> "PicoDAQ":
        if port:
            # Find a specific device
            if port.startswith("ACM"):
                port = "/dev/tty" + port
            for dev in PicoDAQ._opendevs:
                if dev.port==port:
                    return dev
            if isapicodaq(port):
                return PicoDAQ(port)
            raise DeviceError(f"No picoDAQ found on port {port}")
        else:
            # Any device
            if PicoDAQ._opendevs:
                return PicoDAQ._opendevs[-1]
            for port in devices():
                return PicoDAQ(port)
            raise DeviceError("No picoDAQs found")

    
    def __init__(self, port: str | None = None):
        """Representation of USB connection to PicoDAQ device

        Parameters

            port - Serial port of device to connect

        If no port is given, connects to the first device identified
        by `devicelist` as a suitable candidate.

        You typically do not need to use this class directly. The
        AnalogIn, DigitalIn, AnalogOut, and DigitalOut classes can
        find the PicoDAQ by themselves.

        """
        self.ser = None
        self.openstreams = set()
        self._starting = False
        self._stopping = False
        self._committing = False

        self._reset()
        
        self._bytes = bytes()
        self.reader = None
        self.writer = None
        self.nscans = None # meaningfully set by open()
        self.maxahead = None

        if not port:
            port = devicelist[0][0]
        self.port = port
        for dev in PicoDAQ._opendevs:
            if port == dev.port:
                raise DeviceError(f"A connection already exists to {port}")
        
        self.ser = Serial(port, timeout=0.1, write_timeout=0.2)
        log.info(f"Connected to PicoDAQ at {port}")
        self.ser.close()

    def setaichannels(self, channels: Iterable[int]) -> None:
        self.aichannels = channels
        self.aimask = makemask(channels)

    def setdilines(self, lines: Iterable[int]) -> None:
        self.dilines = lines
        self.dimask = makemask(lines)

    def episodic(self, duration: Time,
                 period: Time | None = None,
                 count: int | None = None) -> None:
        """Select episodic recording mode.

        Duration and period may be specified in any unit of time.
        The period is measured start-to-start and is optional if
        triggering is enabled, in which cases it specifies the
        minimum period.

        Count, if given, specifies that the recording will stop
        automatically after that number of episodes have been
        recorded. Otherwise, it continuous until the user stops the
        recording.

        In episodic mode, the timing of any stimuli is modified such
        that there is one train per episode and the defined
        inter-train intervals are ignored. The `delay` parameter sets
        the time between start of episode and first pulse.

        See also `continuous`.

        """
        self.epi_dur = duration
        self.epi_per = period
        self.epi_count = count
        if self.isopen():
            self._postopen()
        return self

    def continuous(self) -> None:
        """Select continuous recording mode.

        This is the default. See also `episodic`.
        """
        self.epi_dur = None
        self.epi_per = None
        self.epi_count = None
        if self.isopen():
            self._postopen()
        return self

    def trigger(self, source: int, polarity: int) -> None:
        """Define triggering.

        The recording (whether continuous or episodic) is not actually
        started until the given trigger condition is met. `Source` (0,
        1, 2, or 3) specifies a digital channel. `Polarity` specifies
        whether the system triggers on rising edge (`polarity` > 0) or
        on falling edge (`polarity` < 0).

        See also `immediate`.
        """
        if source < 0 or source > 3:
            raise ValueError("Unsupported trigger source")
        if not polarity:
            raise ValueError("Unsupported trigger polarity")
        self.trg_source = source
        self.trg_polarity = polarity
        if self.isopen():
            self._postopen()
        return self

    def immediate(self) -> None:
        """Disable triggering.

        Recording commences immediately upon `start`. This is the
        default.

        See also `trigger`.

        """
        self.trg_source = None
        self.trg_polarity = 0
        if self.isopen():
            self._postopen()
        return self

    def start(self) -> None:
        """Start the acquisition.

        Both input and output are started in synchrony.  You typically
        do not have to call this explicitly, as reading from an
        AnalogIn or DigitalIn automatically starts the acquisition.
        """
        if self._starting:
            return
        
        if not self.isopen():
            raise DeviceError("Not open")
        if self.reader:
            if not self.reader.active:
                self.reader.close()
                self.reader = None
            else:
                return # already started

        try:
            self._starting = True # prevent recursion
            for stream in self.openstreams:
                if stream:
                    stream.start() # Propagate to all streams
        finally:
            self._starting = False

        self.command("start")
        if not self.verify():
            log.error("Unsupported parameters:")
            for k, v in self.params:
                log.error(f"  {k}: {v}")
            raise DeviceError("Unsupported parameters")
        self.reader = BinaryReader(self)
        self.nscans = self.params["nscans"]
        log.debug("params = ", self.params)

    def stop(self) -> None:
        """Stop the acquisition.

        You typically do not have to call this explicitly, as closing
        AnalogIn or DigitalIn does it automatically.
        """
        if self._stopping:
            return
        try:
            self._stopping = True # prevent recursion
            for stream in self.openstreams:
                if stream:
                    stream.stop() # Propagate to all streams
            self._stopping = False
        except:
            self._stopping = False
            raise

        if self.reader:
            self.reader.close()
            self.command("nop")
        self.writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self, stream: "Stream" = None) -> None:
        """Open the device.

        You typically do not have to call this directly. If you do,
        make sure that your calls to open() and to close() are
        matched.
        """
        if len(self.openstreams) == 0:
            self.ser.open()
            PicoDAQ._opendevs.append(self)
        self.openstreams.add(stream)
        if self.rate is None:
            # probably being opened for test purposes
            self.nscans = None
        else:
            self._postopen()
        

    def _postopen(self):        
        rate_Hz = int(self.rate.as_(Hz))
        self.command(f"rate {rate_Hz}")
        self.command(f"aimask {self.aimask}")
        self.command(f"dimask {self.dimask}")
        
        if self.trg_source is None:
            self.command("immediate")
        else:
            self.command(f"trigger {self.trg_source} {self.trg_polarity}")
            
        calc = NScanCalc(self.aimask, self.dimask)
        if self.epi_dur is None:
            self.command("nchunks 0")
            nscans = calc.bestforcont()
            self.command(f"nscans {nscans}")
            self.nscans = self.params["nscans"] # get updated value from device

        else:
            scansperepi = int(np.ceil((self.epi_dur * self.rate).plain()))
            nscans = calc.bestforepi(scansperepi)
            self.command(f"nscans {nscans}")
            self.nscans = self.params["nscans"] # get updated value from device
            nchunks = (scansperepi + self.nscans - 1) // nscans
            self.command(f"nchunks {nchunks}")
            self.command(f"period {int(self.epi_per.as_('ms') + 0.5)}")
            if self.epi_count is None:
                self.command(f"nepis 0")
            else:
                self.command(f"nepis {self.epi_count}")

        if "verify" in self.params:
            del self.params["verify"]


    def commit(self, adata=None, ddata=None, maxahead=None):
        if not self.isopen():
            raise DeviceError("Not open")
        if maxahead:
            self.maxahead = maxahead
        if self._committing:
            """This state variable allows one stream to start the commit
            process with its data, while the other stream is called from
            within this function and provides its data recursively."""
            if adata is not None:
                self._adata = adata
            elif ddata is not None:
                self._ddata = ddata
            return
        """Code below here is not reached in recursive call"""
        self._adata = adata if adata else {}
        self._ddata = ddata if ddata else {}
        try:
            self._committing = True # prevent recursion
            for stream in self.openstreams:
                stream.commit() # Propagate to all streams
        finally:
            self._committing = False

        self._setupsampled(self._adata, self._ddata)
            
    def _setupsampled(self, adata: Dict[int, "Sampled"],
                      ddata: Dict[int, "Sampled"]) -> None:
        """Create a BinaryWriter and prefill buffer

        Calculates optimal number of scans per chunk. This needs
        to know whether the acquisition is going to be episodic
        or continuous.

        Sends the amount of data recommended by the response from
        the most recent "sampled" command.

        Users don't call this. It is for AnalogOut to call from
        within its commit() system.

        """
        if not adata and not ddata:
            return
        calc = NScanCalc(makemask(adata.keys()), makemask(ddata.keys()))
        if self.epi_dur is None:
            nscans = calc.bestforcont()
        else:
            scansperepi = int(np.ceil((self.epi_dur * self.rate).plain()))
            nscans = calc.bestforepi(scansperepi)
        self.writer = BinaryWriter(self, nscans, adata, ddata)
        nchunks = self.params["sampled"]
        if self.maxahead:
            nchunks = max(2, min(nchunks, self.maxahead // nscans))
        self.aheadchunks = nchunks
        if debug:
            log.debug(f"nchunks is {nchunks}")
        for k in range(nchunks):
            if debug:
                log.debug(f"k is {k}")
            self.writer.sendchunk(True)
            if self.writer.productionfinished:
                break
        
                
    def close(self, stream: "Stream" = None) -> None:
        """Stop and close the device.

        The device is stopped immediately, but only actually closed if
        no open streams remain. At that point, all parameters are
        reset.
        
        You typically do not have to call this directly, as the
        streams call it internally.

        """
        try:
            self.stop()
        finally:
            if stream in self.openstreams:
                self.openstreams.remove(stream)
                if len(self.openstreams) == 0:
                    if self.reader:
                        del self.reader
                        self.reader = None 
                    PicoDAQ._opendevs.remove(self)
                    self.ser.close()
                    self._reset()
            
    def isopen(self) -> bool:
        """True if the device is currently open

        Returns true whether the device has been opened directly by
        the user or indirectly through AnalogIn and friends.
        """
        return len(self.openstreams) > 0
    
    def verify(self, force: bool = False) -> bool:
        """Confirm whether recording and stimulation parameters are OK

        Parameters:

            force - Re-verify unconditionally

        You typically don't have to call this directly, as streams
        check for you.

        The device keeps track of parameter changes since last call to
        verify and returns without doing work if there have been
        none. The `force` parameter causes unconditional
        re-verification.

        """
        if force or "verify" not in self.params:
            self.command("verify")
        return self.params.get("verify", "") == "ok"

    def _reset(self):
        self.params = {}
        self.rate = None
        self.aimask = 0
        self.aichannels = []
        self.dimask = 0
        self.dilines = []
        self.epi_dur = None
        self.epi_per = None
        self.epi_count = None
        self.trg_source = None
        self.trg_polarity = 0

    def __del__(self):
        if self.ser and self.ser.is_open:
            log.warning(f"device going out of scope while open {self}")
            self.ser.close()

    def _ungets(self, bts: bytes):
        self._bytes += bts

    def _readlinesfrombytes(self):
        if b"\n" not in self._bytes:
            return []
        bits = self._bytes.split(b"\n")
        self._bytes = bits[-1]
        lines = []
        for bit in bits[:-1]:
            try:
                a = str(bit, "utf8")
            except UnicodeDecodeError:
                log.error(f"Unicode failure {a}")
                a = ""
            lines.append(a)
        return lines


    def _handlestop(self, unexpected: bool):
        """To be called only by binreader when **ASCII received

        Argument:

            unexpected - Set true if **ASCII was unexpected

        "Unexpected" stops may be because of error or because of the
        end of episodic acquisition.

        Collects and prints any "!" error lines before stop.

        """

        log.info("Device stopped")
        self.params["stop"] = "??"
        fb = self._getfeedback("+stop")
        reasons = []
        for line in fb:
            if line.startswith("!"):
                reasons.append(line[1:])
        for reason in reasons:
            log.error(f"Reason: {reason}")
        if self.params["stop"] == "ok":
            return
        if reasons:
            raise DeviceError(f"Stopped with error: {reasons[0]}")
        raise DeviceError("Stopped with unknown error")
    
    def _getfeedback(self, until=None) -> List[str]:
        """Collect and return feedback from PicoDAQ device.
        
        If UNTIL is given and not null, returns when that string
        appears in a feedback line. Otherwise, continues to collect
        until a gap in output of 100 ms occurs.
        
        Result is a (possibly empty) list of lines.
        """

        lines = []
        while True:
            a = self.ser.readline()
            if not a: # Timeout
                if lines:
                    return lines
                else:
                    return self._readlinesfrombytes()
            if self._bytes:
                a = self._bytes + a
                self._bytes = bytes()
            try:
                a = str(a, "utf8")
            except UnicodeDecodeError:
                log.error(f"Unicode failure {a}")
                a = ""
            a = a[:-1]
            for line in a.split("\n"):
                log.debug(f"{time.time() - t0:.3f} << {line}")
                if line[:1] == '+':
                    kv = a[1:].split(" ")
                    if len(kv) >= 2:
                        k, v = kv[:2]
                        try:
                            self.params[k] = int(v)
                        except ValueError:
                            self.params[k] = v
                lines.append(line) # drop the newline
            if until and until in a: # Key found
                return lines

    def deviceinfo(self) -> Dict[str, str]:
        wasopen = self.isopen()
        if not wasopen:
            self.ser.open()
        try:
            self.command("picodaq")
            self.command("info")
            pd = self.params["picodaq"]
            info = self.params["info"]
        finally:
            if not wasopen:
                self.ser.close()
        dct = {"picodaq": pd}
        for kv in info.split(","):
            k, v = kv.split("=", 1)
            dct[k] = v
        return dct         
            
    def command(self, cmd, feedback=True) -> List[str]:
        """Send a command and optionally collect feedback.
        
        The given command is sent to the PicoDAQ.

        If `feedback` is True, waits for and returns feedback.
        Otherwise returns immediately and returns None.

        This is a low-level call not intended for typical users.
        """
        log.debug(f"{time.time() - t0:.3f} >> {cmd}")
        self.ser.write(bytes(cmd + "\n", "utf8"))
        if feedback:
            key = cmd.split(" ")[0]
            self.params[key] = None
            return self._getfeedback("+" + key)
        
    def sendwave(self, idx: int, wav: np.ndarray):
        """Send wave data"""
        if wav.dtype != np.int16:
            raise DeviceError("Wave must be int16")
        chk = checksum(wav)
        N = len(wav)
        self.command(f"wave {idx} {N}", feedback=False)
        self.ser.write(wav.tobytes())
        self._getfeedback("+wave")
        if f"{self.params['wave']}" != f"{chk}":
            # (string comparison to catch "??" response)
            log.error(f"checksum {chk} != {self.params['wave']}")
            raise DeviceError("Checksum failed")

    def isrunning(self):
        """Reports whether the device has been started"""
        if self.reader:
            return self.reader.active
        else:
            return False

        
