import numpy as np
import time
from typing import Dict, Tuple
import logging

from .utils import checksum

FLAGS_BINARY = np.uint8(0x80)
FLAGS_LAST = np.uint8(0x04)

log = logging.getLogger(__name__)
debug = False
if debug:
    log.setLevel(logging.DEBUG)

class BinaryWriter:
    def __init__(self, dev: "PicoDAQ", nscans: int,
                 asources: Dict[int, "Sampled"],
                 dsources: Dict[int, "Sampled"]):
        self.dev = dev
        self.nscans = nscans
        self.adata = {}
        self.agen = {}
        self.scale = {}
        self.offset = {}
        self.raw = {}
        self.channels = []
        self.ddata = {}
        self.dgen = {}
        self.lines = []
        for c, src in asources.items():
            if debug:
                log.debug(f"binwriter asrc {c} {src} {src.data}")
            if callable(src.data):
                self.agen[c] = src.data() # yields an iterable
                self.adata[c] = next(self.agen[c])
                log.info(f"binwr {c} {np.mean(self.adata[c]):.3f} {np.std(self.adata[c]):.3f}")
            else:
                self.adata[c] = src.data
            self.scale[c] = src.scale
            self.offset[c] = src.offset
            self.raw[c] = src.raw
            self.channels.append(c)
        self.channels.sort()
        for c, src in dsources.items():
            if callable(src.data):
                self.dgen[c] = src.data() # yields an iterable
                self.ddata[c] = next(self.dgen[c])
            else:
                self.ddata[c] = src.data
            self.lines.append(c)
        self.lines.sort()

        wordsperchunk = 1 + len(self.channels) * self.nscans // 2
        wordsperchunk += len(self.lines) * self.nscans // 32
        self.blocksperchunk = wordsperchunk // 16
        if self.blocksperchunk * 16 < wordsperchunk:
            self.blocksperchunk += 1

        dev.command(f"outnscans {nscans}")
        chans = " ".join([f"A{c}" for c in asources])
        chans += " ".join([f"D{c}" for c in dsources])
        dev.command(f"sampled {chans}")

        self.chunkno = 0
        self.finished = False

    @property
    def productionfinished(self):
        """Has the production finished?

        This property reads as true once all the data for all channels
        have been sent to the device.

        Critically, this does not guarantee that all data have been
        sent out from the device into the world. For that, check the
        flags() from the binary reader.

        It is OK to call sendchunk data after productionfinished
        becomes True.

        """
        return self.finished


    def _filladata(self, data: np.ndarray) -> Tuple[int, int]:
        def _toraw(c, yy):
            if debug:
                log.debug(f"filla {np.std(yy)}")
            if self.raw[c]:
                return yy
            else:
                return ((yy * self.scale[c].as_('V')
                         + self.offset[c].as_('V'))
                        * 32767 / 10).astype(np.int16)

        fin = 0
        i0 = 2 # halfword offset into output data; skip header
        for k, c in enumerate(self.channels):
            n0 = 0 # scans written so far for this channel
            while n0 < self.nscans:
                M = self.nscans - n0
                if debug:
                    log.debug(f"binwr {k} {c} {n0} {M}")
                if self.adata[c] is None:
                    data[i0:i0+M] = _toraw(c, np.array(0))
                    fin += 1
                    i0 += M
                    n0 += M
                    break
                else:
                    M = min(M, len(self.adata[c]))
                    data[i0:i0+M] = _toraw(c, self.adata[c][:M])
                    n0 += M
                    i0 += M
                    if M < len(self.adata[c]):
                        self.adata[c] = self.adata[c][M:]
                    elif c in self.agen:
                        self.adata[c] = next(self.agen[c])
                        log.debug(f"next {c} {np.mean(self.adata[c]):.3f} {np.std(self.adata[c]):.3f} {self.adata[c].shape}")
                    else:
                        self.adata[c] = None
        return fin, i0

    def _fillddata(self, data: np.ndarray, i0: int) -> int:
        if not self.lines:
            return 0
        ddat = np.zeros((self.nscans, len(self.lines)), np.uint8)
        fin = 0
        for k, c in enumerate(self.lines):
            n0 = 0
            while n0 < self.nscans:
                M = self.nscans - n0                
                if self.ddata[c] is None:
                    # ddat[n0:n0+M, k] = 0
                    fin += 1
                    n0 += M
                    break
                else:
                    M = min(M, len(self.ddata[c]))
                    ddat[n0:n0+M,k] = _toraw(c, self.ddata[c][:M])
                    n0 += M
                    if M < len(self.ddata[c]):
                        self.ddata[c] = self.ddata[c][M:]
                    elif c in self.dgen:
                        self.ddata[c] = next(self.dgen[c])
                    else:
                        self.ddata[c] = None
        ddat = np.packbits(ddat)
        N = len(ddat)
        data[i0:i0+N//2] = np.frombuffer(ddat.tobytes(), np.int16)
        return fin
                
    
    def sendchunk(self, pre: bool):
        """Send a single chunk of data to the device

        Parameter

            pre - True if prebuffering

        Returns

            True if more data remain to be sent

        If `pre` is true, sending of the data is preceded by an
        "outdata" command and the returned checksum is checked. Use
        this before start()ing the acquisition. Once the device is in
        binary mode, use pre=False to just send the data.

        """
        if self.finished:
            return False
        data = np.empty(self.blocksperchunk*32, np.int16)
        data[0] = FLAGS_BINARY
        data[1] = self.chunkno
        C = len(self.channels)
        afin, i0 = self._filladata(data)
        L = len(self.lines)
        dfin = self._fillddata(data, i0)

        if afin == C and dfin == L:
            data[0] |= FLAGS_LAST
            self.finished = True

        if debug:
            self.dump(data)
            
        if pre:
            cmd = f"outdata {self.chunkno} {self.blocksperchunk}"
            self.dev.command(cmd, feedback=False)
        log.info(f"(outdata) {data.shape} {data[0]} {data[1]} {np.mean(data[2:49]):.3f} {np.std(data[2:49]):.3f} {np.mean(data[50:]):.3f} {np.std(data[50:]):.3f}")
        dat = data.tobytes()
        log.info(f"binwriter write {len(dat)}")
        self.dev.ser.write(dat)
        log.info("binwriter wrote")
        if pre:
            self.dev._getfeedback("outdata")
            chk = checksum(data)
            if f"{self.dev.params['outdata']}" != f"{chk}":
                log.error("checksum failed {chk} != {self.dev.params['outdata']}")
                raise ValueError("Checksum failed")

        self.chunkno += 1
        return not self.finished
    

    def dump(self, data, ashex=True):
        if len(data)==0:
            return
        s = []
        if ashex:
            for x in data:
                s.append(f"{x:d} ")
        else:
            for k, c in enumerate(data):
                if c>=32 and c<=126:
                    s.append(chr(c))
                elif c==10 and len(s)>2 \
                     and s[-1]>='0' and s[-1]<='z' \
                     and ((s[-2]>='0' and s[-2]<='z') or a[-2]==' '):
                    s.append('\n . ')
                elif c>=128:
                    s.append("'")
                else:
                    s.append('.')
        log.debug(f"<write> {''.join(s[:10])}...")
        
