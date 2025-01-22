import numpy as np
import time
import logging

from .errors import DeviceError

FLAGS_BINARY = np.uint8(0x80)
FLAGS_STIMACTIVE = np.uint8(0x01)
FLAGS_FIRSTINEPISIDE = np.uint8(0x02)
FLAGS_LASTINEPISIDE = np.uint8(0x04)
FLAGS_INBUFMASK = np.uint8(0x30)
FLAGS_INBUF_EMPTY = np.uint8(0x00)
FLAGS_INBUF_CONTINUE = np.uint8(0x10)
FLAGS_INBUF_NEARFULL = np.uint8(0x20)
FLAGS_INBUF_FULL = np.uint8(0x30)

log = logging.getLogger(__name__)
debug = True

log.setLevel(logging.DEBUG)



class BinaryReader:
    """Helper class for reading binary data.

    This is a low-level class not intended for typical users.
    """
    def __init__(self, dev: "PicoDAQ"):
        self.dev = dev
        self.setupaichannels()
        self.setupdilines()
        lines = dev._getfeedback("**BINARY")
        self.active = False
        if len(lines)==0 or "**BINARY" not in lines[-1]:
            raise DeviceError("No binary data found")
        self.blocksperchunk = int(lines[-1].split(" ")[-1])
        self.active = True
        self.nn = 0
        self.t0 = time.time()
        self._adata = []
        self._ddata = []
        self.lastflags = 0x85
        self.laststatus = 0
        self.lastchunkno = -1

    @property
    def flags(self):
        """Flags from latest read

        This property reports the "flags" byte from the most recently
        read chunk.

        """
        return self.lastflags

    @property
    def status(self):
        """Status from latest read

        This property reports the "status" byte from the most recently
        read chunk.

        """
        return self.laststatus

    def setupaichannels(self):
        dest_leader = []
        dest_follower = []
        channels = self.dev.aichannels
        if self.dev.aimask & 3:
            # leader has channels 0 and/or 1
            if 0 in channels:
                dest_leader.append(channels.index(0))
            if 1 in channels:
                dest_leader.append(channels.index(1))
            if 2 in channels:
                dest_follower.append(channels.index(2))
            if 3 in channels:
                dest_follower.append(channels.index(3))
        else:
            if 2 in channels:
                dest_leader.append(channels.index(2))
            if 3 in channels:
                dest_leader.append(channels.index(3))
        self.destleader = dest_leader
        self.destfollower = dest_follower
        self.nleader = len(dest_leader)
        self.nfollower = len(dest_follower)
        self.nchannels = self.nleader + self.nfollower

    def setupdilines(self):
        self.nlines = len(self.dev.dilines)

    def storeadata(self, data):
        if debug:
            log.debug(f"storeadata {data.shape} {data.dtype}")
        self._adata.append(data)
        
    def storeddata(self, data):
        if debug:
            log.debug(f"storeddata {data.shape} {data.dtype}")
        self._ddata.append(data)

    def hasadata(self):
        return len(self._adata) > 0

    def hasddata(self):
        return len(self._ddata) > 0

    def fetchadata(self, maxn=None):
        if maxn and maxn < len(self._adata[0]):
            res = self._adata[0][:maxn]
            self._adata[0] = self._adata[0][maxn:]
            return res
        else:
            return self._adata.pop(0)

    def fetchddata(self, maxn=None):
        if maxn:
            if self.nlines:
                maxb = maxn * self.nlines // 8
            else:
                maxb = maxn
            if maxb < len(self._ddata[0]):
                res = self._ddata[0][:maxb]
                self._ddata[0] = self._ddata[0][maxb:]
                return res
        return self._ddata.pop(0)

    def dump(self, data, ashex=True):
        if len(data)==0:
            return
        s = []
        if ashex:
            for k in range(0, len(data)-1, 2):
                s.append(f"{data[k+1]:02x}{data[k]:02x} ")
        else:
            for k, c in enumerate(data):
                if c>=32 and c<=126:
                    s.append(chr(c))
                elif c==10 and len(s)>2 \
                     and s[-1]>='0' and s[-1]<='z' \
                     and ((s[-2]>='0' and s[-2]<='z') or s[-2]==' '):
                    s.append('\n . ')
                elif c>=128:
                    s.append("'")
                else:
                    s.append('.')
        log.debug(f"<read< {''.join(s)}")

    def read(self):
        """
        Read a single chunk of raw data from device into buffer
        """
        if not self.active:
            #if self.dev.params["stop"] != "ok":
            #    raise DeviceError("Stopped with error")
            raise DeviceError("Not active")
        data = []
        n = 0
        while len(data) < self.blocksperchunk:
            if debug:
                dt = time.time() - self.t0
                log.debug(f"> {dt:.3f} read {self.nn} {n}")
            x = self.dev.ser.read(64)
            if debug:
                if n==0:
                    self.dump(x)
            if n==0 and x.startswith(b"**ASCII"):
                self.active = False
                if debug:
                    self.dump(x, False)
                self.dev._ungets(x)
                return self.dev._handlestop(True)
            elif len(x)==64:
                data.append(x)
                n += 1
            elif len(x)>0:
                for y in data:
                    self.dump(y)
                self.dump(x)
                raise DeviceError("Broken data block")
        self.nn += 1
        self.parsedata(data)

    def parsedata(self, chunk):
        if len(chunk) == 0:
            return
        raw = np.concatenate([np.frombuffer(blk, np.int16)
                              for blk in chunk])
        # Following is unsafe: flags can be missed as they are 
        # overwritten by next chunk
        self.lastflags = np.uint8(raw[0] & 255)
        self.laststatus = np.uint8(raw[0] >> 8)
        self.lastchunkno += 1
        if self.lastchunkno & 65535 != raw[1] & 65535:
            raise RuntimeError("Lost chunk")

        if debug:
            log.debug(f"<parse< flags {self.lastflags} status {self.laststatus} chunk={self.lastchunkno}")
        N = self.dev.nscans
        leadstart = 2
        follstart = leadstart + N*self.nleader
        digistart = follstart + N*self.nfollower
        adata = np.zeros((N, self.nchannels), np.int16)
        for k, d in enumerate(self.destleader):
            adata[:,d] = raw[leadstart+k:follstart:self.nleader]
        for k, d in enumerate(self.destfollower):
            adata[:,d] = raw[follstart+k:digistart:self.nfollower]
        self.storeadata(adata)
        if self.nlines:
            ddata = np.frombuffer(raw[digistart:].tobytes(), np.uint8)
            self.storeddata(ddata[:N*self.nlines//8])
        else:
            self.storeddata(np.zeros((N,0), np.uint8))
        
    def close(self):
        log.debug(f"binreader close {self.active}")
        if not self.active:
            return
        
        self.dev.command("stop", False)
        t0 = time.time()
        n = 0
        while time.time() - t0 < 2:
            x = self.dev.ser.read_until(b"**ASCII")
            if debug:
                self.dump(x)
            if b"**ASCII" in x:
                idx = x.index(b"**ASCII")
                n += idx
                self.active = False
                if n > 0:
                    log.debug(f"(Dropped {n} bytes at end of acq)")
                self.dev._ungets(x[idx:])
                return self.dev._handlestop(False)
            n += len(x)
        raise DeviceError("Failed to stop binary acquisition")
