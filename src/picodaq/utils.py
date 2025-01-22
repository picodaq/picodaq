import numpy as np
from typing import Iterable
import warnings
import logging

log = logging.getLogger(__name__)


def makemask(channels: Iterable[int]) -> int:
    mask = 0
    for c in channels:
        mask |= (1<<c)
    return mask


def countmask(msk: int) -> int:
    n = 0
    for k in range(4):
        if msk & (1<<k):
            n += 1
    return n


def stepsize(aimask: int, dimask: int) -> int:
    nd = countmask(dimask)
    na1 = countmask(aimask & 3)
    na2 = countmask(aimask & 12)
    if nd & 1:
        return 32
    elif nd & 2:
        return 16
    elif nd:
        return 8
    elif na1==1 or na2==1:
        return 2
    else:
        return 1


def roundup(number: int, modulo: int):
    return modulo * ((number + modulo - 1) // modulo)


def checksum(wav: np.ndarray) -> np.uint32:
    chk = np.uint32(0)
    with warnings.catch_warnings(action="ignore",
                                 category=RuntimeWarning) as f:
        for y in wav:
            chk += np.uint16(y)
            chk += chk << 10 # integer overflow ignored
            chk = chk & 0x7fffffff
            chk ^= (chk >> 5)
    return chk


class NScanCalc:
    def __init__(self, aimask, dimask):
        self.aimask = aimask
        self.dimask = dimask
        self.nanalog = countmask(aimask)
        self.ndigital = countmask(dimask)
        self.scansperstep = stepsize(aimask, dimask)
        abytes = self.nanalog * self.scansperstep * 2
        dbytes = self.ndigital * self.scansperstep // 8
        self.bytesperstep = max(abytes + dbytes, 2)

    def maxinchunk(self, nblocks):
        if self.bytesperstep:
            stepsinchunk = (nblocks * 64 - 4) // self.bytesperstep
        else:
            stepsinchunk = 0*nblocks
        return self.scansperstep * stepsinchunk
    
    def efficiency(self, nscans, separate=False):
        usedbytes = nscans * self.bytesperstep // self.scansperstep
        nblocks = (usedbytes + 4 + 63) // 64
        totalbytes = 64 * nblocks
        if separate:
            return usedbytes, totalbytes
        else:
            return usedbytes / totalbytes

    def bestforcont(self, nblockrange=range(3, 20), penalty=0.01):
        nblocks = np.array(nblockrange)
        nscans = self.maxinchunk(nblocks)
        eff = self.efficiency(nscans)
        ibest = np.argmax(eff - penalty * nblocks)
        return nscans[ibest]
    
    def bestforepi(self, scansperepi, penalty=0.02, details=False):
        used = scansperepi
        scansperepi = roundup(scansperepi, self.scansperstep)
        MAXBLOCKS = 640 # Must match with data.cpp in firmware
        maxbytes = MAXBLOCKS * 64
        if self.bytesperstep:
            scansperchunk = self.scansperstep * np.arange(1, maxbytes // self.bytesperstep)
        else:
            scansperchunk = np.array([1])
        usedperchunk, costperchunk = self.efficiency(scansperchunk, True)
        nchunks = roundup(scansperepi, scansperchunk) // scansperchunk
        cost = costperchunk * nchunks
        eff = used / cost * self.bytesperstep / self.scansperstep
        ibest = np.argmax(eff - penalty * costperchunk // 64)
        nscans = scansperchunk[ibest]
        if details:
            details = {"nscans": nscans,
                       "efficiency": eff[ibest],
                       "scansperstep": self.scansperstep,
                       "scansperepi": scansperepi,
                       "trailingscans": roundup(used, nscans) - used,
                       "chunksperepi": nchunks[ibest],
                       "usedbytesperchunk": usedperchunk[ibest],
                       "blocksperchunk": costperchunk[ibest] // 64,
                       "blocksperepi": cost[ibest] // 64}
            return nscans, details
        else:
            return nscans
        
    
if __name__ == "__main__":
    log.info(" Masks  #chans -> #scans #blks   %loss")
    for dimask in [0, 1, 3, 15]:
        for aimask in [0, 1, 3, 5, 7, 15]:
            calc = NScanCalc(aimask, dimask)
            nscans = calc.bestforcont()
            used, cost = calc.efficiency(nscans, True)
            hdr = f"A{aimask:2} D{dimask:2}  a{calc.nanalog} d{calc.ndigital}"
            res = f" {nscans:4}    {cost//64:2}  {100*used/cost - 100:6.1f}"
            log.info(f"\n{hdr} -> {res}")

    log.info("")
    aimask = 5
    dimask = 3
    calc = NScanCalc(aimask, dimask)
    for count in [100, 200, 500, 1000, 2000, 5000, 10000]:
        hdr = f"A{aimask:2} D{dimask:2} n{count}"
        log.info(f"{hdr} -> {calc.bestforepi(count, details=True)}")
    
    log.info("")
    aimask = 15
    dimask = 0
    calc = NScanCalc(aimask, dimask)
    for count in [100, 200, 500, 1000, 2000, 5000, 10000]:
        hdr = f"A{aimask:2} D{dimask:2} n{count}"
        log.info(f"\n{hdr} -> {calc.bestforepi(count, details=True)}")
    
