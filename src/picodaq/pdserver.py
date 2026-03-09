#!/bin/env python3
# -*- python -*-

import sys
import os
import io
from picodaq import AnalogIn, AnalogOut, DigitalOut, Frequency, Hz, ms
from typing import List, Optional
import numpy as np
import logging
import selectors
import time

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR) #INFO)

t0 = time.time()
def rtime():
    return f"{time.time()-t0:.3f}"

#log.setLevel(logging.DEBUG)

def usage() -> int:
    print("Usage: pdserver port rate_Hz channels", file=sys.stderr)
    return 1


stimdata = {}
n_ = {}
def gen(c, dtype):
    while True:
        n_[c] = n_.get(c, 0)        
        if stimdata[c]:
            m_ = len(stimdata[c][0])
            log.debug(f"gen {c} {n_[c]} {m_} {np.mean(stimdata[c][0])} {np.std(stimdata[c][0])} {stimdata[c][0].dtype} {stimdata[c][0].shape}")
            n_[c] += 100
            yield stimdata[c].pop(0)
        else:
            log.debug(f"gen {c} {list(stimdata.keys())} {n_} ---")
            n_[c]  += 100
            yield np.zeros(100, dtype)


def sendoutput(stdout, dat: Optional[np.ndarray]):
    if dat is None:
        return
    log.debug(f"sendoutput {np.mean(dat, 0)} {np.std(dat, 0)}")
    bts = dat.astype(np.float32).tobytes()
    stdout.write(bts)
    stdout.flush()


def receiveinput(stdin, nchans, nscans=100):
    nbytes = nchans * nscans * 4
    dat = stdin.read(nbytes)
    log.debug(f"received {len(dat)} {nbytes}")
    if len(dat) == 0:
        return # EOF
    nscans = len(dat) // nchans // 4
    return np.frombuffer(dat, np.float32).reshape(nscans, nchans)


def handleinput(stdin,
                aochans, aoidx,
                dolines, doidx):
    dat = receiveinput(stdin, len(aoidx) + len(doidx))
    if dat is None:
        log.debug("handleinput - nothing received")
        return
    log.debug(f"handleinput {dat.shape} {dat.dtype} {dat.mean(axis=0)} {dat.std(axis=0)}")
    for idx, c in zip(aoidx, aochans):
        stimdata[f"ao{c}"].append(dat[:,idx].astype(np.float32))
    for idx, c in zip(doidx, dolines):
        stimdata[f"do{c}"].append(dat[:,idx].astype(np.uint8))
    return True

_ai = [None]

def selectandhandle(sel, stdin,
                    aochans, aoidx,
                    dolines, doidx,
                    timeout=0):
    selectmore = True
    #log.debug(f"selectandhandle {rtime()}")
    while selectmore:
        selectmore = False
        for key, mask in sel.select(timeout):
            if key.data=="stdin":
                #log.debug(f"receiving {rtime()} {_ai[0].dev.reader.lastchunkno}")
                selectmore = True
                if not handleinput(stdin,
                                   aochans, aoidx,
                                   dolines, doidx):
                    return False
                chn = f"ao{aochans[0]}"
                log.debug(f"received {rtime()} {len(stimdata[chn])} {stimdata[chn][-1].shape}")
    return True


def run(stdin, stdout,
        port: str, rate: Frequency,
        aichans: List[int],
        aochans: List[int],
        dolines: List[int],
        aoidx: List[int],
        doidx: List[int]):

    ai = AnalogIn(port=port, rate=rate, channels=aichans)
    _ai[0] = ai
    ai.open()
    if aochans:
        ao = AnalogOut(port=port, rate=rate, maxahead=300*ms)
        ao.open()
        def agengen(c):
            return lambda: gen(f"ao{c}", np.float32)
        for c in aochans:
            stimdata[f"ao{c}"] = []
            ao[c].sampled(agengen(c))
    else:
        ao = None    
    if dolines:
        do = DigitalOut(port=port, rate=rate, maxahead=300*ms)
        do.open()
        def dgengen(c):
            return lambda: gen(f"do{c}", np.uint8)
        for c in dolines:
            stimdata[f"do{c}"] = []
            do[c].sampled(dgengen(c))
    else:
        do = None
    log.info("opened")
    
    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ, "stdin")

    # if not selectandhandle(sel, stdin,
    #                        aochans, aoidx,
    #                        dolines, doidx,
    #                        .2):
    #     print("eof before anything", file=sys.stderr)        
    #     return 0
    # 
    # state = {c: len(stimdata[c]) for c in stimdata}
    # log.debug(f"got input {state}")
    # 
    log.debug(f"starting {rtime()}")
    # log.debug(f"select {sel.select(0)}")
    
    ai.start()
    log.debug(f"started {rtime()}")
    while True:
        if not selectandhandle(sel, stdin,
                               aochans, aoidx,
                               dolines, doidx):
            break # input closed
        dat = ai.read()
        if not len(dat):
            break # Run stopped
        sendoutput(stdout, dat)
        if ao:
            ao.poll()
        elif do:
            do.poll()

    if ao:
        ao.close()
    if do:
        do.close()
    ai.close()
    log.info("closed")

    return 0


def main() -> int:
    if len(sys.argv) < 4:
        return usage()

    port = sys.argv[1]
    rate = int(sys.argv[2]) * Hz

    aichans = []
    aochans = []
    dolines = []
    aoidx = []
    doidx = []
    k = 0
    for arg in sys.argv[3:]:
        if arg.startswith("ai"):
            aichans.append(int(arg[2:]))
        elif arg.startswith("ao"):
            aochans.append(int(arg[2:]))
            aoidx.append(k)
            k += 1
        elif arg.startswith("do"):
            dolines.append(int(arg[2:]))
            doidx.append(k)
            k += 1
        else:
            return usage()

    log.info(f"Port {port}")
    log.info(f"Rate {rate}")
    log.info(f"AI channels {aichans}")
    log.info(f"AO channels {aochans} ({aoidx})")
    log.info(f"DO lines {dolines} ({doidx})")

    with io.FileIO(sys.stdout.fileno(), "wb") as stdout:
        with io.FileIO(sys.stdin.fileno(), "rb") as stdin:
            return run(stdin, stdout,
                       port, rate,
                       aichans,
                       aochans, dolines,
                       aoidx, doidx)

        
if __name__ == "__main__":
    main()

