#!/bin/env python3
# -*- python -*-

import sys
import os
import io
from picodaq import AnalogIn, AnalogOut, DigitalOut, Frequency, Hz, ms
from typing import List, Optional
import numpy as np
import logging
import queue
import threading
import time


log = logging.getLogger()

#logging.basicConfig(level=logging.ERROR) #INFO)

t0 = time.time()
def rtime():
    return f"{time.time()-t0:.3f}"

#log.setLevel(logging.ERROR)

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
            #log.debug(f"gen {c} {n_[c]} {m_} {np.mean(stimdata[c][0])} {np.std(stimdata[c][0])} {stimdata[c][0].dtype} {stimdata[c][0].shape}")
            n_[c] += 100
            yield stimdata[c].pop(0)
        else:
            #log.debug(f"gen {c} {list(stimdata.keys())} {n_} ---")
            n_[c]  += 100
            yield np.zeros(100, dtype)


def sendoutput(stdout, dat: Optional[np.ndarray]):
    if dat is None:
        return
    #log.debug(f"sendoutput {np.mean(dat, 0)} {np.std(dat, 0)}")
    bts = dat.astype(np.float32).tobytes()
    stdout.write(bts)
    stdout.flush()


def receiveinput_worker(source_fd, dest_queue, nchans, nscans):
    nbytes = nchans * nscans * 4
    while True:
        dat = source_fd.read(nbytes)
        if len(dat) == 0:
            dest_queue.put(None)
            return # EOF
        nscans = len(dat) // nchans // 4
        dest_queue.put(np.frombuffer(dat, np.float32).reshape(nscans, nchans))

class StdInThread:
    def __init__(self, stdin, aochans, aoidx, dolines, doidx):
        self.stdin = stdin
        self.aochans = aochans
        self.aoidx = aoidx
        self.dolines = dolines
        self.doidx = doidx
        self.queue = queue.Queue()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=receiveinput_worker,
                         args=(self.stdin,
                               self.queue,
                               len(self.aoidx) + len(self.doidx),
                               100))
        self.thread.start()

    def join(self):
        if self.thread is not None:
            self.thread.join()
                     
    def read(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return []


def handleinput(thread):
    dat = thread.read()
    if dat is None:
        return # EOF
    if len(dat) == 0:
        return False
    for idx, c in zip(thread.aoidx, thread.aochans):
        stimdata[f"ao{c}"].append(dat[:,idx].astype(np.float32))
    for idx, c in zip(thread.doidx, thread.dolines):
        stimdata[f"do{c}"].append(dat[:,idx].astype(np.uint8))
    return True


def run(stdin, stdout,
        port: str, rate: Frequency,
        aichans: List[int],
        aochans: List[int],
        dolines: List[int],
        aoidx: List[int],
        doidx: List[int]):

    ai = AnalogIn(port=port, rate=rate, channels=aichans)
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
    
    ai.start()
    thread = StdInThread(stdin, aochans, aoidx, dolines, doidx)
    thread.start()
    while True:
        if handleinput(thread) is None:
            break # input closed
        dat = ai.read()
        if len(dat) == 0:
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

