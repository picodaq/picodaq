#!env python3

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import time
import numpy as np

objs = {}
rate = 10000
f = 200


import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

t0 = time.time()
def rtime():
    return f"{time.time()-t0:.3f}"


n0 = 0
def read(proc):
    global n0
    dat = proc.read(400)
    ar = np.frombuffer(dat, np.float32)
    print(f"{rtime()}: {n0} {len(ar)} {ar.mean():.3f} {ar.std():.3f}")
    n0 += len(ar)


k = 0
def write(proc):
    global k
    tt = np.arange(300).astype(float) / rate
    data = np.sin(2*np.pi*tt*f).astype(np.float32)
    print(f"{rtime()}: writing {k}")
    n = proc.write(data.tobytes())
    print(f"{rtime()}: wrote {n} / {data.shape}")
    k += 1
    if k >= 5:
        stop()


def stop():
    print(f"{rtime()}: stop")    
    objs["timer"].stop()
    objs["proc"].terminate()


def start():
    global t0
    t0 = time.time()
    
    print("hello world")
    proc = QProcess()
    proc.readyReadStandardOutput.connect(lambda: read(proc))
    proc.setProcessChannelMode(QProcess.ForwardedErrorChannel)
    proc.start("../software/pdserver",
               ["ACM0", "10000", "ai2", "ao3"])
    if not proc.waitForStarted():
        print(proc.errorString())
        print(proc.readAllStandardError())
        raise Exception("Not started")
    write(proc)
    timer = QTimer()
    timer.timeout.connect(lambda: write(proc))
    timer.start(1000)
    objs["proc"] = proc
    objs["timer"] = timer


def main():
    app = QApplication(sys.argv)
    win = QPushButton("start")
    win.clicked.connect(lambda: start())
    win.show()
    app.exec()

main()
