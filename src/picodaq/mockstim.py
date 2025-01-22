import numpy as np
from numpy.typing import ArrayLike
import copy
import logging

from .units import V, s, ms, Hz, Voltage, Time, Frequency
from .stimulus import Pulse, Train, Series, Parametrized, Sampled
from .stimulus import TTL, Square, Sawtooth, Triangle, Wave
from .dac import OutRef

log = logging.getLogger(__name__)


def mockpulse(pulse: Pulse, rate: Frequency,
              vv_V: ArrayLike, t0: Time) -> np.ndarray:
    """Represent a pulse as a vector of samples
    
    """

    dt_s = 1/rate.as_("Hz")
    t0_s = t0.as_("s")
    t1_s = pulse.duration1.as_("s")
    t2_s = pulse.duration1.as_("s")
    if isinstance(pulse, TTL):
        v1_V = 5
        v2_V = 0
    else:
        v1_V = pulse.amplitude1.as_("V")
        v2_V = pulse.amplitude2.as_("V")
    # rebuilding tt_s is rather wasteful. but do we care?
    tt_s = np.arange(len(vv_V)) * dt_s - t0_s
    use1 = (tt_s >= 0) & (tt_s < t1_s)
    use2 = (tt_s >= t1_s) & (tt_s < t1_s + t2_s)
    if isinstance(pulse, Square):
        vv_V[use1] = v1_V
        vv_V[use2] = v2_V
    elif isinstance(pulse, Sawtooth):
        vv_V[use1] = v1_V + (tt_s[use1] / (t1_s - dt_s)) * (v2_V - v1_V)
    elif isinstance(pulse, Triangle):
        idx = (tt_s >= 0) & (tt_s < t1_s/2)
        vv_V[idx] = (tt_s[idx] + dt_s) / (t1_s/2) * v1_V
        idx = (tt_s >= t1_s/2) & (tt_s < t1_s)
        vv_V[idx] = (t1_s - dt_s - tt_s[idx]) / (t1_s/2) * v1_V
        idx = (tt_s >= t1_s) & (tt_s < t1_s + t2_s/2)
        vv_V[idx] = (tt_s[idx] - t1_s + dt_s) / (t2_s/2) * v2_V
        idx = (tt_s >= t1_s + t2_s/2) & (tt_s < t1_s + t2_s)
        vv_V[idx] = (t1_s + t2_s - dt_s - tt_s[idx]) / (t2_s/2) * v2_V
    elif isinstance(pulse, Wave):
        i0 = int(t0_s / dt_s)
        N = min(len(tt_s) - i0, len(pulse.data))
        if N > 0:
            vv_V[i0:i0+N] = pulse.data[:N] * v1_V
    elif isinstance(pulse, Pulse):
        vv_V[use1] = v1_V
    else:
        raise ValueError("Unsupported stimulus shape")


def mocktrain(train: Train, rate: Frequency, vv_V: ArrayLike, t0: Time):
    dt_s = 1 / rate.as_("Hz")
    pulse = copy.copy(train.pulse)
    log.debug(f"mocktrain {train.pulse} {pulse}")
    period = train.pulseperiod
    tpulse = t0
    for k in range(train.pulsecount):
        mockpulse(pulse, rate, vv_V, tpulse)
        tpulse += period
        period += train.perpulse.pulseperiod
        pulse.apply(train.perpulse)
    return vv_V


def mockstim(stim: Parametrized,
             rate: Frequency, duration: Time,
             episodic: bool = False) -> np.array:
    isttl = isinstance(stim.series.train.pulse, TTL)
    vv_V = np.zeros(int((rate*duration).plain()),
                    bool if isttl else np.float32)
    train = copy.copy(stim.series.train)
    period = stim.series.trainperiod
    N = stim.series.traincount
    t0 = stim.delay
    if episodic:
        vv_V = vv_V.reshape(1, -1).repeat(N, 0)
        for k in range(N):
            mocktrain(train, rate, vv_V[k], t0)
    else:
        for k in range(N):
            mocktrain(train, rate, vv_V, t0)
            t0 += period
            period += stim.series.pertrain.trainperiod
    if isttl and stim.series.train.pulse.amplitude1:
        vv_V = np.logical_not(vv_V)
    return vv_V


def mocksampled(stim: Sampled, rate: Frequency, duration: Time) -> np.array:
    vv = np.zeros(int((rate*duration).plain()), np.float32)
    N = len(vv)
    if callable(stim.data):
        i0 = 0
        gen = stim.data()
        for dat in gen:
            n = min(len(dat), N - n0)
            vv[i0:i0+n] = dat
            i0 += n
    else:
        n = min(len(stim.data), N)
        vv[:n] = stim.data[:n]
    return vv * stim.scale.as_("V")


def mock(src: OutRef, duration: Time) -> np.array:
    stim = src.stream.stimuli[src.idx]
    rate = src.stream.dev.rate
    episodic = src.stream.dev.epi_dur is not None
    if isinstance(stim, Parametrized):
        return mockstim(stim, rate, duration, episodic)
    elif isinstance(stim, Sampled):
        return mocksampled(stim, rate, duration)
