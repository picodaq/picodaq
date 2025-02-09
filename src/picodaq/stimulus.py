from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable
import copy

from .units import V, s, ms, Voltage, Time, Frequency


class Deltas:
    """Representation of changes to stimulation parameters

    This can be used to apply changes on a per-pulse basis within
    a train, or on a per-train basis within a series.

    Each parameter applies an additive change to the like-named
    parameter of a Pulse or Train. Specifically:
    
    Parameters:
    
        amplitude: additive change to the `amplitude` parameter of
                    `Pulse`, `Square`, or `Triangle` shapes

        amplitude2: additive change to the `amplitude2` parameter of
                     `Square` or `Triangle` shapes
    
        duration: additive change to the `duration` parameter of any shape

        duration2: additive change to the `duration2` parameter of
                     `Square` or `Triangle` shapes
    
        start: additive change to the `start` parameter of the
                `Sawtooth` shape

        end: additive change to the `end` parameter of the
              `Sawtooth` shape

        scale: additive change to the `scale` parameter of the
                `Wave` shape

        pulsecount: additive change to the number of pulses in a train

        pulseperiod: additive change to the period between pulses in
                      a train

        trainperiod: additive change to the period between trains in
                      a series

    (There is no `traincount` parameter that would modify the number
    of trains in a series, as our convention is that there is only a
    single series, which may be repeated verbatim.) 
        
    Applying changes to a parameter that does not exist for a given
    pulse may lead to unexpected behavior. For instance, you cannot
    change the `duration` of a `Wave` stimulus.

    It is an error to apply changes to `pulsecount` or `trainperiod`
    on a per-pulse basis, but changing `pulseperiod` on a per-pulse
    basis is allowed.

    """
    
    def __init__(self,
                 amplitude: Voltage = 0*V, start: Voltage = 0*V,
                 scale: Voltage = 0*V,
                 amplitude2: Voltage = 0*V, end: Voltage = 0*V,
                 duration: Time = 0*ms, 
                 duration2: Time = 0*ms,
                 pulsecount: int = 0,
                 pulseperiod: Time = 0*ms,
                 trainperiod: Time = 0*ms):
        self.amplitude1 = amplitude + start + scale
        self.amplitude2 = amplitude2 + end
        self.duration1 = duration
        self.duration2 = duration2
        self.pulsecount = pulsecount
        self.pulseperiod = pulseperiod
        self.trainperiod = trainperiod
        

class Pulse:
    """A simple monophasic pulse

    Parameters:

        amplitude: The amplitude of the pulse (relative to baseline),
                   in units of voltage. The amplitude may be negative.

        duration: The duration of the pulse, in units of time.

    """
    
    def __init__(self,
                 amplitude: Voltage,
                 duration: Time):
        super().__init__()
        self.amplitude1 = amplitude
        self.duration1 = duration
        self.amplitude2 = 0*V
        self.duration2 = 0*ms
        if self.amplitude1 == 0*V or self.duration1 == 0*V:
            self.name = "off"
        else:
            self.name = "mono"

    def Vmax(self) -> Voltage:
        return max(self.amplitude1.as_(V), self.amplitude2.as_(V), 0) * V

    def Vmin(self) -> Voltage:
        return min(self.amplitude1.as_(V), self.amplitude2.as_(V), 0) * V

    def duration(self) -> Time:
        return self.duration1 + self.duration2

    def apply(self, delta: Deltas):
        # """Modify the pulse based on given Deltas."""
        self.duration1 += delta.duration1
        self.duration2 += delta.duration2
        self.amplitude1 += delta.amplitude1
        self.amplitude2 += delta.amplitude2

            
class TTL(Pulse):
    """Representation of a digital stimulus

    Parameters:

        duration: The duration of the pulse, in units of time.

        active_low: If true, the polarity of the entire stimulus
                     is inverted, that is, the line is high outside
                     the pulse and low inside.

    This may only be applied to digital outputs. To send TTL
    pulses to an analog output, use the Pulse shape with
    amplitude set to 5 V.

    """
    
    def __init__(self,
                 duration: Time,
                 active_low: bool = False):
        super().__init__(0*V, duration)
        self.amplitude1 = active_low
        self.name = "ttl"
        
    def apply(self, delta: Deltas):
        # """Modify the pulse based on given Deltas."""
        self.duration1 += delta.duration1
        self.duration2 += delta.duration2               
    
class Square(Pulse):
    """Representation of a square wave (biphasic pulse)

    Parameters:

        amplitude: The amplitude of the first phase of the wave
                     (relative to baseline), in units of voltage.
                     The amplitude may be negative.

        duration: The duration of the first phase of the wave, in
                   units of time.

        amplitude2: The amplitude of the second phase of the wave.
                     If omitted, defaults to the opposite of the
                     polarity of the first phase.

        duration2: The duration of the second phase of the wave. If
                    omitted, defaults to the duration of the first phase.

    The full duration of the waveform is the sum of the duration of the
    two phases. There is never a delay between first and second phases.

    For a traditional function-generator square wave with peak-to-peak
    amplitude A and frequency F, set amplitude = A/2 and duration = 0.5/F
    and leave amplitude2 and duration2 unspecified.
    """

    def __init__(self,
                 amplitude: Voltage,
                 duration: Time,
                 amplitude2: Voltage | None = None,
                 duration2: Time | None = None):
        super().__init__(amplitude, duration)
        if amplitude2 is None:
            self.amplitude2 = -amplitude
        else:
            self.amplitude2 = amplitude2            
        if duration2 is None:
            self.duration2 = duration
        else:
            self.duration2 = duration2
        self.name = "bi"

        
class Sawtooth(Pulse):
    """Representation of a sawtooth wave

    Parameters:

        start: Voltage at start of the waveform, relative
                to baseline

        end: Voltage at end of the waveform, ditto

        duration: The duration of the ramp, in units of time.

    For a traditional function-generator sawtooth with peak-to-peak
    amplitude A and frequency F, set start = -A/2, end = +A/2,
    and duration = 1/F.
    """    
    def __init__(self,
                 start: Voltage,
                 end: Voltage,
                 duration: Time):
        
        super().__init__(start, duration)
        self.amplitude2 = end
        self.name = "saw"
        
        
class Triangle(Pulse):
    """Representation of a triangle wave

    Parameters:

        amplitude: The amplitude of the first phase of the wave
                     (relative to baseline), in units of voltage.
                     The amplitude may be negative.

        duration: The duration of the first phase of the wave, in
                   units of time.

        amplitude2: The amplitude of the second phase of the wave.
                     If omitted, defaults to the opposite of the
                     polarity of the first phase.

        duration: The duration of the second phase of the wave.

    To create a traditional function-generator triangle wave with
    peak-to-peak amplitude A and frequency F, set amplitude = A/2
    and duration = 0.5/F. A second phase with equal duration
    and opposite polarity will be added automatically.

    To create asymmetric triangles, the parameters of the second
    phase may be specified explicitly. If `amplitude2` is given
    but `duration2` is omitted, the latter is calculated to keep
    the (absolute) slope of the wave constant.

    """
    
    def __init__(self,
                 amplitude: Voltage,
                 duration: Time,
                 amplitude2: Voltage | None = None,
                 duration2: Time | None = None):
        super().__init__(amplitude, duration)
        if amplitude2 is None:
            self.amplitude2 = -amplitude
        else:
            self.amplitude2 = amplitude2
        if duration2 is None:
            ratio = (self.amplitude2/self.amplitude1).plain()
            self.duration2 = duration * np.abs(ratio)
        else:
            self.duration2 = duration2
        self.name = "tri"

        
class Wave(Pulse):
    """Representation of an arbitrary wave stimulus

    Parameters:

        data: A vector in which every value represents a single
               sample.

        scale: Scale factor applied to the data, in units of
                voltage.

    For instance, to create a sine wave with 1 Vpp amplitude,
    you could set data = np.sin(...) and scale = 0.5*V,
    or equally, data = 0.5 * np.sin(...) and scale = 1*V.
    """
    
    def __init__(self, data: ArrayLike, scale: Voltage = 1*V,
                 rate: Frequency | None = None):
        super().__init__(scale, 0*s)
        self.data = data
        self.name = "wave"

    def Vmax(self) -> Voltage:
        return np.max(self.data) * self.amplitude1.abs()

    def Vmin(self) -> Voltage:
        return np.min(self.data) * self.amplitude1.abs()

        
class Train:
    """Representation of a train of pulses

    Parameters:

        pulse: The waveform to be used in the train.

        pulsecount: The number of pulses in the train.

        duration: The total duration of the train.

        pulseperiod: The period between consecutive pulses.

        perpulse: Changes to apply to parameters between pulses

    For analog outputs, the `pulse` may be any of Pulse, Square,
    Sawtooth, Triangle, or Wave; for digital outputs, it may only
    be TTL.

    Either the number of pulses must be specified (with `pulsecount`),
    or the total duration of the train (with `duration`), but not
    both. If `duration` is given, the pulse count is calculated for a
    tight fit. For instance, if the pulse period is 100 ms and each
    pulse has a duration of 20 ms, then specifying duration as 420 ms
    will yield 5 pulses.

    The `pulseperiod` is measured start-to-start. If the `pulseperiod`
    is less than the duration of the pulse, the next pulse starts
    immediately after the previous one.

    Deltas specified with `perpulse` are applied between pulses. That
    is: The first pulse and the following period are as stated; the
    second pulse and following period are modified by adding the
    deltas once; etc.

    """
    
    def __init__(self, pulse: Pulse,
                 pulsecount: int | None = None,
                 pulseperiod: Time = 0*s,
                 duration: Time | None = None,
                 perpulse: Deltas = Deltas()):

        self.pulse = pulse
        if (duration is None) == (pulsecount is None):
            raise ValueError("Either pulsecount or duration must be given (but not both)")
        self.pulseperiod = pulseperiod
        if perpulse.pulsecount:
            raise ValueError("Cannot change pulse count on a per-pulse basis")
        if perpulse.trainperiod.as_(ms) != 0:
            raise ValueError("Cannot change train period on a per-pulse basis")
        self.perpulse = perpulse
        if duration is None:
            if int(pulsecount) != pulsecount:
                raise ValueError("Pulse count must be integer")
            self.pulsecount = pulsecount
        else:
            self.calculate_pulsecount(duration)


    def calculate_pulsecount(self, duration):
        totdur = 0
        per1 = self.pulseperiod
        dur1 = self.pulse.duration()
        if dur1 <= 0*s:
            raise ValueError("Pulse duration must be positive")
        k = 0
        while True:
            per = per1
            dur = dur1
            if per < dur:
                per = dur
            if totdur + per >= duration:
                if totdur + dur > duration:
                    return k
                else:
                    return k + 1
            totdur += per
            k += 1
            per1 += self.perpulse.pulseperiod
            dur1 += self.perpulse.duration1 + self.perpulse.duration2

    def apply(self, delta: Deltas):
        self.pulse.apply(delta)
        self.pulsecount += delta.pulsecount
        self.pulseperiod += delta.pulseperiod

    def nextpulse(self):
        self.apply(self.perpulse)

    def duration(self, tight: bool = False) -> Time:
        """The duration of the train

        Parameters:

            tight: If true, the duration is measured to the end of the
                    final pulse. Otherwise, it includes the (fictive)
                    interval after the final pulse.
        """
        totdur = 0*s
        per1 = self.pulseperiod
        dur1 = self.pulse.duration()
        per = dur = 0*s
        for k in range(self.pulsecount):
            per = per1
            dur = dur1
            if per < dur:
                per = dur
            totdur += per
            per1 += self.perpulse.pulseperiod
            dur1 += self.perpulse.duration1 + self.perpulse.duration2
        if tight:
            totdur -= per - dur
        return totdur
        
    def Vmax(self) -> Voltage:
        p = copy.copy(self.pulse)
        Vmax = 0
        for k in range(self.pulsecount):
            Vmax = max(Vmax, p.Vmax().as_('V'))
            p.apply(self.perpulse)
        return Vmax * V

    def Vmin(self) -> Voltage:
        p = copy.copy(self.pulse)
        Vmin = 0
        for k in range(self.pulsecount):
            Vmin = min(Vmin, p.Vmin().as_(V))
            p.apply(self.perpulse)
        return Vmin * V

        
class Series:
    """Representation of a series of trains

    Parameters:

        train: The constituent train of the series.

        traincount: The number of trains in the series.

        duration: The total duration of the series.

        trainperiod: The period between consecutive trains.

        pertrains: Changes to apply to parameters between trains.

    Either the number of trains much be specified (with `traincount`),
    or the total duration of the series (with `duration`), but not
    both.

    The `trainperiod` is measured start-to-start and is required
    except in episodic mode. If less than the duration of the train,
    the trains follow each other immediately.

    Deltas specified with `pertrain` are applied between trains. That
    is: The first train and the following period are as stated; the
    second train and following period are modified by adding the
    deltas once; the third train and following period are modified by
    twice the deltas; etc.

    Instead of a train, it is allowed to make a series comprise
    repeated individual pulses. This is strictly equivalent to
    putting in a train comprising just that one pulse.


    """
    
    def __init__(self, train: Train | Pulse,
                 traincount: int | None = None,
                 trainperiod: Time = 0*ms,
                 duration: Time | None = None,
                 pertrain: Deltas = Deltas()):
        if isinstance(train, Pulse):
            self.train = Train(train, 1, 0*ms)
        else:
            self.train = train
        if (duration is None) == (traincount is None):
            raise ValueError("Either traincount or duration must be given (but not both)")
        self.trainperiod = trainperiod
        self.pertrain = pertrain
        if duration is None:
            self.traincount = traincount
        else:
            self.traincount = self.calculate_traincount(duration)

    def calculate_traincount(self, duration: Time):
        totdur = 0*s
        per1 = self.trainperiod
        train = copy.copy(self.train)
        train.pulse = copy.copy(self.train.pulse)
        k = 0
        while True:
            dur = train.duration()  # conceptual duration of train incl. following interval
            tightdur = train.duration(tight=True)  # tight duration of train, excl. following interval
            if tightdur < 0*s:
                raise ValueError("Train duration must be positive")
            per = per1
            if per < dur:
                per = dur
            if totdur + per >= duration: # Might just barely fit this, but no more
                if totdur + tightdur > duration: # Doesn't fit, even tightly
                    return k
                else:
                    return k + 1
            totdur += per
            k += 1
            train.apply(self.pertrain)
            per1 += self.pertrain.trainperiod

    def apply(self, delta: Deltas):
        self.train.apply(delta)
        self.trainperiod += delta.trainperiod

    def nexttrain(self):
        self.apply(self.pertrain)

    def Vmax(self) -> Voltage:
        t = copy.copy(self.train)
        t.pulse = copy.copy(self.train.pulse)
        Vmax = 0
        for k in range(self.traincount):
            Vmax = max(Vmax, t.Vmax().as_(V))
            t.apply(self.pertrain)
        return Vmax * V

    def Vmin(self) -> Voltage:
        t = copy.copy(self.train)
        t.pulse = copy.copy(self.train.pulse)
        Vmin = 0
        for k in range(self.traincount):
            Vmin = min(Vmin, t.Vmin().as_(V))
            t.apply(self.pertrain)
        return Vmin * V

    def duration(self, tight: bool = False) -> Time:
        """The duration of the series

        Parameters:

            tight: If true, the duration is measured to the end of the
                    final pulse. Otherwise, it includes the (fictive)
                    interval after the final pulse.
        """
        totdur = 0*s
        per1 = self.trainperiod
        trn = copy.copy(self.train)
        per = dur = 0*s
        for k in range(self.traincount):
            dur = trn.duration(tight)
            per = per1
            if per < dur:
                per = dur
            totdur += per
            per1 += self.pertrain.pulseperiod
            trn.apply(self.pertrain)
        if tight:
            totdur -= per - dur
        return totdur
            


class Parametrized:
    """Define a parametrized stimulation sequence for a single output

    Parameters:

        stim: a Series, a Train, or a single Pulse.

        delay: delay to first pulse

        repeat: repeat period for the stimulus or None

        offset: offset voltage for the channel


    The delay is measured from the start of the recording to the
    first pulse in continuous mode, or from the start of each
    episode to its first pulse in episodic mode.

    The repeat period, if given, is the start-to-start period for
    repeating the entire sequence. If not given, the stimulus does not
    repeat.

    If a stimulus is used on an AnalogOut channel, the offset
    voltage is applied continuously, even outside of pulses and
    trains. On DigitalOut lines, the offset is ignored.

    """

    def __init__(self,
                 stim: Pulse | Train | Series,
                 delay: Time = 0*s,
                 repeat: Time | None = None,
                 offset: Voltage = 0*V):
        
        if isinstance(stim, Train) or isinstance(stim, Pulse):
            self.series = Series(stim, 1, 0*ms)
        else:
            self.series = stim
        self.delay = delay
        self.repeat = repeat
        self.offset = offset


class Sampled:
    """Define raw data to be sent to a single output channel

    Parameters:

        data: output data for the channel
        scale: scale factor to apply to the data
        offset: offset to be added after scaling
        raw: data represent raw binary values

    You may either specify prepared data as an array (T-vector) or
    specify a callable the generates the data on the fly and that
    "yields" data in arbitrary quantities.

    The data (whether predefined or generated) are multiplied by the
    given scale factor. Then, the given offset is added and the result
    is converted to digital units. By default, the scale is one volt
    and the offset is zero. Alternatively, if you specify raw=True,
    you may specify data as raw binary values (16-bit signed
    integers). In that case, scale and offset may not be specified.

    When digital data are represented as a Sampled, nonzero values map
    to digital 1 and zero values to digital 0. In this case, the
    `scale`, `offset`, and `raw` parameters are ignored.

    """
    def __init__(self, data: ArrayLike | Callable,
                 scale: Voltage = 1*V, offset: Voltage = 0*V,
                 raw: bool = False):
        self.data = data
        self.scale = scale
        self.offset = offset
        self.raw = raw
        if raw:
            if self.scale.as_("V") != 1 or self.offset.as_("V") != 0:
                raise ValueError("Raw data cannot be scaled or offset")
    
__all__ = ["Pulse", "Square", "Sawtooth", "Triangle", "Wave",
           "TTL",
           "Train", "Series",
           "Deltas"
           ]
