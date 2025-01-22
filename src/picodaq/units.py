#!/usr/bin/python3

#   UNITS v. 0.10, Copyright (C) 2009, 2020 Daniel Wagenaar. 
#   This software comes with ABSOLUTELY NO WARRANTY. See code for details.
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; version 2 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.



import re
import numpy as np
from numpy.typing import ArrayLike
from typing import Self, Tuple
import copy
import logging

log = logging.getLogger(__name__)


######################################################################
# Prepare unit database
def _mkunitcode() -> dict[str, np.ndarray]:
    units = 'mol A g m s'.split(' ')
    uc = {}
    U = len(units)
    for u in range(U):
        vec = np.zeros(U)
        vec[u] = 1
        uc[units[u]] = vec
    uc[1] = np.zeros(U)
    return uc

_unitcode = _mkunitcode()


def _decodeunit(u: str | int) -> np.ndarray:
    return _unitcode[u].copy()


def _mkprefix() -> dict[str, int]:
    prefix = {'d': -1, 'c': -2,
              'm': -3, 'u': -6, 'n': -9, 'p': -12, 'f': -15,
              'k': 3, 'M': 6, 'G': 9, 'T': 12, 'P': 15}
    altprefix = ['deci=d',
                  'centi=c',
                  'milli=m',
                  'micro=μ=u',
                  'nano=n',
                  'pico=p',
                  'femto=f',
                  'kilo=k',
                  'mega=M',
                  'giga=G',
                  'tera=T',
                  'peta=P']
    for ap in altprefix:
        bits = ap.split('=')
        val = bits.pop()
        for b in bits:
            prefix[b] = prefix[val]
    return prefix

_prefix = _mkprefix()


def _mkunitmap() -> dict[str, str]:
    altunits = ['meter=meters=m', 
                'second=seconds=sec=secs=s', 
                'gram=grams=gm=g',
                'lb=lbs=pound=453.59237 g',
                'amp=amps=ampere=amperes=Amp=Ampere=Amperes=A',
                'min=mins=minute=minutes=60 s',
                'h=hour=hours=60 min',
                'day=days=24 hour',
                'in=inch=2.54 cm',
                'l=L=liter=liters=1e-3 m^3',
                'Hz=Hertz=hertz=cyc=cycles=s^-1',
                'C=Coulomb=coulomb=Coulombs=coulombs=A s',
                'N=newton=Newton=newtons=Newtons=kg m s^-2',
                'lbf=4.4482216 kg m / s^2',
                'J=joule=joules=Joule=Joules=N m',
                'W=watt=Watt=watts=Watts=J s^-1',
                'V=Volt=volt=Volts=volts=W A^-1',
                'Pa=pascal=Pascal=N m^-2',
                'bar=1e5 Pa',
                'atm=101325 Pa',
                'torr=133.32239 Pa',
                'psi=6894.7573 kg / m s^2',
                'Ohm=Ohms=ohm=ohms=V A^-1',
                'mho=Mho=Ohm^-1',
                'barn=1e-28 m^2',
                'M=molar=mol l^-1']
    unitmap = {}
    for au in altunits:
        bits = au.split('=')
        val = bits.pop()
        for b in bits:
            unitmap[b] = val
    return unitmap

_unitmap = _mkunitmap()


def _fracdecode(s: str) -> Tuple[float, np.array]:
    idx = s.find('/')
    if idx<0:
        numer = s
        denom = ''
    else:
        numer = s[:idx]
        denom = s[idx+1:].replace('/', ' ')

    multis = [ numer, denom ]
    mul = []
    code = []
    for q in range(2):
        mul.append(1)
        code.append(_decodeunit(1))
        factors = multis[q].split(' ')
        for fac in factors:
            mu, co = _factordecode(fac)
            mul[q] *= mu
            code[q] += co
    mul = mul[0]/mul[1]
    code = code[0] - code[1]
    return mul, code


_numre = re.compile('^[-0-9+.]')
def _factordecode(fac: str) -> Tuple[float, np.array]:
    if _numre.search(fac):
        # It's a number
        return float(fac), _decodeunit(1)

    idx = fac.find('^')
    if idx>=0:
        base = fac[:idx]
        powfrac = fac[idx+1:]
        if powfrac.find('^')>0:
            raise ValueError('Double exponentiation')
        idx = powfrac.find('|')
        if idx>=0:
            pw = float(powfrac[:idx]) / float(powfrac[idx+1:])
        else:
            pw = float(powfrac)
    else:
        base=fac
        pw = 1

    # Let's decode the UNIT
    if base=='':
        return 1., _decodeunit(1)
    elif base in _unitcode:
        # It's a base unit without a prefix
        mu = 1
        co = _decodeunit(base)*pw
        return mu, co
    elif base in _unitmap:
        mu, co = _fracdecode(_unitmap[base])
        mu = mu**pw
        co = co*pw
        return mu, co
    else:
        # So we must have a prefix
        for pf in reversed(_prefix):
            if base.startswith(pf):
                L = len(pf)
                mu, co = _fracdecode(base[L:])
                mu *= 10**_prefix[pf]
                mu = mu**pw
                co = co*pw
                return mu, co
    raise ValueError(f'I do not know of a unit named “{fac}”')

######################################################################
class Quantity:
    """Representation of a value with associated units
        
    `Quantity(value, units)`, where `value` is a number and
    `units` a string, represents the given quantity. For instance::

        Quantity(9.81, 'm/s^2')

    For convenience, ::

        Quantity('9.81 m/s^2')

    also works.

    Examples
    ========

    ::

        Quantity(4, 'lbs').as_('kg') # -> 1.814
        Quantity('3 V / 200 mA').as_('Ohm') # -> 15.0
        Quantity('psi').definition() # -> '6894.7573 kg m^-1 s^-2'
        (Quantity('2 nA') * Quantity('30 MOhm')).as_('mV') # -> 60
        Quantity('kg m / s^2') # but you can also just say “newton”
        Quantity('J / Hz^1|2') # Joules per root-Hertz
    
    Notes
    =====

    - Within unit strings:
    
      - Multiplication is implicit; do not attempt to write '*'.
      - Fractions in exponents are written with '|' rather than '/'.
      - Division marked by '/' binds most loosely, so that, e.g.,
        "kg / m s"
        is interpreted as kilograms per meter per second.
      - Spaces between units are required. "mN" is a millinewton,
        not a newton meter.
      - Syntax checking is not overly rigorous. Some invalid
        expressions may return meaningless values without a
        reported error.

    - Addition, subtraction, multiplication, and division of
      quantities with each other is supported. Also, multiplication and
      division with numbers.

    - Comparison between quantities is supported. Comparison for
      (in)equality between incompatible units returns false
      (true). Other comparisons between incompatible units raises an
      exception.

    - Dimensionless quantities can be converted back to plain numbers
      with the plain() method. For instance::
    
          (Quantity("10 kHz") * Quantity("10 ms")).plain() # -> 100

    Technical details
    =================
    
    The full syntax for unit specification is:

    BASEUNIT =
        m | s | g | A | mol
    
    PREFIX =
        m | u | n | p | f | k | M | G | T
    
    ALTUNIT =
        meter | meters | second | seconds | sec | secs |
        gram | grams | gm | amp | amps | ampere | amperes | 
        Amp | Ampere | Amperes
    
    ALTPREFIX =
        milli | micro | μ | nano | pico | femto | kilo |
        mega | Mega | giga | Giga | tera | Tera
    
    DERIVEDUNIT =
        Hz | Hertz | hertz | cyc | cycles |
        V | volt | Volt | volts | Volts |
        W | watt | Watt | watts | Watts |
        N | newton | Newton | newtons | Newtons |
        Pa | pascal | Pascal |
        J | joule | joules | Joule | Joules |
        barn | 
        Ohm | Ohms | ohm | ohms | mho | Mho |
        in | inch |
        bar | atm | torr | psi
        M | molar
                  
    UNIT =
        (PREFIX | ALTPREFIX)? (BASEUNIT | ALTUNIT | DERIVEDUNIT)
    
    DIGITS =
        [0-9]
    
    INTEGER =
        ('-' | '+')? DIGIT+
    
    NUMBER =
        ('-' | '+')? DIGIT* ('.' DIGIT*)? ('e' ('+' | '-') DIGIT*)?
    
    POWFRAC =
        INTEGER ('|' INTEGER)?
    
    POWERED =
        UNIT ('^' POWFRAC)?
    
    FACTOR =
        POWERED | NUMBER
    
    MULTI =
        FACTOR | (MULTI ' ' FACTOR)
    
    FRACTION =
        MULTI | (FRACTION '/' MULTI)

    """

    def __init__(self,
                 value: float | ArrayLike | str,
                 unit: str | None = None):
        if isinstance(value, Quantity):
            if unit is None:
                self.value = value.value
                self.code = value.code
            else:
                raise ValueError("Syntax error")
        else:
            if unit is None:
                self.value, self.code = _fracdecode(value)
            elif isinstance(unit, Quantity):
                self.value = value * unit.value
                self.code = unit.code
            else:
                mul, self.code = _fracdecode(unit)
                self.value = value * mul
        
    def definition(self, withoutvalue: bool = False) -> str:
        """Definition of stored value in SI units
        
        `definition()` returns the definition of the stored quantity
        in terms of SI base units.

        `definition(True)` returns only the base units without
        multiplying the value into it.

        """
        val = self.value
        ss = []
        for un, co in zip(_unitcode.keys(), self.code):
            if un=='g':
                val = val / 1000**co
                un = 'kg'
            if co==0:
                pass
            elif co==1:
                ss.append(un)
            elif co==int(co):
                ss.append(f'{un}^{int(co)}')
            else:
                ss.append(f'{un}^{co}')
        if not withoutvalue:
            ss.insert(0, f'{val}')
        return ' '.join(ss)

    def __add__(self, other: Self) -> Self:
        if np.all(other.code == self.code):
            qty = copy.copy(self)
            qty.value = self.value + other.value
            return qty
        else:
            raise ValueError("Incompatible units")

    def __sub__(self, other: Self) -> Self:
        if np.all(other.code == self.code):
            qty = copy.copy(self)
            qty.value = self.value - other.value
            qty.code = self.code
            return qty
        else:
            raise ValueError("Incompatible units")

    def __ge__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value >= other.value
        else:
            raise ValueError("Incompatible units")

    def __gt__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value > other.value
        else:
            raise ValueError("Incompatible units")
        
    def __le__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value <= other.value
        else:
            raise ValueError("Incompatible units")
        
    def __lt__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value < other.value
        else:
            raise ValueError("Incompatible units")

    def __eq__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value == other.value
        else:
            return False

    def __ne__(self, other: Self) -> bool:
        if np.all(other.code == self.code):
            return self.value != other.value
        else:
            return True

    def __neg__(self) -> Self:
        qty = copy.copy(self)
        qty.value = -self.value
        return qty

    def abs(self) -> Self:
        qty = copy.copy(self)
        qty.value = np.abs(self.value)
        return qty
    
        
    def __mul__(self, other: Self | float | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if type(self)==Quantity:
                qty = copy.copy(self)
                qty.value = self.value * other.value
                qty.code = self.code + other.code
            else:
                # drop special type
                qty = Quantity(self.value * other.value, '')
                qty.code = self.code + other.code
        else:
            qty = copy.copy(self)
            qty.value = self.value * other
        return qty

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Self | float | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if type(self) == Quantity:
                qty = copy.copy(self)
                qty.value = self.value / other.value
                qty.code = self.code - other.code
            else:
                # drop special type
                qty = Quantity(self.value / other.value, '')
                qty.code = self.code - other.code                
        else:
            qty = copy.copy(self)
            qty.value = self.value / other
        return qty

    def __rtruediv__(self, other: Self | float | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if type(self) == Quantity:
                qty = copy.copy(self)
                qty.value = other.value / self.value
                qty.code = other.code - self.code
            else:
                qty = Quantity(other.value / self.value, '')
                qty.code = other.code - self.code                
        else:
            # drop special type
            qty = Quantity(other / self.value, '')
            qty.code = -self.code
        return qty

    def __str__(self):
        return self.definition()

    def __repr__(self):
        return f'Quantity("{self.definition()}")'

    def plain(self, warn: bool = False) -> float | np.ndarray:
        """Convert to dimensionless
        
        `plain()` returns a dimensionless quantity as a plain number.
        
        An exception is raised if the units are incompatible. That is ::
       
            (10 * kHz * 5 * ms).plain()
        
        returns `50`, whereas ::

            (10 * V / s).plain()

        raises an exception.

        Optional argument `warn`, if True, causes a warning to be
        printed instead of an exception being raised.

        """
        return self.as_('1', warn)
                
    def as_(self, newunit: str | Self, warn: bool = False) -> float | np.ndarray:
        """Convert to different units
        
        `as_(newunits)` returns the numeric value of the stored
        quantity expressed in the new units.

        An exception is raised if the units are incompatible.
        Optional argument `warn`, if True, turns that into a warning.
        
        Note the underscore in the method name.

        Examples::

            Quantity("2 V").as_("mV") # -> 2000
            Quantity("1 minute").as_("s") # -> 60
            Quantity("5 V").as_("s") # raises exception

        """
        if isinstance(newunit, Quantity):
            newmul = newunit.value
            newcode = newunit.code
        else:
            newmul, newcode = _fracdecode(newunit)
        if np.any(self.code != newcode):
            if warn:
                oldunit = self.definition(True)
                log.warning(f"Incompatible units: {newunit} vs. {oldunit}")
            else:
                raise ValueError("Incompatible units")
        return self.value / newmul


class Time(Quantity):
    """Representation of a quantity that has units of time.

    The constructor checks that the constructed quantity does indeed
    have units of time. Other than that, this behaves just like the
    base class.

    """
    
    def __init__(self,
                 value: float | ArrayLike | str,
                 unit: str | None = None):
        super().__init__(value, unit)
        self.as_("s") # assert that we are time

    def __repr__(self):
        return f'Time("{self.definition()}")'
        
        
class Voltage(Quantity):
    """Representation of a quantity that has units of voltage.

    The constructor checks that the constructed quantity does indeed
    have units of voltage. Other than that, this behaves just like the
    base class.

    """
    def __init__(self,
                 value: float | ArrayLike | str,
                 unit: str | None = None):
        super().__init__(value, unit)
        self.as_("V") # assert that we are voltage

    def __repr__(self):
        return f'Voltage("{self.definition()}")'


class Frequency(Quantity):        
    """Representation of a quantity that has units of frequency.

    The constructor checks that the constructed quantity does indeed
    have units of frequency. Other than that, this behaves just like
    the base class.

    Example::

        Frequency(10/ms) # -> 10 kHz

    """
    def __init__(self,
                 value: float | ArrayLike | str,
                 unit: str | None = None):
        super().__init__(value, unit)
        self.as_("Hz") # assert that we are frequency

    def __repr__(self):
        return f'Frequency("{self.definition()}")'
        
        
V = Voltage(1, 'V')
"""The unit volt

This allows passing expressions like `5 * V` to definitions of stimuli
instead of the more cumbersome `Voltage(5, "V")`."""

mV = Voltage(1, 'mV')
"""The unit millivolt

Provided for convenience because `10 * mV` is easier to read
than `0.01 * V`.
"""

s = Time(1, 's')
"""The unit second

This allows passing expressions like `1.5 * s` to definitions of stimuli
instead of the more cumbersome `Time(15, "s")`.

"""

ms = Time(1, 'ms')
"""The unit millisecond

Provided for convenience because `5 * ms` is easier to read
than `0.005 * s`.

"""

Hz = Frequency(1, 'Hz')
"""The unit hertz"""

kHz = Frequency(1, 'kHz')
"""The unit kilohertz

This allows passing expressions like `30 * kHz` as a sampling rate
instead of the more cumbersome `Frequency("30 kHz")`.
"""
