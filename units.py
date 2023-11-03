#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for handling unit conversion.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)

#  import (general)
from astropy import units
from astropy import constants as const




# Functions



# astropy.quantity related

def set_as_quantity(var, unit, equivalencies=[], copy=True):
    """Convert the var to an astropy quantity with given unit."""
    if issubclass(type(var), units.quantity.Quantity):
        var = var.to(unit, equivalencies=equivalencies, copy=copy)
    else:
        var = units.Quantity(var, unit=unit)
    return var


def set_as_quantity_temperature(var, unit=units.Kelvin, copy=True):
    """Convert the var to an astropy quantity with given unit."""
    return set_as_quantity(var, unit=unit, equivalencies=units.equivalencies.temperature(), copy=copy)



def complete_units_dict(base_units: dict, iverbose: int=3) -> dict:
    """Complete base_units dict using its mass, dist, time, & temp.
    Will write to base_units so be careful.
    """
    base_units['speed'] = base_units['dist'] / base_units['time']
    base_units['energy'] = base_units['mass'] * base_units['speed']**2
    base_units['specificEnergy'] = base_units['energy'] / base_units['mass']
    base_units['lum'] = base_units['energy'] / base_units['time']
    base_units['flux'] = base_units['lum'] / base_units['dist']**2
    base_units['density'] = base_units['mass'] / base_units['dist']**3
    base_units['opacity'] = base_units['dist']**2 / base_units['mass']
    base_units['G'] = base_units['dist']**3 / ( base_units['mass'] * base_units['time']**2 )
    base_units['sigma_sb'] = base_units['lum'] / base_units['dist']**2 / base_units['temp']**4
    return base_units



# default units (not really used much, just for fun)

DEFAULT_UNITS = {
    'dist': units.R_sun,
    'mass': units.M_sun,
    'temp': units.K,
}
# define the unit of time such that G is 1 in the new unit system
DEFAULT_UNITS['time'] = units.def_unit(
    "unit_time",
    ( ( DEFAULT_UNITS['dist']**3 / (DEFAULT_UNITS['mass'] * const.G) )**0.5 ).to(units.s),
)
DEFAULT_UNITS = complete_units_dict(DEFAULT_UNITS)


CGS_UNITS = {
    'dist': units.cm,
    'mass': units.g,
    'time': units.s,
    'temp': units.K,
}
CGS_UNITS = complete_units_dict(CGS_UNITS)