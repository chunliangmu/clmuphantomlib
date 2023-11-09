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





# Functions and global constants



# astropy.quantity related

def set_as_quantity(
    var  : tuple | list | np.ndarray | units.Quantity,
    unit : units.Unit | None = None,
    equivalencies: list = [],
    copy : bool=True,
) -> units.Quantity:
    """Convert the var to an astropy quantity with given unit."""
    if issubclass(type(var), units.Quantity):
        var = var.to(unit, equivalencies=equivalencies, copy=copy)
    else:
        var = units.Quantity(var, unit=unit)
    return var


def set_as_quantity_temperature(
    var  : tuple | list | np.ndarray | units.Quantity,
    unit : units.Unit = units.Kelvin,
    copy : bool = True,
) -> units.Quantity:
    """Convert the var to an astropy quantity with given unit."""
    return set_as_quantity(var, unit=unit, equivalencies=units.equivalencies.temperature(), copy=copy)


def get_val_in_unit(
    var      : tuple | list | np.ndarray | units.Quantity,
    unit     : units.Unit | None = None,
    unit_new : units.Unit | None = None,
    equivalencies: list = [],
    copy     : bool = True,
) -> np.ndarray | np.float64:
    """Convert the var from one unit to a new given unit.

    No need to supply 'unit' if var is already an astropy quantity; in which case, should have no effect even if supplied.
    """
    return set_as_quantity(var, unit, equivalencies=equivalencies, copy=copy).to_value(unit_new)


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


def get_units_field_name(val_name: str) -> str:
    """Translate sarracen data frame column name into units field name,
    as shown in complete_units_dict().
    """
    if val_name == 'rho':
        return 'density'
    elif val_name in ['u']:
        return 'specificEnergy'
    elif val_name in ['T', 'temp', 'temperature', 'Tdust', 'Tgas']:
        return 'temp'
    elif val_name in ['m', 'mass']:
        return 'mass'
    elif val_name in ['x', 'y', 'z']:
        return 'dist'
    elif val_name in ['t', 'time']:
        return 'time'
    elif val_name in ['v', 'vx', 'vy', 'vz']:
        return 'speed'
    elif val_name in ['kappa', 'opacity']:
        return 'opacity'
    else:
        raise NotImplementedError



# default units (not really used much, just for fun)

__DEFAULT_UNITS = {
    'dist': units.R_sun,
    'mass': units.M_sun,
    'temp': units.K,
}
# define the unit of time such that G is 1 in the new unit system
__DEFAULT_UNITS['time'] = units.def_unit(
    "unit_time",
    ( ( __DEFAULT_UNITS['dist']**3 / (__DEFAULT_UNITS['mass'] * const.G) )**0.5 ).to(units.s),
)
__DEFAULT_UNITS = complete_units_dict(__DEFAULT_UNITS)
DEFAULT_UNITS = __DEFAULT_UNITS.copy()


__CGS_UNITS = {
    'dist': units.cm,
    'mass': units.g,
    'time': units.s,
    'temp': units.K,
}
__CGS_UNITS = complete_units_dict(__CGS_UNITS)
CGS_UNITS = __CGS_UNITS.copy()



def get_units_cgs(val_name: str) -> units.core.Unit:
    """Get cgs units for a type of values."""
    return __CGS_UNITS[get_units_field_name(val_name)]
    