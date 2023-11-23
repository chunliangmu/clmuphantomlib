#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for handling Equation of State (EoS) related stuff.

E.g. getting temperature (T) from density (rho) and internal energy (u).

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info
from .settings import Settings, DEFAULT_SETTINGS
from .eos_base import EoS_Base
from .eos_mesa import EoS_MESA

#  import (general)

def get_eos(ieos: int, params: dict, settings: Settings=DEFAULT_SETTINGS, verbose: int=3) -> EoS_Base:
    """Get an EoS object, which you can use to get temp etc values fro rho and u."""
    if ieos == 10:
        return EoS_MESA(params, settings, verbose)
    else:
        error('get_eos()', verbose, f"Unrecognized ieos={ieos}.")
        raise NotImplementedError



