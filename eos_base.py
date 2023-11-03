#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for handling Equation of State (EoS) related stuff.
Defines the base class for EoS.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info
from .settings import Settings, DEFAULT_SETTINGS

#  import (general)
import numpy as np



# Classes


class EoS_Base:
    """Base Class for Equation of State Objects."""
    def __init__(self, params: dict, settings: Settings=DEFAULT_SETTINGS, iverbose: int=3):
        note('EoS_Base', iverbose, "Loading EoS_Base.")
        return

    def get_temp(self, rho: np.ndarray, u: np.ndarray):
        raise NotImplementedError


