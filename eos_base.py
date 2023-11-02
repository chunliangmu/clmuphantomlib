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

#  import (general)
import numpy as np



# Classes


class EoS_Base:
    def __init__(self):
        pass

    def get_temp_from_rho_u(self, rho: np.ndarray, u: np.ndarray):
        raise NotImplementedError


