#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module to deal with MESA EoS (from its table).

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info
from .settings import SETTINGS

#  import (general)
from os.path import sep


# Set global variables
MESA_DATA_DIR = SETTINGS['MESA_DATA_DIR']





