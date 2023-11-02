#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module to deal with MESA EoS (from its table).

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info
from .settings  import DEFAULT_SETTINGS
from .readwrite import fortran_read_file_unformatted
from .eos_base  import EoS_Base

#  import (general)
from os.path import sep


# Set global variables
mesa_data_dir = DEFAULT_SETTINGS['MESA_DATA_DIR']



# Classes


class EoS_MESA(EoS_Base):
    def __init__(self):
        pass





