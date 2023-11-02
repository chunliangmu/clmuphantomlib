#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module to deal with MESA EoS (from its table).

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info
from .settings  import Settings, DEFAULT_SETTINGS
from .readwrite import fortran_read_file_unformatted
from .eos_base  import EoS_Base

#  import (general)
import os


# Set global variables




# Functions


def _load_mesa_eos_table(settings:Settings):
    """Load mesa table.

    Assuming mesa EoS table stored in directory settings['MESA_DATA_DIR'].
    """
    # init
    mesa_data_dir = settings['EoS_MESA_DATA_DIR']
    if mesa_data_dir is None or not os.path.isdir(mesa_data_dir):
        raise ValueError(f"settings['MESA_DATA_DIR']={mesa_data_dir} is not a valid directory.")
    
    mesa_table = None
    
    raise NotImplementedError
    return mesa_table



# Classes


class EoS_MESA(EoS_Base):
    """Class for MESA Equation of State Objects."""
    def __init__(self, settings:Settings=DEFAULT_SETTINGS):
        self.__mesa_table = _load_mesa_eos_table(settings=settings)



