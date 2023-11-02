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
import numpy as np

# Set global variables




# Functions


def _load_mesa_eos_table(settings:Settings, iverbose:int=3):
    """Load mesa table.

    Assuming mesa EoS table stored in directory settings['EoS_MESA_DATA_DIR'].
    Required keywords in settings:
        EoS_MESA_DATA_DIR
        EoS_MESA_table_Z_float
        EoS_MESA_table_Z_str
        EoS_MESA_table_X_float
        EoS_MESA_table_X_str

    Returns
    -------
    (
    mesa_table_Z_float : np.ndarray (1d),
    mesa_table_X_float : np.ndarray (1d),
    mesa_table_logE_float : np.ndarray (1d),
    mesa_table_logV_float : np.ndarray (1d),
    mesa_table_var2_float : np.ndarray (1d),
    ), mesa_table : np.ndarray (5d),
        
    """
    # init
    mesa_data_dir = settings['EoS_MESA_DATA_DIR']
    mesa_table_Z_float = settings['EoS_MESA_table_Z_float']
    mesa_table_Z_str   = settings['EoS_MESA_table_Z_str']
    mesa_table_X_float = settings['EoS_MESA_table_X_float']
    mesa_table_X_str   = settings['EoS_MESA_table_X_str']
    mesa_table_logE_float = None
    mesa_table_logV_float = None
    mesa_table_var2_float = None
    mesa_table = None
    no_Z = len(mesa_table_Z_float)
    no_X = len(mesa_table_X_float)
    
    if mesa_data_dir is None or not os.path.isdir(mesa_data_dir):
        raise ValueError(f"settings['MESA_DATA_DIR']={mesa_data_dir} is not a valid directory.")


    
    for i_Z, Z_float, Z_str in zip(range(no_Z), mesa_table_Z_float, mesa_table_Z_str):
        # sanity check
        if f'{Z_float:.2f}' != Z_str:
            warn(
                '_load_mesa_eos_table()', iverbose,
                f"{Z_float=} is not the same as {Z_str=}.",
            )
        for i_X, X_float, X_str in zip(range(no_X), mesa_table_X_float, mesa_table_X_str):
            # sanity check
            if f'{X_float:.2f}' != X_str:
                warn(
                    '_load_mesa_eos_table()', iverbose,
                    f"{X_float} is not the same as {X_str=}.",
                )
            with open(f"{mesa_data_dir}{os.path.sep}output_DE_z{Z_str}x{X_str}.bindata", 'rb') as f:
                
                no_E, no_V, no_var2 = fortran_read_file_unformatted(f, 'i', 3)
                
                logV_float = fortran_read_file_unformatted(f, 'd', no_V)
                if mesa_table_logV_float is None:
                    mesa_table_logV_float = logV_float
                elif not np.allclose(mesa_table_logV_float, logV_float):
                    warn(
                        '_load_mesa_eos_table()', iverbose,
                        "Warning: logV array not the same across the data files.",
                        "This is not supposed to happen! Check the code and the data!",
                    )

                logE_float = fortran_read_file_unformatted(f, 'd', no_E)
                if mesa_table_logE_float is None:
                    mesa_table_logE_float = logE_float
                elif not np.allclose(mesa_table_logE_float, logE_float):
                    warn(
                        '_load_mesa_eos_table()', iverbose,
                        "Warning: logE array not the same across the data files.",
                        "This is not supposed to happen! Check the code and the data!",
                    )

                if mesa_table is None:
                    # init mesa_table
                    mesa_table = np.full((no_Z, no_X, no_E, no_V, no_var2), np.nan, dtype=np.float64)

                for i_V in range(no_V):
                    for i_E in range(no_E):
                        mesa_table[i_Z, i_X, i_E, i_V] = fortran_read_file_unformatted(f, 'd', no_var2)
                    
            
        
        
    
    
    #raise NotImplementedError
    return (
        mesa_table_Z_float,
        mesa_table_X_float,
        mesa_table_logE_float,
        mesa_table_logV_float,
        mesa_table_var2_float,
        ), mesa_table



# Classes


class EoS_MESA(EoS_Base):
    """Class for MESA Equation of State Objects."""
    def __init__(self, settings:Settings=DEFAULT_SETTINGS):
        self.__mesa_table = _load_mesa_eos_table(settings=settings)



