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
        EoS_MESA_table_dtype

    Note: data stored in MESA EoS table (var2- last column):
    (as stated in phantom source file comments: phantom/src/main/eos_mesa_microphysics.f90 line 435-438)
    ! The columns in the data are:
    ! 1. logRho        2. logP          3. logPgas       4. logT
    ! 5. dlnP/dlnrho|e 6. dlnP/dlne|rho 7. dlnT/dlnrho|e 8. dlnT/dlne|rho
    ! 9. logS         10. dlnT/dlnP|S  11. Gamma1       12. gamma
    

    Returns
    -------
    (
    mesa_table_Z_float : np.ndarray (1d),    # Z
    mesa_table_X_float : np.ndarray (1d),    # X
    mesa_table_logE_float : np.ndarray (1d), # Eint
    mesa_table_logV_float : np.ndarray (1d), # V
    ), mesa_table : np.ndarray (5d),
        
    """
    # init
    mesa_data_dir = settings['EoS_MESA_DATA_DIR']
    mesa_table_Z_float = settings['EoS_MESA_table_Z_float']
    mesa_table_Z_str   = settings['EoS_MESA_table_Z_str']
    mesa_table_X_float = settings['EoS_MESA_table_X_float']
    mesa_table_X_str   = settings['EoS_MESA_table_X_str']
    mesa_table_dtype   = settings['EoS_MESA_table_dtype']
    mesa_table_logE_float = None
    mesa_table_logV_float = None
    mesa_table_var2_str = None
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

                # load meta data
                no_E, no_V, no_var2 = fortran_read_file_unformatted(f, 'i', 3)
                
                logV_float = np.array(fortran_read_file_unformatted(f, 'd', no_V))
                if mesa_table_logV_float is None:
                    mesa_table_logV_float = logV_float
                elif not np.allclose(mesa_table_logV_float, logV_float):
                    warn(
                        '_load_mesa_eos_table()', iverbose,
                        "Warning: logV array not the same across the data files.",
                        "This is not supposed to happen! Check the code and the data!",
                    )

                logE_float = np.array(fortran_read_file_unformatted(f, 'd', no_E))
                if mesa_table_logE_float is None:
                    mesa_table_logE_float = logE_float
                elif not np.allclose(mesa_table_logE_float, logE_float):
                    warn(
                        '_load_mesa_eos_table()', iverbose,
                        "Warning: logE array not the same across the data files.",
                        "This is not supposed to happen! Check the code and the data!",
                    )

                # init mesa_table as a structured array
                if mesa_table is None:
                    mesa_table = np.full((no_Z, no_X, no_E, no_V), np.nan, dtype=mesa_table_dtype)

                # fill mesa table
                for i_V in range(no_V):
                    for i_E in range(no_E):
                        mesa_table[i_Z, i_X, i_E, i_V] = fortran_read_file_unformatted(f, 'd', no_var2)
                    
            
        
        
    
    
    #raise NotImplementedError
    return (
        mesa_table_Z_float,
        mesa_table_X_float,
        mesa_table_logE_float,
        mesa_table_logV_float,
        ), mesa_table




class EoS_MESA_table:
    """A class to store and handle stored MESA tables."""
    def __init__(self, settings:Settings, iverbose:int=3):
        self.data_dir    = ""
        self.Z_arr       = np.array([])
        self.Z_str       = []
        self.X_arr       = np.array([])
        self.X_str       = []
        self.logE_arr    = np.array([])
        self.logV_arr    = np.array([])
        self.table       = np.array([])
        self.table_dtype = []

        self.load_mesa_eos_table(settings, iverbose=iverbose)
        return


    
    def load_mesa_eos_table(self, settings:Settings, iverbose:int=3):
        """Load MESA table.
    
        Assuming mesa EoS table stored in directory settings['EoS_MESA_DATA_DIR'].
        Required keywords in settings:
            EoS_MESA_DATA_DIR
            EoS_MESA_table_Z_float
            EoS_MESA_table_Z_str
            EoS_MESA_table_X_float
            EoS_MESA_table_X_str
            EoS_MESA_table_dtype
    
        Note: data stored in MESA EoS table (var2- last column):
        (as stated in phantom source file comments: phantom/src/main/eos_mesa_microphysics.f90 line 435-438)
        ! The columns in the data are:
        ! 1. logRho        2. logP          3. logPgas       4. logT
        ! 5. dlnP/dlnrho|e 6. dlnP/dlne|rho 7. dlnT/dlnrho|e 8. dlnT/dlne|rho
        ! 9. logS         10. dlnT/dlnP|S  11. Gamma1       12. gamma
        
    
        Returns self.
            
        """
        # init
        self.data_dir    = settings['EoS_MESA_DATA_DIR']
        self.Z_arr       = settings['EoS_MESA_table_Z_float']
        self.Z_str       = settings['EoS_MESA_table_Z_str']
        self.X_arr       = settings['EoS_MESA_table_X_float']
        self.X_str       = settings['EoS_MESA_table_X_str']
        self.table_dtype = settings['EoS_MESA_table_dtype']
        self.logE_arr    = None
        self.logV_arr    = None
        self.table       = None
        no_Z = len(self.Z_arr)
        no_X = len(self.X_arr)
        
        if self.data_dir is None or not os.path.isdir(self.data_dir):
            raise ValueError(f"settings['MESA_DATA_DIR']={self.data_dir} is not a valid directory.")
    
    
        
        for i_Z, Z_float, Z_str in zip(range(no_Z), self.Z_arr, self.Z_str):
            # sanity check
            if f'{Z_float:.2f}' != Z_str:
                warn(
                    '_load_mesa_eos_table()', iverbose,
                    f"{Z_float=} is not the same as {Z_str=}.",
                )
            for i_X, X_float, X_str in zip(range(no_X), self.X_arr, self.X_str):
                # sanity check
                if f'{X_float:.2f}' != X_str:
                    warn(
                        '_load_mesa_eos_table()', iverbose,
                        f"{X_float} is not the same as {X_str=}.",
                    )
                with open(f"{self.data_dir}{os.path.sep}output_DE_z{Z_str}x{X_str}.bindata", 'rb') as f:
    
                    # load meta data
                    no_E, no_V, no_var2 = fortran_read_file_unformatted(f, 'i', 3)
                    
                    logV_float = np.array(fortran_read_file_unformatted(f, 'd', no_V))
                    if self.logV_arr is None:
                        self.logV_arr = logV_float
                    elif not np.allclose(self.logV_arr, logV_float):
                        warn(
                            '_load_mesa_eos_table()', iverbose,
                            "Warning: logV array not the same across the data files.",
                            "This is not supposed to happen! Check the code and the data!",
                        )
    
                    logE_float = np.array(fortran_read_file_unformatted(f, 'd', no_E))
                    if self.logE_arr is None:
                        self.logE_arr = logE_float
                    elif not np.allclose(self.logE_arr, logE_float):
                        warn(
                            '_load_mesa_eos_table()', iverbose,
                            "Warning: logE array not the same across the data files.",
                            "This is not supposed to happen! Check the code and the data!",
                        )
    
                    # init mesa_table as a structured array
                    if self.table is None:
                        self.table = np.full((no_Z, no_X, no_E, no_V), np.nan, dtype=self.table_dtype)
    
                    # fill mesa table
                    for i_V in range(no_V):
                        for i_E in range(no_E):
                            self.table[i_Z, i_X, i_E, i_V] = fortran_read_file_unformatted(f, 'd', no_var2)

        #raise NotImplementedError
        return self









# Classes


class EoS_MESA(EoS_Base):
    """Class for MESA Equation of State Objects."""
    def __init__(self, settings:Settings=DEFAULT_SETTINGS, iverbose:int=3):
        self.__mesa_table = _load_mesa_eos_table(settings=settings)

        return



