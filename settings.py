#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module defining settings for the code.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info

#  import (general)
import os
from os import sep
import numpy as np





class Settings:
    """A structure for storing settings.
    
    Using a class and private vars to make sure I don't stupidly overwrite it.
    will set __getitem__() but not __setitem__()
    """
    def __init__(self, set_as:dict|str="default", iverbose:int=3):
        """Init.
        """
        self.__data = {
            'PHANTOM_DIR': None,    # assuming normalized path ( use os.path.normpath() )
            'EoS_MESA_DATA_DIR': None,    # assuming normalized path ( use os.path.normpath() )
            'EoS_MESA_table_Z_float': np.array([]),
            'EoS_MESA_table_Z_str': [],
            'EoS_MESA_table_X_float': np.array([]),
            'EoS_MESA_table_X_str': [],
            'EoS_MESA_table_dtype': [],
            
        }

        if set_as == "default":
            self.set_as_default(iverbose=iverbose)
        elif isinstance(set_as, dict):
            raise NotImplementedError
        return
    
    def __getitem__(self, i):
        return self.__data[i]

    
    def __setitem__(self, i):
        raise NotImplementedError

    
    def __str__(self):
        return self.__data.__str__()

    
    def set_as_default(self, iverbose=3):
        self.__data['PHANTOM_DIR'] = os.getenv('PHANTOM_DIR')
        if isinstance(self.__data['PHANTOM_DIR'], str):
            self.__data['PHANTOM_DIR']  = os.path.normpath(self.__data['PHANTOM_DIR'])
            self.__data['EoS_MESA_DATA_DIR']= f"{self.__data['PHANTOM_DIR']}{sep}data{sep}eos{sep}mesa" 
        else:
            if self.__data['PHANTOM_DIR'] is not None:
                warn("Settings.set_as_default()", iverbose, f"Unrecognized env variable PHANTOM_DIR={self.__data['PHANTOM_DIR']}")
            self.__data['EoS_MESA_DATA_DIR'] = None
            
        self.__data['EoS_MESA_table_Z_float'] = np.array([0.00, 0.02, 0.04])
        self.__data['EoS_MESA_table_X_float'] = np.array([0.00, 0.20, 0.40, 0.60, 0.80])
        self.__data['EoS_MESA_table_dtype'] = [
            ('log10_rho'    , np.float64), #  1
            ('log10_P'      , np.float64), #  2
            ('log10_Pgas'   , np.float64), #  3
            ('log10_T'      , np.float64), #  4
            ('dlnP/dlnrho|e', np.float64), #  5
            ('dlnP/dlne|rho', np.float64), #  6
            ('dlnT/dlnrho|e', np.float64), #  7
            ('dlnT/dlne|rho', np.float64), #  8
            ('log10_S'      , np.float64), #  9
            ('dlnT/dlnP|S'  , np.float64), # 10
            ('Gamma1'       , np.float64), # 11
            ('gamma'        , np.float64), # 12
        ]
        self.normalize(force=True)

    
    def normalize(self, force=True, iverbose=3):
        """Post processing info stored in self."""

        for elem in ['X', 'Z']:
            # elem means element
            do_EoS_MESA_table_elem = f'EoS_MESA_table_{elem}_float' in self.__data.keys() and (
                force or f'EoS_MESA_table_{elem}_str' not in self.__data.keys() or self.__data[f'EoS_MESA_table_{elem}_str']
            )
            if do_EoS_MESA_table_elem:
                self.__data[f'EoS_MESA_table_{elem}_str'] = [
                    f'{it:.2f}'
                    for it in self.__data[f'EoS_MESA_table_{elem}_float']
                ]


DEFAULT_SETTINGS = Settings(set_as="default")
