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





class Settings:
    """A structure for storing settings.
    
    Using a class and private vars to make sure I don't stupidly overwrite it.
    will set __getitem__() but not __setitem__()
    """
    def __init__(self, as_default:bool=True, iverbose:int=3):
        """"""
        self.__data = {
            'PHANTOM_DIR': None,
            'MESA_DATA_DIR': None,
        }

        if as_default:
            self.set_as_default(iverbose=iverbose)

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
            self.__data['MESA_DATA_DIR']= f"{self.__data['PHANTOM_DIR']}{os.path.sep}data{os.path.sep}eos{os.path.sep}mesa" 
        else:
            if self.__data['PHANTOM_DIR'] is not None:
                warn("Settings.set_as_default()", iverbose, f"Unrecognized env variable PHANTOM_DIR={self.__data['PHANTOM_DIR']}")
            self.__data['MESA_DATA_DIR'] = None



SETTINGS = Settings()
