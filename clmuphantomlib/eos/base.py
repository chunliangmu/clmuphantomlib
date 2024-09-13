#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for handling Equation of State (EoS) related stuff.
Defines the base class for EoS.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from ..log import is_verbose, say
from ..settings import Settings, DEFAULT_SETTINGS
from ..units_util import set_as_quantity, get_units_cgs

#  import (general)
import numpy as np
from astropy import units



# Classes


class EoS_Base:
    """Base Class for Equation of State Objects."""
    def __init__(self, params: dict, settings: Settings=DEFAULT_SETTINGS, verbose: int=3):
        if is_verbose(verbose, 'note'):
            say('note', None, verbose, "Loading EoS_Base.")
        return

    def get_val_cgs(
        self,
        val_name: str,
        rho: np.ndarray|units.Quantity,
        u  : np.ndarray|units.Quantity,
        *params_list,
        verbose: int = 3,
        **params_dict,
    ) -> np.ndarray:
        """Virtual function for getting values from EoS and rho and u in cgs units.


        *** Please implement this func in specific EoS classes. ***


        See self.get_val_cgs() for more info.
        
        Parameters
        ----------
        val_name: str
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, u: np.ndarray | units.Quantity
            density and specific internal energy in cgs units.
            should have same shape.

        return_as_quantity: bool | None
            if the results should be returned as a astropy.units.Quantity.
            If None, will only return as that if one of the input rho or u is that.
            
        verbose: int
            How much errors, warnings, notes, and debug info to be print on screen.


        ... and other params specified by specific EoS (see their respective docs.)


        Returns
        -------
        ans: np.ndarray | units.Quantity
            calc-ed EoS values.
        """
        if is_verbose(verbose, 'debug'):
            say('debug', None, verbose, "Calling EoS_Base.get_val_cgs().")
        raise NotImplementedError


    
    def get_val(
        self,
        val_name: str,
        rho: np.ndarray|units.Quantity,
        u  : np.ndarray|units.Quantity,
        *params_list,
        return_as_quantity: bool|None = None,
        verbose: int = 3,
        **params_dict,
    ) -> np.ndarray|units.Quantity:
        """Getting specific values from EoS and rho and u.

        See self.get_val_cgs() for more info.
        
        Parameters
        ----------
        val_name: str
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, u: np.ndarray | units.Quantity
            density and specific internal energy
            if numpy array, WILL ASSUME CGS UNITS.
            should have same shape.

        return_as_quantity: bool | None
            if the results should be returned as a astropy.units.Quantity.
            If None, will only return as that if one of the input rho or u is that.
            
        verbose: int
            How much errors, warnings, notes, and debug info to be print on screen.


        ... and other params specified by specific EoS (see their respective docs.)


        Returns
        -------
        ans: np.ndarray | units.Quantity
            calc-ed EoS values.
        
        """
        if is_verbose(verbose, 'debug'):
            say('debug', None, verbose, "Calling EoS_Base.get_val().")
        return_quantity = False
        if isinstance(rho, units.Quantity):
            rho = rho.cgs.value
            return_quantity = True
        if isinstance(u, units.Quantity):
            u = u.cgs.value
            return_quantity = True
        if return_as_quantity is not None:
            return_quantity = return_as_quantity
        
        ans = self.get_val_cgs(
            val_name, rho, u, *params_list,
            return_as_quantity=return_as_quantity, verbose=verbose,
            **params_dict,
        )

        if return_quantity:
            ans = set_as_quantity(ans, get_units_cgs(val_name))
        return ans
        

    

    def get_temp(
        self,
        rho: np.ndarray|units.Quantity,
        u  : np.ndarray|units.Quantity,
        *params_list,
        return_as_quantity: bool|None = None,
        verbose: int = 3,
        **params_dict
    ):
        """Getting temperature from EoS and rho and u.
        
        
        See self.get_val_cgs() for more info.
        """
        if is_verbose(verbose, 'debug'):
            say('debug', None, verbose, "Calling EoS_Base.get_temp().")
        return self.get_val(
            'T', rho, u, *params_list,
            return_as_quantity=return_as_quantity, verbose=verbose,
            **params_dict,
        )


