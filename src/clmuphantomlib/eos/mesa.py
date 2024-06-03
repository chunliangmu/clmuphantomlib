#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module to deal with MESA EoS (from its table).

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from ..log import error, warn, note, debug_info
from ..settings   import Settings, DEFAULT_SETTINGS
from ..io  import fortran_read_file_unformatted
from ..units_util import set_as_quantity, get_units_cgs
from .base import EoS_Base

#  import (general)
import os
import numpy as np
from astropy import units
from scipy.interpolate import RegularGridInterpolator



# Classes






class _EoS_MESA_table_opacity:
    """A class to store and handle stored MESA opacity tables from Phantom."""
    def __init__(self, params: dict, settings: Settings = DEFAULT_SETTINGS, verbose: int=3):
        self._data_dir    = ""
        self._Z           = np.nan
        self._Z_arr       = np.array([])
        self._X           = np.nan
        self._X_arr       = np.array([])
        self._log10_R_arr = np.array([])
        self._log10_T_arr = np.array([])
        self._grid_withZX = ()
        self._table_withZX= np.array([])
        self._grid_noZX   = ()
        self._table_noZX  = np.array([])
        self._table_dtype = []
        self._interp      = None
        
        self.load_mesa_table(params, settings, verbose=verbose)
        return

    
    def load_mesa_table(self, params: dict, settings: Settings, verbose: int=3):
        """Load MESA table.
    
        Assuming mesa EoS table stored in directory settings['EoS_MESA_DATA_DIR'].
        Required keywords in settings:
            EoS_MESA_DATA_DIR
            EoS_MESA_table_dtype
    
        Required Keywords in params:
            X: Hydrogen mass fraction
            Z: Metallicity
        (Because Phantom assumes a constant pre-determined X and Z when loading MESA EoS-
         it does linear interpolation on that.)

        Using Z, X, T (temperature), R ($\log_{10}{R} \equiv \log_{10}{\rho} + 18 - 3 \log_{10}{T}$)
        
    
        Returns self.
            
        """
        # init
        self._data_dir    = settings['EoS_MESA_DATA_DIR']
        self._Z_arr       = None #settings['EoS_MESA_table_Z_float']
        self._X_arr       = None #settings['EoS_MESA_table_X_float']
        self._table_dtype = settings['EoS_MESA_table_dtype']
        self._Z           = params['Z']
        self._X           = params['X']
        self._log10_R_arr = None
        self._log10_T_arr = None
        self._table_withZX= None
        
        if self._data_dir is None or not os.path.isdir(self._data_dir):
            raise ValueError(f"settings['MESA_DATA_DIR']={self._data_dir} is not a valid directory.")

        with open(f"{self._data_dir}{os.path.sep}opacs.bindata", 'rb') as f:
            no_Z, no_X, no_R, no_T = fortran_read_file_unformatted(f, 'i', 4)
            # as for why does the order of z-x-r-t changed to z-x-t-r, I have zero idea
            self._Z_arr       = np.array(fortran_read_file_unformatted(f, 'd', no_Z))
            self._X_arr       = np.array(fortran_read_file_unformatted(f, 'd', no_X))
            self._log10_T_arr = np.array(fortran_read_file_unformatted(f, 'd', no_T))
            self._log10_R_arr = np.array(fortran_read_file_unformatted(f, 'd', no_R))
            
            # init mesa_table
            self._table_withZX = np.full((no_Z, no_X, no_T, no_R), np.nan)

            # fill mesa table
            for i_Z in range(no_Z):
                for i_X in range(no_X):
                    for i_t in range(no_T):
                        self._table_withZX[i_Z, i_X, i_t] = fortran_read_file_unformatted(f, 'd', no_R)
            self._table_withZX = np.swapaxes(self._table_withZX, 2, 3)
            
            self._grid_withZX = (self._Z_arr, self._X_arr, self._log10_R_arr, self._log10_T_arr)
            self._grid_noZX   = (self._log10_R_arr, self._log10_T_arr)
            meshgrid_noZX = np.meshgrid(*self._grid_noZX, indexing='ij')
            meshgrid_noZX_coord = np.stack((
                np.full(meshgrid_noZX[0].shape, self._Z),
                np.full(meshgrid_noZX[0].shape, self._X),
                *meshgrid_noZX),
                axis=-1)
            self._table_noZX = RegularGridInterpolator(
                self._grid_withZX, self._table_withZX, method='linear',
            )(meshgrid_noZX_coord)

            self._interp = RegularGridInterpolator(
                self._grid_noZX, self._table_noZX, method='linear', bounds_error=False, fill_value=None)
            self._interp_no_extrap = RegularGridInterpolator(
                self._grid_noZX, self._table_noZX, method='linear', bounds_error=False, fill_value=np.nan)
            # derivatives- ignored
        return self


    
    def get_kappa_cgs(
        self,
        rho: np.ndarray,
        T  : np.ndarray,
        *params_list,
        method   : str|None = None,
        do_extrap: bool = False,
        verbose : int  = 3,
        **params_dict,
    ) -> np.ndarray:
        """Interpolate value and return.
        
        Parameters
        ----------
        val_name: str | list
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, T: np.ndarray
            density and specific internal energy in cgs units.

        method: str | None
            Method to be used for interpolation.
            See scipy.interpolate.RegularGridInterpolator.__call__ docs.

        do_extrap: bool
            If true, will extrapolate
            otherwise, will return nan when out of bounds
            *** Note: There will be NO WARNINGS if True! ***

        verbose: int
            How much errors, warnings, notes, and debug info to be print on screen.

            

        Note: As described by line 451 of the file phantom/src/main/eos_mesa_microphysics.f90,
            ! logRho = logV + 0.7*logE - 20
            Daniel's explanation of it is that
            "It’s some combination of pressure and density that makes sense only to people who compile equations of state"
            (2023-11-02 over Slack.)
            
        """

        log10_T = np.log10(T)
        log10_R = np.log10(rho) + 18. - 3. * log10_T
        _interp_coord = (log10_R, log10_T)
        
        if do_extrap:
            _interp = self._interp
        else:
            _interp = self._interp_no_extrap
        
        return 10**_interp(_interp_coord, method=method)











class EoS_MESA_opacity(_EoS_MESA_table_opacity):
    """Wrapper for MESA opacity class"""

    # do not define __init__ so the old __init__() from parent class is called


    def get_kappa(
        self,
        rho: np.ndarray|units.Quantity,
        T  : np.ndarray|units.Quantity,
        *params_list,
        return_as_quantity: bool|None = None,
        verbose: int = 3,
        **params_dict,
    ) -> np.ndarray|units.Quantity:
        """Getting specific values from EoS and rho and u.

        See self.get_val_cgs() for full details of possible parameters.
        
        Parameters
        ----------
        rho, T: np.ndarray | units.Quantity
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
        return_quantity = False
        if isinstance(rho, units.Quantity):
            rho = rho.cgs.value
            return_quantity = True
        if isinstance(T, units.Quantity):
            T = T.cgs.value
            return_quantity = True
        if return_as_quantity is not None:
            return_quantity = return_as_quantity
        
        ans = self.get_kappa_cgs(
            rho, T, *params_list,
            verbose=verbose,
            **params_dict,
        )

        if return_quantity:
            ans = set_as_quantity(ans, get_units_cgs('kappa'))
        return ans














class _EoS_MESA_table:
    """A class to store and handle stored MESA EoS tables from Phantom."""
    def __init__(self, params: dict, settings: Settings, verbose: int=3):
        self._data_dir    = ""
        self._Z           = np.nan
        self._Z_arr       = np.array([])
        self._Z_str       = []
        self._X           = np.nan
        self._X_arr       = np.array([])
        self._X_str       = []
        self._log10_E_arr = np.array([])
        self._log10_V_arr = np.array([])
        self._grid_withZX = ()
        self._table_withZX= np.array([])
        self._grid_noZX   = ()
        self._table_noZX  = np.array([])
        self._table_dtype = []
        self._interp_dict = {}
        
        self.load_mesa_eos_table(params, settings, verbose=verbose)
        return

    
    def __getitem__(self, i):
        return self._table_withZX[i]

    
    def __setitem__(self, i, val):
        raise NotImplementedError(
            "_EoS_MESA_table: Please don't try to change things that are not supposed to be changed.")
        
    
    def load_mesa_eos_table(self, params: dict, settings: Settings, verbose: int=3):
        """Load MESA table.
    
        Assuming mesa EoS table stored in directory settings['EoS_MESA_DATA_DIR'].
        Required keywords in settings:
            EoS_MESA_DATA_DIR
            EoS_MESA_table_Z_float
            EoS_MESA_table_Z_str
            EoS_MESA_table_X_float
            EoS_MESA_table_X_str
            EoS_MESA_table_dtype

        Required Keywords in params:
            X: Hydrogen mass fraction
            Z: Metallicity
        (Because Phantom assumes a constant pre-determined X and Z when loading MESA EoS-
         it does linear interpolation on that.)
    
        Note: data stored in MESA EoS table (var2- last column):
        (as stated in phantom source file comments: phantom/src/main/eos_mesa_microphysics.f90 line 435-438)
        ! The columns in the data are:
        ! 1. logRho        2. logP          3. logPgas       4. logT
        ! 5. dlnP/dlnrho|e 6. dlnP/dlne|rho 7. dlnT/dlnrho|e 8. dlnT/dlne|rho
        ! 9. logS         10. dlnT/dlnP|S  11. Gamma1       12. gamma
        
    
        Returns self.
            
        """
        # init
        self._data_dir    = settings['EoS_MESA_DATA_DIR']
        self._Z_arr       = settings['EoS_MESA_table_Z_float']
        self._Z_str       = settings['EoS_MESA_table_Z_str']
        self._X_arr       = settings['EoS_MESA_table_X_float']
        self._X_str       = settings['EoS_MESA_table_X_str']
        self._table_dtype = settings['EoS_MESA_table_dtype']
        self._Z           = params['Z']
        self._X           = params['X']
        self._log10_E_arr = None
        self._log10_V_arr = None
        self._table_withZX= None
        no_Z = len(self._Z_arr)
        no_X = len(self._X_arr)
        
        if self._data_dir is None or not os.path.isdir(self._data_dir):
            raise ValueError(f"settings['MESA_DATA_DIR']={self._data_dir} is not a valid directory.")
    
    
        
        for i_Z, Z_float, Z_str in zip(range(no_Z), self._Z_arr, self._Z_str):
            # sanity check
            if f'{Z_float:.2f}' != Z_str:
                warn(
                    '_load_mesa_eos_table()', verbose,
                    f"{Z_float=} is not the same as {Z_str=}.",
                )
            for i_X, X_float, X_str in zip(range(no_X), self._X_arr, self._X_str):
                # sanity check
                if f'{X_float:.2f}' != X_str:
                    warn(
                        '_load_mesa_eos_table()', verbose,
                        f"{X_float} is not the same as {X_str=}.",
                    )
                with open(f"{self._data_dir}{os.path.sep}output_DE_z{Z_str}x{X_str}.bindata", 'rb') as f:
    
                    # load meta data
                    no_E, no_V, no_var2 = fortran_read_file_unformatted(f, 'i', 3)
                    
                    logV_float = np.array(fortran_read_file_unformatted(f, 'd', no_V))
                    if self._log10_V_arr is None:
                        self._log10_V_arr = logV_float
                    elif not np.allclose(self._log10_V_arr, logV_float):
                        warn(
                            '_load_mesa_eos_table()', verbose,
                            "Warning: logV array not the same across the data files.",
                            "This is not supposed to happen! Check the code and the data!",
                        )
    
                    logE_float = np.array(fortran_read_file_unformatted(f, 'd', no_E))
                    if self._log10_E_arr is None:
                        self._log10_E_arr = logE_float
                    elif not np.allclose(self._log10_E_arr, logE_float):
                        warn(
                            '_load_mesa_eos_table()', verbose,
                            "Warning: logE array not the same across the data files.",
                            "This is not supposed to happen! Check the code and the data!",
                        )
    
                    # init mesa_table as a structured array
                    if self._table_withZX is None:
                        self._table_withZX = np.full((no_Z, no_X, no_E, no_V), np.nan, dtype=self._table_dtype)
    
                    # fill mesa table
                    for i_V in range(no_V):
                        for i_E in range(no_E):
                            self._table_withZX[i_Z, i_X, i_E, i_V] = fortran_read_file_unformatted(f, 'd', no_var2)

        self._grid_withZX = (self._Z_arr, self._X_arr, self._log10_E_arr, self._log10_V_arr)
        self._grid_noZX   = (self._log10_E_arr, self._log10_V_arr)
        meshgrid_noZX = np.meshgrid(*self._grid_noZX, indexing='ij')
        meshgrid_noZX_coord = np.stack((
            np.full(meshgrid_noZX[0].shape, self._Z),
            np.full(meshgrid_noZX[0].shape, self._X),
            *meshgrid_noZX),
            axis=-1)
        self._table_noZX = np.full((no_E, no_V), np.nan, dtype=self._table_dtype)
        for val_name in self._table_withZX.dtype.names:
            self._table_noZX[val_name] = RegularGridInterpolator(
                self._grid_withZX, self._table_withZX[val_name], method='linear',
            )(meshgrid_noZX_coord)
        
        self._interp_dict = {
            val_name: RegularGridInterpolator(self._grid_noZX, self._table_noZX[val_name], method='linear', bounds_error=False)
            for val_name in self._table_noZX.dtype.names
        }

        del self._grid_withZX
        del self._table_withZX
        self._grid_withZX = ()
        self._table_withZX= np.array([])
            
        return self

    


    def get_val_cgs(
        self,
        val_name: str|list,
        rho: np.ndarray,
        u  : np.ndarray,
        *params_list,
        method  : str|None = None,
        verbose: int = 3,
        **params_dict,
    ) -> np.ndarray:
        """Interpolate value and return.
        
        Parameters
        ----------
        val_name: str | list
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, u: np.ndarray
            density and specific internal energy in cgs units.

        method: str | None
            Method to be used for interpolation.
            See scipy.interpolate.RegularGridInterpolator.__call__ docs.

        verbose: int
            How much errors, warnings, notes, and debug info to be print on screen.

            

        Note: As described by line 451 of the file phantom/src/main/eos_mesa_microphysics.f90,
            ! logRho = logV + 0.7*logE - 20
            Daniel's explanation of it is that
            "It’s some combination of pressure and density that makes sense only to people who compile equations of state"
            (2023-11-02 over Slack.)
            
        """

        log10_E = np.log10(u)
        log10_V = 20. + np.log10(rho) - 0.7 * log10_E
        _interp_coord = (log10_E, log10_V)

        # interpret val_name
        if val_name in ['rho', 'P', 'Pgas', 'T']:
            # data in table stored as log10
            val_type = 'log10_' + val_name
        else:
            val_type = val_name

        # get results
        ans = self._interp_dict[val_type](_interp_coord, method=method)

        # post-processing
        if val_name in ['rho', 'P', 'Pgas', 'T']:
            ans = 10**ans

        return ans

        








class EoS_MESA(EoS_Base):
    """Class for MESA Equation of State Objects."""
    
    def __init__(self, params: dict, settings: Settings=DEFAULT_SETTINGS, verbose: int=3):
        self.__mesa_table = _EoS_MESA_table(params, settings, verbose)

        return

    
    def get_val_cgs(
        self,
        val_name: str,
        rho: np.ndarray,
        u  : np.ndarray,
        *params_list,
        method  : str|None = None,
        verbose: int = 3,
        **params_dict,
    ) -> np.ndarray:
        """Getting specific values from EoS and rho and u in cgs units.

        Parameters
        ----------
        val_name: str
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, u: np.ndarray
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
        debug_info("EoS_MESA.get_val_cgs()", verbose, "Calling this.")
        return self.__mesa_table.get_val_cgs(
            val_name, rho, u, *params_list,
            method=method, verbose=verbose,
            **params_dict,
        )


