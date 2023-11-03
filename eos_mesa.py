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
from .units import set_as_quantity, CGS_UNITS

#  import (general)
import os
import numpy as np
from astropy import units
from scipy.interpolate import RegularGridInterpolator



# Classes



class EoS_MESA_table:
    """A class to store and handle stored MESA tables."""
    def __init__(self, params: dict, settings: Settings, iverbose: int=3):
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
        
        self.load_mesa_eos_table(params, settings, iverbose=iverbose)
        return

    
    def __getitem__(self, i):
        return self._table_withZX[i]

    
    def __setitem__(self, i, val):
        raise NotImplementedError(
            "EoS_MESA_table: Please don't try to change things that are not supposed to be changed.")
        
    
    def load_mesa_eos_table(self, params: dict, settings: Settings, iverbose: int=3):
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
                    '_load_mesa_eos_table()', iverbose,
                    f"{Z_float=} is not the same as {Z_str=}.",
                )
            for i_X, X_float, X_str in zip(range(no_X), self._X_arr, self._X_str):
                # sanity check
                if f'{X_float:.2f}' != X_str:
                    warn(
                        '_load_mesa_eos_table()', iverbose,
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
                            '_load_mesa_eos_table()', iverbose,
                            "Warning: logV array not the same across the data files.",
                            "This is not supposed to happen! Check the code and the data!",
                        )
    
                    logE_float = np.array(fortran_read_file_unformatted(f, 'd', no_E))
                    if self._log10_E_arr is None:
                        self._log10_E_arr = logE_float
                    elif not np.allclose(self._log10_E_arr, logE_float):
                        warn(
                            '_load_mesa_eos_table()', iverbose,
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

        #raise NotImplementedError
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
                self._grid_withZX, self._table_withZX[val_name], method='linear')(
                meshgrid_noZX_coord
                )
        
        self._interp_dict = {
            val_name: RegularGridInterpolator(self._grid_noZX, self._table_noZX[val_name], method='linear')
            for val_name in self._table_noZX.dtype.names
        }
            
        return self


    def get_val(
        self,
        val_name: str|list,
        rho: np.ndarray|units.Quantity,
        u  : np.ndarray|units.Quantity,
        return_as_quantity: bool|None = None,
        method  : str|None = None,
    ) -> np.ndarray|units.Quantity:
        """Interpolate value and return.
        
        Parameters
        ----------
        val_name: str | list
            name of value to be interpolated.
            see the fields specified in self._table_dtype.
            e.g. 'rho'

        rho, u: np.ndarray | units.Quantity
            density and specific internal energy
            if numpy array, WILL ASSUME CGS UNITS.
            should have same units.

        X, Z : float
            hydrogen mass fraction, metallicity, respectively.

        return_as_quantity: bool | None
            if the results should be returned as a astropy.units.Quantity.
            If None, will only return as that if one of the input rho or u is that.

        method: str | None
            Method to be used for interpolation.
            See scipy.interpolate.RegularGridInterpolator.__call__ docs.


        Note: As described by line 451 of the file phantom/src/main/eos_mesa_microphysics.f90,
            ! logRho = logV + 0.7*logE - 20
            Daniel's explanation of it is that
            "It’s some combination of pressure and density that makes sense only to people who compile equations of state"
            (2023-11-02 over Slack.)
            
        """

        return_quantity = False
        if isinstance(rho, units.Quantity):
            rho = rho.cgs.value
            return_quantity = True
        if isinstance(u, units.Quantity):
            u = u.cgs.value
            return_quantity = True
        if return_as_quantity is not None:
            return_quantity = return_as_quantity

        log10_E = np.log10(u)
        log10_V = 20. + np.log10(rho) - 0.7 * log10_E
        _interp_coord = (log10_E, log10_V)

        # interpret val_name
        if val_name in ['rho', 'P', 'Pgas', 'T']:
            # data in table stored as log10
            val_type = 'log10_' + val_name
        else:
            val_type = val_name
            

        ans = self._interp_dict[val_type](_interp_coord, method=method)

        if val_name in ['rho', 'P', 'Pgas', 'T']:
            ans = 10**ans

        if return_quantity:
            if val_name == 'rho':
                val_units_text = 'density'
            elif val_name == 'T':
                val_units_text = 'temp'
            else:
                raise NotImplementedError
            ans = set_as_quantity(ans, CGS_UNITS['density'])
            

        return ans
    





class EoS_MESA(EoS_Base):
    """Class for MESA Equation of State Objects."""
    def __init__(self, params: dict, settings: Settings=DEFAULT_SETTINGS, iverbose: int=3):
        self.__mesa_table = EoS_MESA_table(params, settings, iverbose)

        return



