#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared helper functions
"""

# imports and internal settings
import numpy as np
from astropy import units
from clmuphantomlib.log import is_verbose, say
from clmuphantomlib import MyPhantomDataFrames, get_eos
from clmuphantomlib.eos import EoS_MESA_opacity
from clmuphantomlib.units_util import get_val_in_unit


def mpdf_read(
    job_name: str,
    file_index: int,
    eos_opacity: EoS_MESA_opacity,
    mpdf: MyPhantomDataFrames|None = None,
    reset_xyz_by: str='CoM',
    kappa_gas : units.Quantity = 2e-4*(units.cm**2/units.g),
    kappa_tol : units.Quantity = 1e-7*(units.cm**2/units.g),
    T_cond_oxy: units.Quantity = 1450 * units.K,
    do_extrap: bool = False,
    verbose: int = 3,
) -> MyPhantomDataFrames:
    """Read the dump files and get T and kappa,
    
    using a blend of mesa opacity (for high T) and stored kappa value in the dump (for low T; in my project, this col is nucleation dust opacity).
    If no kappa is found, will use default 2e-4 cm2/g for the low T part instead. You can disable this by setting T_cond_oxy to 0K.
    """
    if mpdf is None:
        mpdf = MyPhantomDataFrames()

    mpdf.read(job_name, file_index, reset_xyz_by=reset_xyz_by, calc_params=['vr'], verbose=verbose)
    temp_key = {'T', 'temperature', 'Tdust'}.intersection(mpdf.data['gas'].keys()).pop()
    mpdf.data['gas']['T'    ] = mpdf.data['gas'][temp_key]
    if 'kappa' in mpdf.data['gas'].keys():
        unit_opacity = units.cm**2/units.g    # opacity units in original phantom dumpfiles
        kappa_mesa = eos_opacity.get_kappa(mpdf.get_val('rho'), mpdf.get_val('T'), do_extrap=do_extrap)
        mpdf.data['gas']['kappa_dust'] = mpdf.data['gas']['kappa'] - kappa_gas.to_value(unit_opacity)
        # fix negative opacities
        mpdf.data['gas']['kappa_dust'] = np.where(
            mpdf.data['gas']['kappa_dust'] < kappa_tol.to_value(unit_opacity),
            0.,
            mpdf.data['gas']['kappa_dust'],
        )
        mpdf.data['gas']['kappa'] = np.where(
            mpdf.data['gas']['T'] < T_cond_oxy,
            mpdf.data['gas']['kappa'],
            mpdf.data['gas']['kappa_dust'] + kappa_mesa.to_value(unit_opacity),
        )
        mpdf.data['gas']['kappa'] = get_val_in_unit(mpdf.data['gas']['kappa'], unit_opacity, mpdf.units['opacity'])
        mpdf.data['gas']['kappa_dust'] = get_val_in_unit(mpdf.data['gas']['kappa_dust'], unit_opacity, mpdf.units['opacity'])
        
        if is_verbose(verbose, 'debug'):
            say('debug', 'mpdf_read()', verbose,
                f"{np.count_nonzero(mpdf.data['gas']['kappa_dust']) = }",
                f"{np.count_nonzero(mpdf.data['gas']['kappa'])      = }")
    else:
        #raise NotImplementedError("non-dusty sims (no kappa column in dump files) not yet implemented")

        # just use mesa opacity
        
        kappa_mesa = eos_opacity.get_kappa(mpdf.get_val('rho'), mpdf.get_val('T'), do_extrap=do_extrap)
        mpdf.data['gas']['kappa'] = np.where(
            mpdf.data['gas']['T'] < T_cond_oxy,
            kappa_gas.to_value( mpdf.units['opacity']),
            kappa_mesa.to_value(mpdf.units['opacity']),
        )

    return mpdf