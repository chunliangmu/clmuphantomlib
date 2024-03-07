#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for light-related funcs.

E.g. ray-tracing / radiative transfer / finding photosphere / lightcurves

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import say, is_verbose
from .geometry import get_dist2_between_2pt, get_closest_pt_on_line
from .sph_interp import get_sph_interp, get_h_from_rho, get_no_neigh
from .units_util import set_as_quantity, set_as_quantity_temperature, get_units_field_name
from .eos_base import EoS_Base

#  import (general)
import numpy as np
from numba import jit
import sarracen



# Functions


@jit(nopython=False)
def get_ray_unit_vec(ray: np.ndarray) -> np.ndarray:
    """Get unit vector of a ray (which is a line).
    
    Parameters
    ----------
    ray: (2, N)-dimensional array_like, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        
    Returns
    -------
    ray_unit_vec: (N,)-dimensional np.ndarray
        unit vector of ray
    """
    ray = np.array(ray, copy=False)
    ray_unit_vec = ray[1, :] - ray[0, :]
    ray_unit_vec = ray_unit_vec / np.sum(ray_unit_vec**2)**0.5
    return ray_unit_vec





# get optical depth by ray-tracing

@jit(nopython=False)
def get_optical_depth_by_ray_tracing_3D(
    sdf    : sarracen.SarracenDataFrame,
    ray    : np.ndarray,
    kernel : sarracen.kernels.BaseKernel = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get an array of optical depth for a ray traced backwards.
    
    Assuming 3D.
    
    
    Parameters
    ----------
    sdf: sarracen.SarracenDataFrame
        Must contain columns: x, y, z, m, h, kappa
        
    ray: 2D list/array with shape of (2, 3), i.e. [[float, float, float], [float, float, float]]
        Representing the ray trajectory. Currently only straight infinite lines are supported.
        [[begin point], [end point]]
        where the end point is closer to the observer.
        
    kernel: sarracen.kernels.base_kernel
        Smoothing kernel for SPH data interpolation.
        If None, will use the one in sdf.
    
    
    Returns
    -------
    pts_on_ray, dtaus, pts_order
    
    pts_on_ray: np.ndarray
        Orthogonal projections of the particles' locations onto the ray.
    
    dtaus: np.ndarray
        Optical depth tau contributed by each particles. In order of the original particles order in the dump file.
        Remember tau is a dimensionless quantity.
    
    pts_order: np.ndarray
        indices of the particles where dtaus are non-zero.
        The indices are arranged by distances to the observer, i.e. the particles closest to the observer comes first, 
        and the furtherest comes last.
    
    """
    # init
    ndim = 3
    npart = len(sdf)
    if kernel is None: kernel = sdf.kernel
    col_kernel = kernel.get_column_kernel_func(samples=1000) # w integrated from z
    pts = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array (must be this shape for pts_order sorting below)
    hs  = np.array(sdf['h'])    # npart-shaped array
    masses = np.array(sdf['m'])
    kappas = np.array(sdf['kappa'])
    ray = np.array(ray)
    if ray.shape != (2, ndim):
        raise ValueError(f"var 'ray' should have 2 points (i.e. with shape (2, ndim={ndim})), but ray.shape={ray.shape} is not.")
    ray_unit_vec = get_ray_unit_vec(ray)
    

    # dtaus = kappa * m / h**2 * col_kernel(q, ndim-1)
    # q for SPH weight input
    pts_on_ray = get_closest_pt_on_line(pts, ray)
    qs = np.sum((pts - pts_on_ray)**2, axis=-1)**0.5 / hs
    terms = kappas * masses / hs**2
    dtaus = np.array([term * col_kernel(q, ndim-1) for term, q in zip(terms, qs)])
    
    
    # calculate the order in which particles lie along the ray from observer to the opposite side
    pts_on_ray_factor = -np.sum(pts_on_ray * ray_unit_vec, axis=-1) # the lower, the more on the pt2 side (observer)
    ## simple ver.
    #pts_order = np.argsort(pts_on_ray_factor)
    # longer ver.
    # remove zero dtaus terms in pts_order to optimize
    pts_indices = np.arange(len(pts_on_ray_factor))
    pts_nonzero_indices = dtaus.nonzero()[0]
    pts_order = pts_indices[pts_nonzero_indices][np.argsort(pts_on_ray_factor[pts_nonzero_indices])]
    
    return pts_on_ray, dtaus, pts_order







#@jit(nopython=False)
def get_photosphere_on_ray(
    pts_on_ray            : np.ndarray,
    dtaus                 : np.ndarray,
    pts_order             : np.ndarray,
    sdf                   : sarracen.SarracenDataFrame,
    ray                   : np.ndarray,
    calc_params           : list       = ['loc', 'R1'],
    hfact                 : float      = None,
    mpart                 : float      = None,
    eos                   : EoS_Base   = None,
    sdf_units             : dict       = None,
    ray_unit_vec          : np.ndarray = None,
    kernel                : sarracen.kernels.base_kernel = None,
    do_skip_zero_dtau_pts : bool       = True,
    photosphere_tau       : float      = 1.,
    return_as_quantity    : bool|None  = True,
    verbose : int = 3,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calc the location where the photosphere intersect with the ray.

    Assuming 3D.

    
    Parameters
    ----------
    pts_on_ray, dtaus, pts_order
        output from get_optical_depth_by_ray_tracing_3D().

        pts_on_ray: np.ndarray
            Orthogonal projections of the particles' locations onto the ray.
        
        dtaus: np.ndarray
            Optical depth tau contributed by each particles. In order of the original particles order in the dump file.
            Remember tau is a dimensionless quantity.
        
        pts_order: np.ndarray
            indices of the particles where dtaus are non-zero.
            The indices are arranged by distances to the observer, i.e. the particles closest to the observer comes first, 
            and the furtherest comes last.

    sdf: sarracen.SarracenDataFrame
        Must contain columns: x, y, z, h    # kappa, rho,
        
    ray: (2, 3)-shaped numpy array, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        First  point pt1 is the reference of the distance if R1 is calc-ed.
        Second point pt2 points to the observer, and is closer to the observer.

    calc_params: list or tuple of str
        parameters to be calculated / interpolated at the photosphere location.
        Results will be put into the photosphere dict in the output.
        Acceptable input: (Note: will always calc 'loc' if calc_params is not empty)
            'is_found': will return bool.
                Will always be outputted regardless of in calc_params or not.
            'loc': will return (3,)-shaped numpy array.
                photophere location.
            'R1' : will return float.
                distance between photosphere location and the ray[0].
                Could be negative if loc is on the other side of the ray.
            'nneigh': will return int.
                Number of neighbour particles of the photosphere loc.
            'rho': will return float.
                density at the photosphere.
            'u': will return float.
                specific internel energy at the photosphere.
            'h'  : will return float.
                smoothing length at the photosphere.
                Will always calc 'rho' if to calc 'h'.
            'T'  : will return float.
                Temperature at the photosphere.
                Warning: if not supplied 'temp' keyword in sdf_units, will return in cgs units.
    
    hfact, mpart: float
        Only useful if you are calc-ing 'h'
        $h_\mathrm{fact}$ and particle mass used in the phantom sim.
        If None, will get from sdf.params['hfact'] and sdf.params['mass']

    eos: .eos_base.EoS_BASE
        Only useful if you are calc-ing 'T'
        Equation of state object defined in eos_base.py

    sdf_units: dict
        Only useful if you are calc-ing 'T'
        in which case, supply rho, u, and T units in this dict
        e.g.
        sdf_units = {
            'density': units.Msun / units.Rsun**3,
            'specificEnergy': units.Rsun**2 / units.s**2,
            'temp': units.K,
        }
    
    ray_unit_vec: (3,)-shaped np.ndarray
        unit vector of ray. will calc this if not supplied.
        
    kernel: sarracen.kernels.base_kernel
        Smoothing kernel for SPH data interpolation.
        If None, will use the one in sdf.
        
    do_skip_zero_dtau_pts: bool
        Whether or not to skip particles with zero dtaus (i.e. no contribution to opacity) to save computational time.
        If skiped, these particles' locs will be excluded from results as well
        
    photosphere_tau: float
        At what optical depth (tau) is the photosphere defined.

    return_as_quantity: bool | None
        If True or None, the results in photosphere will be returned as a astropy.units.Quantity according to sdf_units.
        (pts_waypts, pts_waypts_t, taus_waypts) will also be returned as numpy array and NOT as Quantity.
        The diff between True and None is that True will raise an error if units not supplied in sdf_units,
        while None will just return as numpy array in such case.
        
    
    verbose: int
        How much warnings, notes, and debug info to be print on screen. 


    Returns
    -------
    photosphere, (pts_waypts, pts_waypts_t, taus_waypts)

    photosphere: dict
        dict of values found at the photosphere intersection point with the ray.
        will always have 

    pts_waypts: (npart, 3)-shaped numpy array
        location of the waypoints on ray

    pts_waypts_t: (npart)-shaped numpy array
        distance of the waypoints from ray[0]

    taus_waypts: (npart)-shaped numpy array
        optical depth at the waypoints.
        
    """

    # init
    if ray_unit_vec is None:
        ray_unit_vec = get_ray_unit_vec(ray)
    if kernel is None:
        kernel = sdf.kernel
    if do_skip_zero_dtau_pts:
        pts_order = pts_order[np.where(dtaus[pts_order])[0]]
    ray_0 = np.array(ray[0], copy=False)
    pts_ordered    = np.array(sdf[['x', 'y', 'z']].iloc[pts_order])
    hs_ordered     = np.array(sdf[ 'h'           ].iloc[pts_order])
    #kappas_ordered = np.array(sdf[ 'kappa'       ].iloc[pts_order])
    #rhos_ordered   = np.array(sdf[ 'rho'         ].iloc[pts_order])
    pts_on_ray_ordered = pts_on_ray[pts_order]
    dtaus_ordered = dtaus[pts_order]


    
    # get waypts (way points) for pts (point locations) and taus (optical depths)
    #  waypts are suitable for linear interpolation
    #  taus_waypts[0] is 0; taus_waypts[-1] is total optical depth from the object

    
    #  step 1: determine the waypts location by assuming pts as balls with constant kappa and density
    
    #   step 1a: getting the size of pts balls on the ray
    pts_dist2_to_ray = get_dist2_between_2pt(pts_ordered, pts_on_ray_ordered)
    #    Assuming a h radius ball
    pts_radius = kernel.get_radius() * hs_ordered
    pts_size_on_ray = pts_radius**2 - pts_dist2_to_ray
    # put a small number (1e-8*h) in case of negative pts_size_on_ray, so that the code does not freak out
    pts_size_on_ray_min = 1e-8*hs_ordered
    pts_size_on_ray = np.where(pts_size_on_ray < pts_size_on_ray_min**2, pts_size_on_ray_min, pts_size_on_ray**0.5)
    #pts_size_on_ray = dtaus_ordered / (kappas_ordered * rhos_ordered)    # does not work because rho is not a constant within the particle

    #   step 1b: getting the waypoint locs
    #    pts_waypts will be sorted next
    pts_waypts = np.full((pts_ordered.shape[0]*2, pts_ordered.shape[1]), np.nan)
    pts_waypts[0::2] = pts_on_ray_ordered + ray_unit_vec[np.newaxis, :] * pts_size_on_ray[:, np.newaxis]
    pts_waypts[1::2] = pts_on_ray_ordered - ray_unit_vec[np.newaxis, :] * pts_size_on_ray[:, np.newaxis]

    #   step 1c: sort waypoint locs
    #    pts_waypts_t: the distance from waypts to ray_0 (negative if in the opposite direction)
    pts_waypts_t = np.sum((pts_waypts - ray_0) * ray_unit_vec, axis=-1)
    pts_waypts_t_left  = pts_waypts_t[0::2]
    pts_waypts_t_right = pts_waypts_t[1::2]
    pts_waypts_inds = np.argsort(pts_waypts_t)[::-1]
    pts_waypts   = pts_waypts[  pts_waypts_inds]
    pts_waypts_t = pts_waypts_t[pts_waypts_inds]
    
    #  step 2: determine the waypts optical depth
    taus_waypts = np.zeros(len(dtaus_ordered)*2)
    for waypt_t_left, waypt_t_right, dtau in zip(pts_waypts_t_left, pts_waypts_t_right, dtaus_ordered):
        # Note: np.interp assumes xp increasing, so we need to reverse this
        taus_waypts += np.interp(pts_waypts_t[::-1], [waypt_t_right, waypt_t_left], [dtau, 0.], left=dtau, right=0.)[::-1]
        

    # prepare answers
    # is found?
    if np.isfinite(taus_waypts[-1]):
        # in case there is nan in the later part of the array
        taus_max = taus_waypts[-1]
    else:
        taus_max = np.nanmax(taus_waypts)
    photosphere = {
        'is_found': (taus_max > photosphere_tau)
    }
    
    # get photosphere parameters
    if calc_params:
        # always calc location if anything needs to be calc-ed
        photosphere['loc'] = np.array([
            np.interp(photosphere_tau, taus_waypts, pts_waypts[:, ax], right=np.nan)
            for ax in range(pts_waypts.shape[1])
        ])

        # do prerequisite check
        calc_params = list(calc_params)
        if 'h' in calc_params:
            if 'rho' not in calc_params: calc_params.append('rho')
        if 'T'   in calc_params:
            if 'rho' not in calc_params: calc_params.append('rho')
            if 'u'   not in calc_params: calc_params.append('u')

        # first calc prerequisites
        calc_these = []
        for calc_name in calc_params:
            if   calc_name == 'loc':
                # already calc-ed
                pass
            elif calc_name == 'R1':
                photosphere['R1']  = np.interp(photosphere_tau, taus_waypts, pts_waypts_t, right=np.nan)
            elif calc_name in {'rho', 'u'}:
                photosphere[calc_name]  = get_sph_interp(sdf, calc_name, photosphere['loc'], kernel=kernel, verbose=verbose)
            elif calc_name in {'nneigh'}:
                photosphere[calc_name]  = get_no_neigh(sdf, photosphere['loc'], kernel=kernel, verbose=verbose)
            else:
                calc_these.append(calc_name)
    
        # now the rest
        for calc_name in calc_these:
            if calc_name == 'h':
                if hfact is None: hfact = sdf.params['hfact']
                if mpart is None: mpart = sdf.params['mass']
                photosphere['h']  = get_h_from_rho(photosphere['rho'], mpart, hfact)
            elif calc_name == 'T':
                if eos   is None: raise ValueError("get_photosphere_on_ray(): Please supply equation of state to calculate temperature.")
                try:
                    photosphere['T']  = eos.get_temp(
                        set_as_quantity(photosphere['rho'], sdf_units['density']),
                        set_as_quantity(photosphere['u'  ], sdf_units['specificEnergy']))
                    if 'temp' in sdf_units:
                        photosphere['T'] = set_as_quantity_temperature(photosphere['T'], sdf_units['temp']).value
                    else:
                        photosphere['T'] = photosphere['T'].value
                except ValueError:
                    # eos interp could go out of bounds if it's a tabulated EoS
                    # which will raise a Value Error
                    photosphere['T'] = np.nan
            else:
                # just interpolate it (#IT JUST WORKS)
                photosphere[calc_name]  = get_sph_interp(sdf, calc_name, photosphere['loc'], kernel=kernel, verbose=verbose)

        # add units
        if return_as_quantity or return_as_quantity is None:
            for calc_name in photosphere.keys():
                if calc_name not in {'is_found', 'nneigh'}:
                    # find appropriate unit
                    try:
                        unit_field_name = get_units_field_name(calc_name)
                    except NotImplementedError:
                        # failed to find unit type
                        unit_field_name = None
                    if unit_field_name in sdf_units.keys():
                        # add units
                        photosphere[calc_name] = set_as_quantity(photosphere[calc_name], sdf_units[unit_field_name])
                    # errors
                    elif unit_field_name is None:
                        if is_verbose(verbose, 'warn'):
                            say('warn', 'get_photosphere_on_ray()', verbose,
                                f"Cannot find the corresponding unit for {calc_name}. Will return as numpy array instead.")
                    elif return_as_quantity is not None:
                        raise ValueError(f"Please supply {unit_field_name} in sdf_units.")
        
        
    return photosphere, (pts_waypts, pts_waypts_t, taus_waypts)
