#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for SPH interpolation.

Assuming Phantom.
    (That is, the smoothing length h is dynamically scaled with density rho using
    rho = hfact**d * (m / h**d)
    for d-dimension and constant hfact.)

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info

#  import (general)
import numpy as np
import numba
from numba import jit
import sarracen





# Functions


@jit(nopython=False)
def get_h_from_rho(rho: np.ndarray|float, mpart: float, hfact: float, ndim:int = 3) -> np.ndarray|float:
    """Getting smoothing length from density.
    
    Assuming Phantom, where smoothing length h is dynamically scaled with density rho using
    rho = hfact**ndim * (m / h**ndim)
    for ndim-dimension and hfact the constant.
    So,
    h = hfact * (mpart / rho)**(1./ndim)
    """
    return hfact * (mpart / rho)**(1./ndim)






@jit(nopython=True)
def _get_sph_interp_phantom_np(
    locs : np.ndarray,
    vals : np.ndarray,
    xyzs : np.ndarray,
    hs   : np.ndarray,
    kernel_w  : numba.core.registry.CPUDispatcher,
    kernel_rad: float,
    ndim : int = 3,
) -> np.ndarray:
    """SPH interpolation subprocess.

    WARNING:
        * This func requires a very specific input array shape, and it does NOT do sanity check!
        * kernel_rad MUST be float instead of int!
        * All input numpy array must be in 'C' order (because stupid numba doesn't support 'F' order)
        * all inputs in locs, xyzs, hs must be finite (no np.inf nor np.nan)

    Using numpy array as input and numba for acceleration.
    
    *** THIS FUNC DOES NOT DO SAINTY CHECK *** 


    Parameters
    ----------
    locs : (nlocs, ndim )-shaped np.ndarray,
    vals : (npart, nvals)-shaped np.ndarray,
    xyzs : (npart, ndim )-shaped np.ndarray,
    hs   : (npart,      )-shaped np.ndarray,
    kernel_w  : sarracen.kernels.BaseKernel.w
    kernel_rad: float
        smoothing kernel radius in unit of h (outside this w goes to 0)
    ndim : int = 3
    
    Returns
    -------
    ans  : (nlocs, nvals)-shaped np.ndarray
    """
    nlocs = locs.shape[0]
    npart = vals.shape[0]
    nvals = vals.shape[1]
    ans_s = np.zeros((nlocs, nvals), dtype=vals.dtype)
    ans_w = np.zeros((nlocs, 1))

    # h * w_rad
    hw_rad = kernel_rad * hs

    for j in range(npart):
        # pre-select
        maybe_neigh = np.abs(locs - xyzs[j][np.newaxis, :]) < hw_rad[j]
        for i in range(nlocs):
            if np.all(maybe_neigh[i]):
                q_ij = np.sum((locs[i] - xyzs[j])**2)**0.5 / hs[j]
                if q_ij <= kernel_rad:
                    w_q = kernel_w(q_ij, ndim)
                    ans_s[i   ] += w_q * vals[j]
                    ans_w[i, 0] += w_q
    return ans_s / ans_w








def get_sph_interp_phantom(
    sdf      : sarracen.sarracen_dataframe.SarracenDataFrame,
    val_names: str|list,
    locs     : np.ndarray,
    kernel   : sarracen.kernels.BaseKernel = None,
    #hfact    : float = None,
    ndim     : int = 3,
    xyzs_names_list : list = ['x', 'y', 'z'],
    verbose : int = 3,
) -> np.ndarray:
    """SPH interpolation.

    Note: You should only interpolate conserved quantities! (i.e. density rho / specific energy u / momentum v)

    Make sure locs are in the same unit as sdf distance unit!


    Assuming Phantom.
        (That is, the smoothing length h is dynamically scaled with density rho using
        rho = hfact**d * (m / h**d)
        for d-dimension and constant hfact.)


    Theories:
    In SPH kernel interpolation theories, for arbitrary quantity A,
        \braket{A} (\mathbf{r}) 
        \equiv \sum_{j} \frac{m_j A_j}{\rho_j h_j^d} w(q_j(\mathbf{r}))
        = \frac{1}{h_\mathrm{fact}^d} \sum_{j} A_j w(q_j(\mathbf{r}))
    where rho = hfact**d * (m / h**d) is assumed.

    Taylor expansion of the above shows that
        \braket{A}
        = A \braket{1} + \nabla A \cdot (\braket{\mathbf{r}} - \mathbf{r}\braket{1}) + \mathcal{O}(h^2)

    So,
        A
        \approx \frac{\braket{A}}{\braket{1}}
        = \frac{ \sum_{j} A_j w(q_j(\mathbf{r})) }{\sum_{j} w(q_j(\mathbf{r}))}
    
        
    
    Parameters
    ----------
    sdf: sarracen.SarracenDataFrame
        Must contain columns: x, y, z, h.
        if hfact is None or kernel is None, will get from sdf.
        
    val_names: str
        Column label of the target smoothing data in sdf
        
    locs: np.ndarray
        (3) or (..., 3)-shaped array determining the location for interpolation.
        
    kernel: sarracen.kernels.base_kernel
        Smoothing kernel for SPH data interpolation.
        If None, will use the one in sdf.
        
    hfact: float
        constant factor for h.
        If None, will use the one in sdf.params['hfact'].

    ndim: int
        dimension of the space. Default is 3 (for 3D).
        DO NOT TOUCH THIS UNLESS YOU KNOW WHAT YOU ARE DOING.
        
    xyzs_names_list: list
        list of names of the columns that represents x, y, z axes (i.e. coord axes names)
        Make sure to change this if your ndim is something other than 3.

    verbose: int
        How much warnings, notes, and debug info to be print on screen. 
        
    Returns
    -------
    ans: float or np.ndarray
        Depending on the shape of locs, returns float or array of float.
    """

    
    # init
    npart = len(sdf)
    if kernel is None:
        kernel = sdf.kernel
    kernel_rad = float(kernel.get_radius())
    #if hfact is None:
    #    hfact = float(sdf.params['hfact'])
    locs = np.array(locs, copy=False, order='C')
    vals = np.array(sdf[val_names], copy=False, order='C')
    xyzs = np.array(sdf[xyzs_names_list], copy=False, order='C')    # (npart, ndim)-shaped array
    hs   = np.array(sdf['h'], copy=False, order='C')                # (npart,)-shaped array

    
    # fix input shapes
    if locs.ndim == 1:
        locs = locs[np.newaxis, :]
    do_squeeze = False
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]
        do_squeeze = True
    if xyzs.ndim == 1:
        xyzs = xyzs[:, np.newaxis]

    
    # sanity checks

    # warn if try to interp unexpected quantities
    if val_names not in ['rho', 'u', 'vx', 'vy', 'vz']:
        warn(
            'get_sph_interp()', verbose,
            "Kernel interpolation should be used with conserved quantities (density, energy, momentum),",
            f"but you are trying to do it with '{val_names}', which could lead to problematic results."
        )
    if ndim != 3:
        warn(
            'get_sph_interp()', verbose,
            f"You have set ndim={ndim}, which assumes a {ndim}D world instead of 3D.",
            "if the simulation is 3D, this means that the following calculations will be wrong.",
            "Are you sure you know what you are doing?",
        )
    if xyzs.shape != (npart, ndim):
        warn(
            'get_sph_interp()', verbose,
            f"xyzs.shape={xyzs.shape} is not (npart, ndim)={(npart, ndim)}!",
            "This is not supposed to happen and means that the following calculations will be wrong.",
            "Please check input to this function.",
        )


    # calc
    ans = _get_sph_interp_phantom_np(locs, vals, xyzs, hs, kernel.w, kernel_rad, ndim)
    
    if do_squeeze:
        ans = np.squeeze(ans)

    return ans


# set alias
get_sph_interp = get_sph_interp_phantom