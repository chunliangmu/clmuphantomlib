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



@jit(nopython=True)
def _get_sph_interp_phantom_np(
    locs : np.ndarray,
    vals : np.ndarray,
    xyzs : np.ndarray,
    hs   : np.ndarray,
    kernel_w: numba.core.registry.CPUDispatcher,
    ndim : int = 3,
    #iverbose : int = 3,
) -> np.ndarray:
    """SPH interpolation subprocess.

    WARNING: This func requires a very specific input array shape, and it does NOT do sanity check!

    Using numpy array as input and numba for acceleration.
    
    *** THIS FUNC DOES NOT DO SAINTY CHECK ***

    Note: All input numpy array must be in 'C' order (because stupid numba doesn't support 'F' order)


    Parameters
    ----------
    locs : (nlocs, 1,  ndim)-shaped np.ndarray,
    vals : (1, npart, nvals)-shaped np.ndarray,
    xyzs : (1, npart,  ndim)-shaped np.ndarray,
    hs   : (1, npart,      )-shaped np.ndarray,
    kernel_w: sarracen.kernels.BaseKernel.w
    ndim : int = 3
    
    Returns
    -------
    ans: (nlocs,) or (nlocs, nvals)-shaped np.ndarray
        The dimension of the array should match vals.
        i.e. if vals is (npart, )-shaped, then (,) or (nlocs,)-shaped array will be returned;
        if vals is (npart, nvals)-shaped, then (nlocs,  nvals)-shaped array will be returned.
    """

    # dist2: (nlocs, npart)-shaped np.ndarray
    dist2 = np.sum((xyzs - locs)**2, axis=-1)
    # qs : (nlocs, npart)-shaped array
    qs = dist2**0.5 / hs
    # w_q: (nlocs, npart, 1)-shaped array
    w_q = np.expand_dims(kernel_w(qs, ndim), 2) # [:, :, np.newaxis]
    print(w_q.shape)
    # ans: (nlocs, nvals)-shaped array
    ans = np.sum(vals * w_q, axis=1) / np.sum(w_q, axis=1)
    return ans





"""
    # calc
    if locs.ndim == 1:
        loc = locs
        if len(loc) != ndim:
            raise ValueError(f"{loc.shape=} is not (3)")
            
        qs = np.sum((xyzs - loc)**2, axis=-1)**0.5 / hs
        ans = np.sum(mA_div_rhoh3 * kernel.w(qs, ndim))
        return ans
    elif locs.ndim == 2:
        if locs.shape[-1] != ndim:
            raise ValueError(f"{loc.shape=} is not (..., {ndim})")
            
        ans_shape = (*locs.shape[:-1], *vals.shape[1:])
        ans = np.full(ans_shape, np.nan, dtype=vals.dtype)
        # non-zero range of the kernel
        r2_range = (kernel_radius * hs)**2
        for i, loc in enumerate(locs):
            r2s = np.sum((xyzs - loc)**2, axis=-1)
            indexs = r2s < r2_range
            qs_sliced = r2s[indexs]**0.5 / hs[indexs]
            ans[i] = np.sum(mA_div_rhoh3[indexs] * kernel.w(qs_sliced, ndim))
        return ans
    else:
        raise NotImplementedError(f"locs.ndim={locs.ndim} higher than 2 is not implemented")
"""





def get_sph_interp_phantom(
    sdf      : sarracen.sarracen_dataframe.SarracenDataFrame,
    val_names: str|list,
    locs     : np.ndarray,
    kernel   : sarracen.kernels.BaseKernel = None,
    #hfact    : float = None,
    ndim     : int = 3,
    xyzs_names_list : list = ['x', 'y', 'z'],
    iverbose : int = 3,
) -> np.ndarray:
    """SPH interpolation.

    Note: You should only interpolate conserved quantities! (i.e. density rho / specific energy u / momentum v)

    make sure locs are in the same unit as sdf distance unit!


    valssuming Phantom.
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

    iverbose: int
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
    kernel_radius = kernel.get_radius()
    #if hfact is None:
    #    hfact = float(sdf.params['hfact'])
    locs = np.array(locs, copy=False, order='C')
    vals = np.array(sdf[val_names], copy=False, order='C')
    xyzs = np.array(sdf[xyzs_names_list], copy=False, order='C')    # (npart, ndim)-shaped array
    hs   = np.array(sdf['h'], copy=False, order='C')                # npart-shaped array

    
    # fix input shapes
    if locs.ndim == 1:
        locs = locs[np.newaxis, :]
    do_squeeze = False
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]
        do_squeeze = True
    if xyzs.ndim == 1:
        xyzs = xyzs[:, np.newaxis]

    vals = vals[np.newaxis, :, :] # (1, npart, nvals)-shaped
    locs = locs[:, np.newaxis, :] # (nlocs, 1, ndim)-shaped
    xyzs = xyzs[np.newaxis, :, :] # (1, npart, ndim)-shaped
    hs   = hs[  np.newaxis, :]    # (1, npart,)-shaped

    
    # sanity checks

    # warn if try to interp unexpected quantities
    if val_names not in ['rho', 'u', 'vx', 'vy', 'vz']:
        warn(
            'get_sph_interp()', iverbose,
            "Kernel interpolation should be used with conserved quantities (density, energy, momentum),",
            f"but you are trying to do it with '{val_names}', which could lead to problematic results."
        )
    if ndim != 3:
        warn(
            'get_sph_interp()', iverbose,
            f"You have set ndim={ndim}, which assumes a {ndim}D world instead of 3D.",
            "if the simulation is 3D, this means that the following calculations will be wrong.",
            "Are you sure you know what you are doing?",
        )
    if xyzs.shape != (npart, ndim):
        warn(
            'get_sph_interp()', iverbose,
            f"xyzs.shape={xyzs.shape} is not (npart, ndim)={(npart, ndim)}!",
            "This is not supposed to happen and means that the following calculations will be wrong.",
            "Please check input to this function.",
        )


    # calc
    ans = _get_sph_interp_phantom_np(locs, vals, xyzs, hs, kernel.w, ndim)
    
    if do_squeeze:
        ans = np.squeeze(ans)

    return ans
