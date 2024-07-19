#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for SPH interpolation.

Assuming Phantom.
    (i.e., the smoothing length h is dynamically scaled with density rho using
    rho = hfact**d * (m / h**d)
    for d-dimension and constant hfact.)

Owner: Chunliang Mu


-------------------------------------------------------------------------------

Side note: Remember to limit line length to 79 characters according to PEP-8
    https://peps.python.org/pep-0008/#maximum-line-length    
which is the length of below line of '-' characters.

-------------------------------------------------------------------------------

"""



# Init


#  import (my libs)
from .log import is_verbose, say

#  import (general)
import math
from typing import Callable
import numpy as np
from numpy import typing as npt
from scipy.spatial import kdtree
import numba
from numba import jit
from astropy import units
import sarracen





# Functions




def get_col_kernel_funcs(
    kernel : sarracen.kernels.BaseKernel,
    kernel_nsamples : int = 1000,
    ndim   : int = 3,
):
    """Get numba-accelerated cum-sum-along-z kernel & column kernel functions.
    ---------------------------------------------------------------------------

    Parameters
    ----------
    kernel : sarracen.kernels.BaseKernel
    
    kernel_nsamples: int
       number of sample points. Determines resolution. 
       Total data points used are (2*kernel_nsamples+1) * (kernel_nsamples+1)

    ndim : int
        Dimensions. should be 3 for 3D.


    Returns: w_col(), w_csz()
    -------
    w_col(q_xy: np.float64, ndim: np.int64) -> np.float64
        Column kernel.
        Returns \int_{-w_\mathrm{rad}}^{+w_\mathrm{rad}} w(\sqrt{q_{xy}^2 + q_z^2}) dq_z
        
    w_csz(q_xy: np.float64, q_z : np.float64, ndim: np.int64) -> np.float64
        Cumulative summed kernel along z axis.
        Returns \int_{-w_\mathrm{rad}}^{q_z} w(\sqrt{q_{xy}^2 + q_z^2}) dq_z

    """

    kernel_rad = kernel.get_radius()
    
    qs_z  = np.linspace(-kernel_rad, kernel_rad, 2*kernel_nsamples+1)
    qs_xy = np.linspace(0., kernel_rad, kernel_nsamples+1)
    dq_z  = qs_z[1] - qs_z[0]
    dq_xy = qs_xy[1] - qs_xy[0]
    qs_xy_z = np.sqrt((qs_xy**2)[:, np.newaxis] + (qs_z**2)[np.newaxis, :])
    wq_xy_z = kernel.w(qs_xy_z, ndim)
    # cumsum of the kernel along z; (q_xy.size, q_z.size)-shaped
    w_csz_arr = np.cumsum(wq_xy_z, axis=1) * dq_z
    
    @jit(nopython=True, fastmath=True)
    def w_csz(
        q_xy: np.float64,
        q_z : np.float64,
        ndim: np.int64,
    ) -> np.float64:
        """Cumulative summed kernel along z axis.
        ---------------------------------------------------------------------------
    
        Does 2D interpolation on a pre-calc-ed table to calc this.
    
        
        Parameters
        ----------
    
        q_xy: float
            np.sqrt(x**2 + y**2) / h
        q_z : float
            z / h
        ndim: int
            Dimensions. should be 3 for 3D.
    
        Returns
        -------
        ans
            \int_{-w_\mathrm{rad}}^{q_z} w(\sqrt{q_{xy}^2 + q_z^2}) dq_z 
        """
    
        ind_xy = q_xy / dq_xy
        ind_z  = (kernel_rad + q_z) / dq_z
    
        
        ind_xy_m = math.floor(ind_xy)
        ind_xy_p = ind_xy_m + 1
        ind_z_m  = math.floor(ind_z)
        ind_z_p  = ind_z_m  + 1
        
        if ind_xy_m < 0 or ind_xy_p > kernel_nsamples or ind_z_m < 0:
            ans = 0
        else:
            tx = ind_xy - ind_xy_m
            tz = ind_z  - ind_z_m
            
            if  ind_z_p > 2*kernel_nsamples:
                ans = (
                    (1.-tx) * w_csz_arr[ind_xy_m, -1] +
                        tx  * w_csz_arr[ind_xy_p, -1]
                )
            else:
                ans = (
                    (1.-tx) * (1.-tz) * w_csz_arr[ind_xy_m  , ind_z_m  ] +
                    (1.-tx) *     tz  * w_csz_arr[ind_xy_m  , ind_z_m+1] +
                        tx  * (1.-tz) * w_csz_arr[ind_xy_m+1, ind_z_m  ] +
                        tx  *     tz  * w_csz_arr[ind_xy_m+1, ind_z_m+1]
                )
        return ans
    
    
    @jit(nopython=True, fastmath=True)
    def w_col(
        q_xy: np.float64,
        ndim: np.int64,
    ) -> np.float64:
        """Column kernel.
        ---------------------------------------------------------------------------
    
        Does interpolation on a pre-calc-ed table to calc this.
    
        
        Parameters
        ----------
    
        q_xy: float
            np.sqrt(x**2 + y**2) / h
        ndim: int
            Dimensions. should be 3 for 3D.
    
        Returns
        -------
        ans
            \int_{-w_\mathrm{rad}}^{+w_\mathrm{rad}} w(\sqrt{q_{xy}^2 + q_z^2}) dq_z 
        """
    
        ind_xy = q_xy / dq_xy
        ind_xy_m = math.floor(ind_xy)
        ind_xy_p = ind_xy_m + 1
        
        if ind_xy_m < 0 or ind_xy_p > kernel_nsamples:
            ans = 0
        else:
            tx = ind_xy - ind_xy_m
            ans = (
                (1.-tx) * w_csz_arr[ind_xy_m, -1] +
                    tx  * w_csz_arr[ind_xy_p, -1]
            )
        return ans

    return w_col, w_csz




def get_h_from_rho(
    rho: float | npt.NDArray[np.float64] | units.Quantity,
    mpart: float,
    hfact: float,
    ndim :int = 3,
) -> float | npt.NDArray[np.float64] | units.Quantity:
    """Getting smoothing length from density.
    
    Assuming Phantom,
    where smoothing length h is dynamically scaled with density rho using
    rho = hfact**ndim * (m / h**ndim)
    for ndim-dimension and hfact the constant.
    So,
    h = hfact * (mpart / rho)**(1./ndim)
    """
    return hfact * (mpart / rho)**(1./ndim)


@jit(nopython=True)
def get_h_from_rho_nb(
    rho: float | npt.NDArray[np.float64],
    mpart: float,
    hfact: float,
    ndim :int = 3,
) -> float | npt.NDArray[np.float64]:
    """Getting smoothing length from density. Numba version.
    
    Assuming Phantom,
    where smoothing length h is dynamically scaled with density rho using
    rho = hfact**ndim * (m / h**ndim)
    for ndim-dimension and hfact the constant.
    So,
    h = hfact * (mpart / rho)**(1./ndim)
    """
    return hfact * (mpart / rho)**(1./ndim)



def get_rho_from_h(
    h: float | npt.NDArray[np.float64] | units.Quantity,
    mpart: float,
    hfact: float,
    ndim :int = 3,
) -> float | npt.NDArray[np.float64] | units.Quantity:
    """Getting density from smoothing length.
    
    Assuming Phantom,
    where smoothing length h is dynamically scaled with density rho using
    rho = hfact**ndim * (m / h**ndim)
    for ndim-dimension and hfact the constant.

    No safety check is performed-
        make sure neither hfact and h are not in int datatype! 
    """
    return mpart * (hfact / h)**ndim


@jit(nopython=True)
def get_rho_from_h_nb(
    h: float | npt.NDArray[np.float64],
    mpart: float,
    hfact: float,
    ndim: int = 3,
) -> float | npt.NDArray[np.float64]:
    """Getting density from smoothing length. Numba version.
    
    Assuming Phantom,
    where smoothing length h is dynamically scaled with density rho using
    rho = hfact**ndim * (m / h**ndim)
    for ndim-dimension and hfact the constant.

    No safety check is performed-
        make sure neither hfact and h are not in int datatype! 
    """
    return mpart * (hfact / h)**ndim





def get_dw_dq(
    kernel: sarracen.kernels.BaseKernel,
    ndim: int = 3,
    nsample_per_h: int = 1000,
) -> Callable[[float, int], float]:
    """Return a function of the derivative of the kernel w."""
    w = kernel.w
    w_rad = kernel.get_radius()
    nsample = nsample_per_h*w_rad
    qs = np.linspace(0., w_rad, nsample)
    w_qs = w(qs, ndim)
    dw_dqs = np.gradient(w_qs, np.diff(qs)[0])
    # fix endpoints
    dw_dqs[0] = 0
    dw_dqs[-1]= 0 
    ndim_used = ndim
    
    @jit(nopython=True)
    def dw_dq(q : float, ndim: int) -> float:
        if ndim != ndim_used: raise NotImplementedError
        return np.interp(q, qs, dw_dqs)

    return dw_dq





def get_no_neigh(
    sdf       : None|sarracen.SarracenDataFrame,
    locs      : npt.ArrayLike,
    kernel    : None|sarracen.kernels.BaseKernel = None,
    kernel_rad: None|float = None,
    hs_at_locs: None|float | npt.NDArray[np.float64] = None,
    sdf_kdtree: None|kdtree.KDTree = None,
    ndim      : int = 3,
    xyzs_names_list : list = ['x', 'y', 'z'],
    verbose   : int = 3,
) -> np.ndarray|int:
    """Get the number of neighbour particles.


    Provide either:
    1. (Searching by hs_at_loc)
        locs, hs_at_locs, (kernel_rad or kernel or sdf), (sdf_kdtree or (sdf and xyzs_names_list))
        * You can set sdf to None if you provide both kernel_rad and sdf_kdtree.
    2. (Searching by hs of the particles in sdf)
        sdf, locs, xyzs_names_list.
        * Remember to set hs_at_loc to None! (it is None by default)

    
    Parameters
    ----------
    sdf: sarracen.SarracenDataFrame
        Must contain columns: x, y, z, h.
        if hfact is None or kernel is None, will get from sdf.
        
    locs: np.ndarray
        (3) or (..., 3)-shaped array determining the location for neighbour counting.
        
    kernel: sarracen.kernels.base_kernel
    kernel_rad: float
        radius of the smoothing kernel in unit of smoothing length
        If None, will infer from kernel; if kernel is also None, will use the one in sdf.

    hs_at_locs : None|np.ndarray
        The smoothing length at the locs.
        *** if provided, will search for neighbours
            using      the smoothing length of the locations and sdf_kdtree,
            instead of the smoothing length of the particles.
    
    sdf_kdtree: None|kdtree.KDTree
        kdTree for the particles.
        Only used when searching by h of the locs (i.e. when given hs_at_locs)
        if None, will build one.
        
    ndim: int
        dimension of the space. Default is 3 (for 3D).
        DO NOT TOUCH THIS UNLESS YOU KNOW WHAT YOU ARE DOING.
        
    xyzs_names_list: list
        list of names of the columns that represents x, y, z axes (i.e. coord axes names)
        Make sure to change this if your ndim is not 3.

    verbose: int
        How much warnings, notes, and debug info to be print on screen. 
        
    Returns
    -------
    ans: float or np.ndarray
        Depending on the shape of locs, returns float or array of float.
    """

    do_squeeze : bool = False
    
    # init
    if kernel_rad is None:
        if kernel is None:
            kernel = sdf.kernel
        kernel_rad = float(kernel.get_radius())
    locs = np.array(locs, copy=False, order='C')
    if sdf is not None:
        xyzs = np.array(sdf[xyzs_names_list], copy=False, order='C')    # (npart, ndim)-shaped

    
    if hs_at_locs is not None:
        # search using provided h with kdTree
        if sdf_kdtree is None: sdf_kdtree = kdtree.KDTree(xyzs)
        neigh_inds_list  = sdf_kdtree.query_ball_point(locs, hs_at_locs * kernel_rad)
        return np.array([len(neigh_inds) for neigh_inds in neigh_inds_list])
    else:

        hs   = np.array(sdf['h'], copy=False, order='C')    # (npart,)-shaped
        #hw_rad = kernel_rad * hs    # h * w_rad
    
        # fix input shapes
        if locs.ndim == 1:
            locs = locs[np.newaxis, :]
            do_squeeze = True
        else:
            do_squeeze = False
    
        nlocs = locs.shape[0]
        npart = xyzs.shape[0]
        
        # sanity checks
        if is_verbose(verbose, 'warn'):
            if ndim != 3:
                say('warn', None, verbose,
                    f"You have set ndim={ndim}, which assumes a {ndim}D world instead of 3D.",
                    "if the simulation is 3D, this means that the following calculations will be wrong.",
                    "Are you sure you know what you are doing?",
                )
            if xyzs.shape != (npart, ndim):
                say('warn', None, verbose,
                    f"xyzs.shape={xyzs.shape} is not (npart, ndim)={(npart, ndim)}!",
                    "This is not supposed to happen and means that the following calculations will be wrong.",
                    "Please check input to this function.",
                )
    
        
        ans = np.zeros(nlocs, dtype=int)
        for i in range(nlocs):
            ans[i] = np.count_nonzero(
                np.sum((locs[i] - xyzs)**2, axis=-1)**0.5 / hs <= kernel_rad
            )

        if do_squeeze: ans = np.squeeze(ans)
    
        return ans





@jit(nopython=True)
def _get_sph_interp_phantom_np_basic(
    locs : np.ndarray,
    vals : np.ndarray,
    xyzs : np.ndarray,
    hs   : np.ndarray,
    hfact: float,
    kernel_w  : numba.core.registry.CPUDispatcher,
    kernel_rad: float,
    ndim : int = 3,
) -> np.ndarray:
    """SPH interpolation subprocess. Most basic form.

    Basic version uses basic kernel interpolation: A =  <A> = \sum_j A_j w(q_j) / h_fact**ndim,
        with the assumption of Phantom h: rho = hfact**ndim * (m / h**ndim)

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
    hfact: float,
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
    return ans_s / (hfact**ndim)







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

    Improved version corrects for zero-th order error in the basic version: A = <A> / <1> = \sum_j A_j w(q_j) / \sum_j w(q_j),
        with the assumption of Phantom h: rho = hfact**ndim * (m / h**ndim)
        
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
    hfact    : float = None,
    ndim     : int = 3,
    xyzs_names_list : list = ['x', 'y', 'z'],
    method   : str = 'improved',
    verbose  : int = 3,
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
        \\braket{A} (\\mathbf{r}) 
        \\equiv \sum_{j} \\frac{m_j A_j}{\\rho_j h_j^d} w(q_j(\\mathbf{r}))
        = \\frac{1}{h_\\mathrm{fact}^d} \\sum_{j} A_j w(q_j(\\mathbf{r}))
    where rho = hfact**d * (m / h**d) is assumed.

    Taylor expansion of the above shows that
        \\braket{A}
        = A \\braket{1} + \\nabla A \\cdot (\\braket{\\mathbf{r}} - \\mathbf{r}\\braket{1}) + \\mathcal{O}(h^2)

    So,
        A
        \\approx \\frac{\\braket{A}}{\\braket{1}}
        = \\frac{ \\sum_{j} A_j w(q_j(\\mathbf{r})) }{\\sum_{j} w(q_j(\\mathbf{r}))}
    
        
    
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
        Depending on the shape of locs and val_names, returns float or array of float.
    """

    
    # init
    npart = len(sdf)
    if kernel is None:
        kernel = sdf.kernel
    kernel_rad = float(kernel.get_radius())
    locs = np.array(locs, copy=False, order='C')
    vals = np.array(sdf[val_names], copy=False, order='C')
    xyzs = np.array(sdf[xyzs_names_list], copy=False, order='C')    # (npart, ndim)-shaped array
    hs   = np.array(sdf['h'], copy=False, order='C')                # (npart,)-shaped array

    
    # fix input shapes
    do_squeeze = False
    if locs.ndim == 1:
        locs = locs[np.newaxis, :]
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]
        do_squeeze = True
    if xyzs.ndim == 1:
        xyzs = xyzs[:, np.newaxis]

    
    # sanity checks

    # warn if try to interp unexpected quantities
    if is_verbose(verbose, 'warn'):
        val_names_set = {val_names} if isinstance(val_names, str) else set(val_names)
        if val_names_set.difference({'rho', 'u', 'vx', 'vy', 'vz', 'vr'}):
            say('warn', None, verbose,
                "Kernel interpolation should be used with conserved quantities (density, energy, momentum),",
                f"but you are trying to do it with '{val_names}', which could lead to problematic results.",
            )
        if 'rho' in val_names_set:
            say('warn', None, verbose,
                "You are kernel interpolating density 'rho'.",
                "Consider using get_rho_from_h() instead to directly calc density from smoothing length, as phantom itself would have done.",
            )
        if ndim != 3:
            say('warn', None, verbose,
                f"You have set ndim={ndim}, which assumes a {ndim}D world instead of 3D.",
                "if the simulation is 3D, this means that the following calculations will be wrong.",
                "Are you sure you know what you are doing?",
            )
        if xyzs.shape != (npart, ndim):
            say('warn', None, verbose,
                f"xyzs.shape={xyzs.shape} is not (npart, ndim)={(npart, ndim)}!",
                "This is not supposed to happen and means that the following calculations will be wrong.",
                "Please check input to this function.",
            )


    # calc
    if method in {'improved'}:
        ans = _get_sph_interp_phantom_np(locs, vals, xyzs, hs, kernel.w, kernel_rad, ndim)
    elif method in {'basic'}:
        if hfact is None:
            hfact = float(sdf.params['hfact'])
        ans = _get_sph_interp_phantom_np_basic(locs, vals, xyzs, hs, hfact, kernel.w, kernel_rad, ndim)
    else:
        raise ValueError("Unrecognized 'method' parameter. See function doc for help.")
        
    if do_squeeze:
        ans = np.squeeze(ans)

    return ans










def get_sph_gradient_phantom(
    sdf      : sarracen.sarracen_dataframe.SarracenDataFrame,
    val_names: str|list,
    locs     :   None|np.ndarray = None,
    vals_at_locs:None|np.ndarray = None,
    hs_at_locs  :None|np.ndarray = None,
    kernel   :   None|sarracen.kernels.BaseKernel = None,
    hfact    :   None|float = None,
    sdf_kdtree : None|kdtree.KDTree = None,
    ndim     : int = 3,
    xyzs_names_list : list = ['x', 'y', 'z'],
    #method   : str = 'improved',
    parallel : bool = False,
    verbose  : int = 3,
) -> np.ndarray:
    """Getting the gradient for each SPH particles.

    Assuming Phantom.
        (That is, the smoothing length h is dynamically scaled with density rho using
        rho = hfact**d * (m / h**d)
        for d-dimension and constant hfact.)


    Theories:
    In SPH kernel interpolation theories, for arbitrary quantity A,
        \nabla A \approx \braket{\nabla A} - A \braket{\nabla 1}
        = \frac{1}{h_\mathrm{fact}^d} \sum_{j} \frac{A_j-A}{h_j} \frac{dw}{dq}(q_j) \hat{\mathbf{r}}
    where rho = hfact**d * (m / h**d) is assumed.
        
    
    Parameters
    ----------
    sdf: sarracen.SarracenDataFrame
        Must contain columns: x, y, z, h.
        if hfact is None or kernel is None, will get from sdf.
        
    val_names: str
        Column label of the target smoothing data in sdf
        
    locs: None|np.ndarray
        (..., 3)-shaped array determining the location for interpolation.
        if None, will use ALL particle locations in sdf.
        if supplied, must supply vals_at_locs as well.

    vals_at_locs, hs_at_locs: None|np.ndarray
        Give either none of (locs, vals_at_locs, hs_at_locs), or all three.
        The values and smoothing length interpolated at locs.
        
    kernel: sarracen.kernels.base_kernel
        Smoothing kernel for SPH data interpolation.
        If None, will use the one in sdf.
        
    hfact: float
        constant factor for h.
        If None, will use the one in sdf.params['hfact'].

    sdf_kdtree: None|kdtree.KDTree
        kdTree for the particles.
        if None, will build one.
    
    ndim: int
        dimension of the space. Default is 3 (for 3D).
        DO NOT TOUCH THIS UNLESS YOU KNOW WHAT YOU ARE DOING.
        
    xyzs_names_list: list
        list of names of the columns that represents x, y, z axes (i.e. coord axes names)
        Make sure to change this if your ndim is something other than 3.

    parallel: bool
        Whether to parallel neighbour search process.

    verbose: int
        How much warnings, notes, and debug info to be print on screen. 
        
    Returns
    -------
    ans: float or np.ndarray
        Depending on the shape of locs and val_names, returns float or array of float.
    locs : (nlocs, 1,  ndim)-shaped np.ndarray,
    vals : (1, npart, nvals)-shaped np.ndarray,
    xyzs : (1, npart,  ndim)-shaped np.ndarray,
    hs   : (1, npart,      )-shaped np.ndarray,
    kernel_w  : sarracen.kernels.BaseKernel.w
    kernel_rad: float
        smoothing kernel radius in unit of h (outside this w goes to 0)
    ndim : int = 3

    Returns
    -------
    ans  : (nlocs, ndim, nvals)-shaped np.ndarray
    """

    xyzs = np.array(sdf[xyzs_names_list], copy=False, order='C') # shape=(npart, ndim)
    vals = np.array(sdf[val_names], copy=False, order='C') # shape=(npart, nvals)
    hs   = np.array(sdf['h'], copy=False, order='C')  # (npart,)

    
    if vals.ndim == 1:
        vals = vals[:, np.newaxis]

    if kernel is None:
        kernel = sdf.kernel
    if hfact is None:
        hfact = float(sdf.params['hfact'])
    if sdf_kdtree is None:
        sdf_kdtree = kdtree.KDTree(xyzs)
    if locs is None:
        locs = xyzs
        vals_at_locs = vals
        hs_at_locs   = hs
    if vals_at_locs is None:
        raise NotImplementedError()


    
    # h * w_rad
    kernel_rad = float(kernel.get_radius())
    hw_rad_at_locs = kernel_rad * hs_at_locs
    dw_dq = get_dw_dq(kernel, ndim=ndim)
    
    
    nlocs = locs.shape[0]
    npart = vals.shape[0]
    nvals = vals.shape[1]
    ans = np.zeros((nlocs, ndim, nvals), dtype=vals.dtype)


    nworker = -1 if parallel else 1
    if parallel:
        say('warn', None, verbose, "Only half of the process is parallelized. Pending improvements.")
    
    for i, js in enumerate(sdf_kdtree.query_ball_point(locs, r=hw_rad_at_locs, workers=nworker)):
        # i   is the index of the point where we are calculating gradient for
        # js are the indexes of its neighbours
        
        rs_ij = locs[i] - xyzs[js]    # shape=(njs, ndim)
        rs_ij_norm = np.sum(rs_ij**2, axis=-1)**0.5    # shape=(njs,)
        
        # some items in rs_ij_norm (usually the first one) will be 0
        # because it's the same particle.
        # let's remove that
        js_ind_cleaned = np.where(rs_ij_norm)
        js = np.array(js)[js_ind_cleaned]
        rs_ij = rs_ij[js_ind_cleaned]
        rs_ij_norm = rs_ij_norm[js_ind_cleaned]

        rs_ij_hat = rs_ij/rs_ij_norm[:, np.newaxis]    # shape=(njs, ndim)
        qs_ij = rs_ij_norm / hs[js]    # shape=(njs,)

        # get the gradient for i-th particle
        ans[i] = np.sum(
            (
                (vals[js] - vals_at_locs[i]) / hs[js][:, np.newaxis]
                * dw_dq(qs_ij, ndim)[:, np.newaxis]
            )[:, np.newaxis, :]
            * rs_ij_hat[:, :, np.newaxis],
        axis=0)

    
    ans /= hfact**ndim
    return ans






# set alias
get_sph_interp   = get_sph_interp_phantom
get_sph_gradient = get_sph_gradient_phantom