#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for geometries.

Mostly 3D Cartesian vector transforms.

Owner: Chunliang Mu
"""



# Init


import math
import numpy as np
from numpy import pi
from numpy import typing as npt
import numba
from numba import jit




# Functions



@jit(nopython=True, fastmath=True)
def get_floor_nb(x: np.float64 | npt.NDArray[np.float64]) -> np.int64 | npt.NDArray[np.int64]:
    if isinstance(x, float):
        return math.floor(x)
    else:
        return np.floor(x).astype(np.int64)




@jit(nopython=True)
def get_dist2_between_2pt_nb(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray|np.float64:
    """Return distance squared between two N-dimensional points (arrays). numba version.
    
    Parameters
    ----------
    pt1, pt2: (..., N)-dimensional numpy array.
    """
    return np.sum((pt2 - pt1)**2, axis=-1)

get_dist2_between_2pt = get_dist2_between_2pt_nb





@jit(nopython=True)
def get_norm_of_vec_nb(vec: np.ndarray) -> np.ndarray|np.float64:
    """Return the norm squared of a N-dimensional points (arrays). numba version.
    
    Parameters
    ----------
    vec: (..., N)-dimensional numpy array.
    """
    return np.sum(vec**2, axis=-1)**0.5

get_norm_of_vec = get_norm_of_vec_nb





@jit(nopython=True)
def get_closest_pt_on_line_nb(pt0: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Return the closest point on a line to another point(s) pt0. numba version.
    
    Parameters
    ----------
    pt0: N or (..., N)-dimensional numpy array.
        An N-dimensional point, or M points of N dimensions.
    
    line: (2, N)-dimensional array_like, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        
    Returns
    -------
    X_t: (..., N)-dimensional np.ndarray
    """
    if len(line) != 2:
        raise ValueError("Input var 'line' should have 2 points (i.e. with shape (2, N)), but ", len(line), " is not 2.")
    #pt0 = np.asarray(pt0)
    pt1 = line[0] #np.asarray(line[0])
    pt2 = line[1] #np.asarray(line[1])
    t_0 = np.sum((pt0 - pt1) * (pt2 - pt1), axis=-1) / np.sum((pt2 - pt1)**2, axis=-1)
    X_t = pt1 + t_0.reshape((*t_0.shape,1)) * (pt2 - pt1)
    return X_t

get_closest_pt_on_line = get_closest_pt_on_line_nb





def get_dist2_from_pts_to_line(pt0: np.ndarray, line: np.ndarray) -> np.ndarray|np.float64:
    """Return the distance squared between a (series of) point and a line.
    
    Parameters
    ----------
    pt0: N or (M, N)-shaped numpy array.
        An N-dimensional point, or M points of N dimensions.
    
    line: (2, N)-shaped array_like, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        
    Returns
    -------
    dist2: (M,)-shaped np.ndarray or np.float64
    """
    return get_dist2_between_2pt(pt0, get_closest_pt_on_line(pt0, line))



@jit(
    numba.types.float64(
        numba.types.Array(numba.types.float64, 1, 'A', readonly=True),
        numba.types.Array(numba.types.float64, 2, 'A', readonly=True),
    ),
    nopython=True)
def get_dist2_from_pt_to_line_nb(pt0: np.ndarray, line: np.ndarray) -> np.float64:
    """Return the distance squared between ONE point and a line. numba version.

    Read-only input version.
    No sanity check. Assumes specific input array shape.
    
    Parameters
    ----------
    pt0: N-shaped numpy array.
        An N-dimensional point, or M points of N dimensions.
    
    line: (2, N)-shaped array_like, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        
    Returns
    -------
    dist2: np.float64
    """

    # get closest point on line
    pt1 = line[0] #np.asarray(line[0])
    pt2 = line[1] #np.asarray(line[1])
    t_0 = np.sum((pt0 - pt1) * (pt2 - pt1), axis=-1) / np.sum((pt2 - pt1)**2, axis=-1)
    X_t = pt1 + t_0 * (pt2 - pt1)
    # get distance between these two points
    return np.sum((X_t - pt0)**2, axis=-1)




    

@jit(nopython=True)
def get_ray_unit_vec_nb(ray: np.ndarray) -> np.ndarray:
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
    ray_unit_vec = ray[1, :] - ray[0, :]
    ray_unit_vec = ray_unit_vec / np.sum(ray_unit_vec**2)**0.5
    return ray_unit_vec

get_ray_unit_vec = get_ray_unit_vec_nb





def get_rays_unit_vec(rays: np.ndarray) -> np.ndarray:
    """Get unit vector of a list of rays.
    
    Parameters
    ----------
    ray: (M, 2, N)-dimensional array_like, i.e. [pt1, pt2]
        2 points required to determine a line.
        The line is described as X(t) = pt1 + t*(pt2-pt1)
        
    Returns
    -------
    ray_unit_vec: (M, N)-dimensional np.ndarray
        unit vector of ray
    """
    ray = np.asarray(rays)
    ray_unit_vec = ray[..., 1, :] - ray[..., 0, :]
    ray_unit_vec = ray_unit_vec / np.sum(ray_unit_vec**2, axis=-1)[:, np.newaxis]**0.5
    return ray_unit_vec



@jit(nopython=True)
def get_rand_unit_vecs(no_vec: int, cos_theta_mid: None|float = None, cos_theta_delta: None|float = None) -> np.ndarray:
    """Generate a series of unit vectors pointing at random directions.
        
    Parameters
    ----------
    no_vec: int
        number of unit vecs to be generated
    cos_theta_mid  : (optional) float in (-1., 1.)
    cos_theta_delta: (optional) float in [ 0., 1.)
        if both supplied, will only generate directions with cos_theta in between cos_theta_mid +/- cos_theta_delta
        Warning: results may contain np.nan if cos_theta_mid +/- cos_theta_delta falls outside the range of [-1., 1.]

    
    Returns
    -------
    unit_vecs: (no_vec, 3)-shaped array
    """
    phis       = np.random.uniform( 0., 2*pi, no_vec)
    cos_thetas = np.random.uniform(-1.,   1., no_vec)
    if cos_theta_mid is not None and cos_theta_delta is not None:
        cos_thetas = cos_theta_mid + cos_thetas * cos_theta_delta
    sin_thetas = (1 - cos_thetas**2)**0.5
    unit_vecs = np.column_stack((
        sin_thetas * np.sin(phis),
        sin_thetas * np.cos(phis),
        cos_thetas,
    ))
    return unit_vecs



def get_rays(orig_vecs: np.ndarray, unit_vecs: np.ndarray) -> np.ndarray:
    """Generate ray object from origin vector and unit vector.

    orig_vecs should have the same shape as unit_vecs.
        alternatively, unit_vecs can be of the shape (ndim,) and orig_vecs can be (..., ndim).
        if orig_vecs is of the shape (ndim,), then the unit_vecs cannot have more than 2 axes.
        (... just use the same shape)
    

    Parameters
    ----------
    orig_vecs: (ndim,) or (no_ray, ndim)-shaped np.ndarray
    unit_vecs: (ndim,) or (no_ray, ndim)-shaped np.ndarray

    Returns
    -------
    (2, ndim) or (no_ray, 2, ndim)-shaped np.ndarray
    """
    if   len(orig_vecs.shape) == 1 and len(unit_vecs.shape) == 2:
        return np.array([[orig_vecs, orig_vecs + unit_vec] for unit_vec in unit_vecs])
    else:
        return np.stack((orig_vecs, orig_vecs + unit_vecs), axis=-2)