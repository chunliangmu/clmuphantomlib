#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for geometries.

Mostly 3D Cartesian vector transforms.

Owner: Chunliang Mu
"""



# Init


import numpy as np
from numpy import pi
from numba import jit



# Functions


@jit(nopython=False)
def get_dist2_between_2pt(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """Return distance squared between two N-dimensional points (arrays).
    
    Parameters
    ----------
    pt1, pt2: (..., N)-dimensional numpy array.
    """
    #pt1 = np.array(pt1, copy=False)
    #pt2 = np.array(pt2, copy=False)
    return np.sum((pt2 - pt1)**2, axis=-1)


@jit(nopython=False)
def get_norm_of_vec(vec: np.ndarray) -> np.ndarray|float:
    """Return the norm squared of a N-dimensional points (arrays).
    
    Parameters
    ----------
    vec: (..., N)-dimensional numpy array.
    """
    return np.sum(vec**2, axis=-1)**0.5



#@jit(nopython=False)
#def get_r_from_loc(loc) -> float:
#    """[Deprecated] Return norm of a 3D vector.
#
#    Deprecated: Use get_norm_of_vec(vec) instead
#    
#    Parameters
#    ----------
#    loc: 3-element list/array.
#    """
#    return get_norm_of_vec(loc)




@jit(nopython=False)
def get_closest_pt_on_line(pt0: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Return the closest point on a line to another point(s) pt0.
    
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
    #pt0 = np.array(pt0, copy=False)
    pt1 = line[0] #np.array(line[0], copy=False)
    pt2 = line[1] #np.array(line[1], copy=False)
    t_0 = np.sum((pt0 - pt1) * (pt2 - pt1), axis=-1) / np.sum((pt2 - pt1)**2, axis=-1)
    X_t = pt1 + t_0.reshape((*t_0.shape,1)) * (pt2 - pt1)
    return X_t






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
