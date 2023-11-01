#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for geometries.

Mostly 3D Cartesian vector transforms.

Owner: Chunliang Mu
"""



# Init


import numpy as np
from numba import jit



# Functions


@jit(nopython=True)
def get_r_from_loc(loc) -> float:
    """Return norm of a 3D vector.
    
    Parameters
    ----------
    loc: 3-element list/array.
    """
    if len(loc) != 3:
        raise ValueError(f"Input vector dim {len(loc)} is not 3.")
    return (loc[0]**2 + loc[1]**2 + loc[2]**2)**0.5


@jit(nopython=True)
def get_dist2_between_2pt(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """Return distance squared between two N-dimensional points (arrays).
    
    Parameters
    ----------
    pt1, pt2: (..., N)-dimensional numpy array.
    """
    #pt1 = np.array(pt1, copy=False)
    #pt2 = np.array(pt2, copy=False)
    return np.sum((pt2 - pt1)**2, axis=-1)


@jit(nopython=True)
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
    