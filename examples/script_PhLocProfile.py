#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Scripts for analyzing of phantom outputs.

This script writes json files for one dump
    to plot params vs distance to the primary star, averaged over equatorial plane

"""


# ## Imports & Settings

# In[2]:


#%matplotlib inline
import math
import numpy as np
from numpy import pi
#import pandas
from astropy import units
from astropy import constants as const
import matplotlib.pyplot as plt
import matplotlib as mpl
#from moviepy.editor import ImageSequenceClip


# In[3]:


# import modules listed in ./lib/

import clmuphantomlib as mupl
from clmuphantomlib.readwrite import json_load, json_dump
from clmuphantomlib.settings import DEFAULT_SETTINGS as settings
from clmuphantomlib.log import is_verbose, say
from clmuphantomlib.units_util import set_as_quantity


#     ## import modules in arbitrary directory
#     
#     #import sys
#     
#     ## path to my python module lib directory
#     ## *** CHECK THIS! *** #
#     #SRC_LIB_PATH = sys.path[0] + '/lib'
#     
#     #if SRC_LIB_PATH not in sys.path:
#     #    sys.path.append(SRC_LIB_PATH)
#     ##print(*sys.path, sep='\n')    # debug
#     #print(
#     #    "\n*   Please Make sure my module files are located in this directory (or change the SRC_LIB_PATH variable):",
#     #    f"\n{SRC_LIB_PATH = }\n"
#     #)

# In[4]:


# parallels & optimizations


#import os
## Fixing stupid numba killing kernel
## See here https://github.com/numba/numba/issues/3016
#os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
#from numba import njit, prange


from multiprocessing import cpu_count, Pool #Process, Queue
NPROCESSES = 1 if cpu_count() is None else max(cpu_count(), 1)


# In[5]:


# settings
#
#   imported from script_input.py file

from script_PhLocProfile__input import verbose, ray_no, PHOTOSPHERE_TAU, JOB_PROFILES
from _sharedFuncs import mpdf_read


# set metadata
with open("_metadata__input.json", 'r') as f:
    metadata = mupl.json_load(f)
metadata['Title'] = "Getting photosphere parameters along rays"
metadata['Description'] = f"""Tracing {ray_no} rays to get h, rho, u, T, vr etc. params from them."""


plt.rcParams.update({'font.size': 20})


# print debug info
if __name__ == '__main__' and is_verbose(verbose, 'note'):
    # remember to check if name is '__main__' if you wanna say anything
    #    so when you do multiprocessing the program doesn't freak out
    say('note', "script", verbose, f"Will use {NPROCESSES} processes for parallelization")
    


# # Analysis

# ## Photosphere size vs time

# In[6]:


def write_ph_pars(
    job_profile : dict,
    file_indexes : np.ndarray,
    ray_no: int, # no of rays on xy-axis
    #rays: np.ndarray,    # list of rays
    eoses : (mupl.eos_base.EoS_Base, mupl.eos_mesa.EoS_MESA_opacity),
    photosphere_tau = PHOTOSPHERE_TAU,
    verbose : int = 2,
):

    """Writing the photosphere locations of each dump to json files.

    Notes:
    Using mpdf.params['hfact']
    """
    
    
    mpdf = mupl.MyPhantomDataFrames()

    
    job_name = job_profile['job_name']

    eos, eos_opacity = eoses


    # init rays unit vec
    phis = np.linspace(0., 2.*pi, ray_no, endpoint=False)
    ray_unit_vecs = np.column_stack((np.sin(phis), np.cos(phis), np.zeros_like(phis)))

    # main
    for file_index in file_indexes:

        # read data
        mpdf = mpdf_read(job_name, file_index, eos_opacity, mpdf=mpdf, reset_xyz_by='CoM', verbose=verbose)
        hfact = mpdf.params['hfact']
        mpart = mpdf.params['mass']
        kernel_radius = mpdf.data['gas'].kernel.get_radius()
        # init rays
        star_loc = np.array(mpdf.data['sink'][['x', 'y', 'z']].iloc[0])
        rays = np.stack((star_loc + np.zeros_like(ray_unit_vecs), star_loc + ray_unit_vecs), axis=1)

        
        # init answer dict / array
        pars_on_ray = { # [legend][par_name][time]
            'dump_info': {
                'time' : mpdf.get_time(),
                'orbsep_Rsun': mpdf.get_orb_sep(),
                'nparttot': mpdf.params['nparttot'],
                'sinks_locs': np.array(mpdf.data['sink'][['x', 'y', 'z']]) * mpdf.units['dist'],
            },
            'data': {},
            'rays': None,
        }
        

        sdf = mpdf.data['gas']
        hs_all = np.array(sdf['h'])
        pts_all = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array
        min_rad = 10.
        max_rad = 3e5

        # get maximun radius
        if False:
            #for ray in rays:
            pts_on_ray = mupl.get_closest_pt_on_line(pts, ray)
            sdf_selected_indices = (np.sum((pts - pts_on_ray)**2, axis=-1) <= (kernel_radius * hs)**2)
            max_rad_candidate = np.max(np.sum(np.array(mpdf.data['gas'].iloc[sdf_selected_indices][['x', 'y']])**2, axis=1))**0.5
            max_rad = max(max_rad, max_rad_candidate) 


        samples_no = 1000
        pars_on_ray['data']['R1_on_ray'] = set_as_quantity(
            np.logspace(np.log10(min_rad), np.log10(max_rad), samples_no), mpdf.units['dist'])
        pars_on_ray['data'][  'rho_on_ray'] = np.full((ray_no, samples_no), np.nan)
        pars_on_ray['data'][    'u_on_ray'] = np.full((ray_no, samples_no), np.nan)
        pars_on_ray['data'][   'vr_on_ray'] = np.full((ray_no, samples_no), np.nan)
        pars_on_ray['data']['kappa_on_ray'] = np.full((ray_no, samples_no), np.nan)


        # construct rays_dict
        for i, ray in enumerate(rays):
            ray_unit_vec = mupl.light.get_ray_unit_vec(ray)

            # optimization- first select only the particles affecting the ray
            #  because interpolation of m points with N particles scales with O(N*m),
            #  reducing N can speed up calc significantly
            pts_on_ray = mupl.get_closest_pt_on_line(pts_all, ray)
            sdf_selected_indices = (np.sum((pts_all - pts_on_ray)**2, axis=-1) <= (kernel_radius * hs_all)**2)
            if is_verbose(verbose, 'debug'):
                say('debug', 'write_ph_pars()', verbose,
                    f"{np.count_nonzero(sdf_selected_indices)} particles are close enough to the ray to have effects."
                )
            sdf = mpdf.data['gas'].iloc[sdf_selected_indices]
            pts = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array


            # get optical depth
            if is_verbose(verbose, 'debug'):
                say('debug', 'write_ph_pars()', verbose, f"{ray = }")
            pts_on_ray, dtaus, pts_order = mupl.light.get_optical_depth_by_ray_tracing_3D(sdf=sdf, ray=ray)
            if False:
                # commented
                photosphere, (pts_waypts, pts_waypts_t, taus_waypts) = mupl.get_photosphere_on_ray(
                    pts_on_ray, dtaus, pts_order, sdf, ray,
                    calc_params = ['loc', 'R1', 'rho', 'u', 'h', 'T', 'kappa'],
                    hfact = hfact, mpart=mpart, eos=eos, sdf_units=mpdf.units,
                    ray_unit_vec=ray_unit_vec, verbose=verbose,
                    return_as_quantity=False,
                )
            R1_on_ray  = pars_on_ray['data']['R1_on_ray'].value
            #tau_on_ray = np.interp(R1_on_ray, pts_waypts_t[::-1], taus_waypts[::-1])
            pts_on_ray = ray[0][np.newaxis, :] + R1_on_ray[:, np.newaxis] * ray_unit_vec[np.newaxis, :]
            pars_on_ray['data'][  'rho_on_ray'][i] = mupl.sph_interp.get_sph_interp(sdf, 'rho', pts_on_ray, verbose=verbose)
            pars_on_ray['data'][    'u_on_ray'][i] = mupl.sph_interp.get_sph_interp(sdf, 'u'  , pts_on_ray, verbose=verbose)
            pars_on_ray['data'][   'vr_on_ray'][i] = mupl.sph_interp.get_sph_interp(sdf, 'vr' , pts_on_ray, verbose=verbose)
            pars_on_ray['data']['kappa_on_ray'][i] = mupl.sph_interp.get_sph_interp(sdf,'kappa',pts_on_ray, verbose=1)

        pars_on_ray['data'][  'rho_on_ray'] = set_as_quantity(
            pars_on_ray['data'][  'rho_on_ray'], mpdf.units['density'])
        pars_on_ray['data'][    'u_on_ray'] = set_as_quantity(
            pars_on_ray['data'][    'u_on_ray'], mpdf.units['specificEnergy'])
        pars_on_ray['data'][   'vr_on_ray'] = set_as_quantity(
            pars_on_ray['data'][   'vr_on_ray'], mpdf.units['speed'])
        pars_on_ray['data']['kappa_on_ray'] = set_as_quantity(
            pars_on_ray['data']['kappa_on_ray'], mpdf.units['opacity'])
        pars_on_ray['data'][  'T_on_ray'] = eos.get_temp(
            pars_on_ray['data']['rho_on_ray'], pars_on_ray['data']['u_on_ray'],
            return_as_quantity=True, bounds_error=False)
        
        with open(f"{mpdf.get_filename()}__photosphere-pars-on-ray.json", 'w') as f:
            json_dump(pars_on_ray, f, metadata=metadata, indent=None)
            if verbose: print(f"\n\nWritten to {f.name}\n")

    return None


# ## Main

# In[8]:


# main process

# run main

if __name__ == '__main__':
    
    
    # get ph loc for each dump file
    for job_profile in JOB_PROFILES[:2]:
    
        #file_indexes = job_profile['file_indexes']
        file_indexes = [2000, 4800]
        eos = mupl.get_eos(job_profile['ieos'], job_profile['params'], settings)
        eos_opacity = mupl.eos_mesa.EoS_MESA_opacity(job_profile['params'], settings)
    
        
        if True: #NPROCESSES <= 1:
            
            # single process
    
            write_ph_pars(
                job_profile = job_profile, file_indexes = file_indexes, ray_no = ray_no,
                eoses = (eos, eos_opacity), photosphere_tau = PHOTOSPHERE_TAU, verbose = verbose,
            )
            
        else:
            
            # multi-process

            args = [(
                job_profile,
                [file_index],
                ray_no,
                (eos, eos_opacity),
                PHOTOSPHERE_TAU,
                0,
                ) for file_index in file_indexes
            ]

            with Pool(processes=NPROCESSES) as pool:
                pool.starmap(write_ph_pars, args)
    
    

