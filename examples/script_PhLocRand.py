#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Scripts for analyzing of phantom outputs.

This script analyze the photospheric properties by randomly select rays in 3D space

"""


# # Main

# ## Imports & Settings

# In[2]:


import numpy as np
from numpy import pi
from astropy import units
import matplotlib.pyplot as plt
import matplotlib as mpl
#from moviepy.editor import ImageSequenceClip
#from os import path


# In[3]:


# import my modules listed in ./main/

import clmuphantomlib as mupl
#from main.clmuphantomlib.readwrite import json_load
from clmuphantomlib.log import is_verbose, say
from clmuphantomlib.settings   import DEFAULT_SETTINGS as settings
from clmuphantomlib.units_util import get_val_in_unit #set_as_quantity, get_units_field_name, get_units_cgs
from clmuphantomlib.readwrite  import json_dump, json_load
from clmuphantomlib.eos_mesa   import EoS_MESA_opacity
from clmuphantomlib import MyPhantomDataFrames, get_eos
from clmuphantomlib.light import get_optical_depth_by_ray_tracing_3D, get_photosphere_on_ray
from multiprocessing import cpu_count, Pool #Process, Queue
NPROCESSES = 1 if cpu_count() is None else max(cpu_count(), 1)


# In[4]:


# settings
#
#   imported from script_input.py file


from script_PhLocRand__input import verbose, fps, unitsOut, JOB_PROFILES_DICT, PHOTOSPHERE_TAU, ray_no, cos_theta_sample_no, job_nicknames
from _sharedFuncs import mpdf_read

unitsOutTxt = {  key  : unitsOut[key].to_string('latex_inline') for key in unitsOut.keys() }


# set metadata
with open("_metadata__input.json", 'r') as f:
    metadata = json_load(f)
metadata['Title'] = "Getting photosphere values (temperature, size) with random sampling on rays"
metadata['Description'] = f"""Tracing ({ray_no=}) of rays with random directions.
photosphere is defined as where optical depth reaches {PHOTOSPHERE_TAU}.
Results are the values at the intersection point between the photosphere and the rays.
rays all originates from the primary star (sdf_sink.iloc[0])
"""


plt.rcParams.update({'font.size': 20})
if __name__ == '__main__' and is_verbose(verbose, 'note'):
    # remember to check if name is '__main__' if you wanna say anything
    #    so when you do multiprocessing the program doesn't freak out
    say('note', "script", verbose, f"Will use {NPROCESSES} processes for parallelization")


# In[5]:


def get_rand_rays_unit_vec(ray_no: int, cos_theta_mid: None|float = None, cos_theta_delta: None|float = None) -> np.ndarray:
    """Generate a series of rays pointing at random directions.

    if both cos_theta_mid and cos_theta_delta is supply,
        will only generate directions with cos_theta in between cos_theta_mid +/- cos_theta_delta

    returns: (ray_no, 3)-shaped array
    """
    phis       = np.random.uniform( 0., 2*pi, ray_no)
    cos_thetas = np.random.uniform(-1.,   1., ray_no)
    if cos_theta_mid is not None and cos_theta_delta is not None:
        cos_thetas = cos_theta_mid + cos_thetas * cos_theta_delta
    sin_thetas = (1 - cos_thetas**2)**0.5
    rays = np.column_stack((
        sin_thetas * np.sin(phis),
        sin_thetas * np.cos(phis),
        cos_thetas,
    ))
    return rays


#     # testing get_rand_rays_unit_vec
#     %matplotlib widget
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#     plt.close('all')
#     
#     rays = get_rand_rays_unit_vec(100)
#     
#     lims = (-1.1, 1.1)
#     fig = plt.figure(figsize=(8, 8))
#     ax = Axes3D(fig)
#     ax.scatter(rays[:, 0], rays[:, 1], rays[:, 2])
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.set_zlim(lims)
#     fig.add_axes(ax)
#     plt.show(fig)

# In[6]:


def get_ph_vals(
    vals_names: list,
    mpdf: MyPhantomDataFrames,
    eos: mupl.eos_base.EoS_Base,
    rays_unit_vecs : np.ndarray, # (ray_no, 3)-shaped
    verbose: int,
):
    sdf_all = mpdf.data['gas']
    hs = np.array(sdf_all['h'])
    pts = np.array(sdf_all[['x', 'y', 'z']])    # (npart, 3)-shaped array
    kernel_radius = sdf_all.kernel.get_radius()

    plane_orig_vec = np.array(mpdf.data['sink'][['x', 'y', 'z']].iloc[0])

    # random direction in the sphere
    #rays_unit_vecs = get_rand_rays_unit_vec(ray_no)
    ray_no = len(rays_unit_vecs)
    
    vals_dict = {
        'tau_dust': np.full(ray_no, np.nan),
        'inner_dust_shell_rad': np.full(ray_no, np.nan) * mpdf.units['dist'],
    }
    
    for iray, ray_unit_vec in enumerate(rays_unit_vecs):
        ray = np.array([
            plane_orig_vec,
            plane_orig_vec + ray_unit_vec,
        ])
        
        pts_on_ray = mupl.get_closest_pt_on_line(pts, ray)
        sdf_selected_indices = (np.sum((pts - pts_on_ray)**2, axis=-1) <= (kernel_radius * hs)**2)
        sdf = sdf_all.iloc[sdf_selected_indices]
        
        pts_on_ray, dtaus, pts_order = get_optical_depth_by_ray_tracing_3D(sdf, ray)
        photosphere, waypts_list = get_photosphere_on_ray(
            pts_on_ray, dtaus, pts_order, sdf, ray,
            calc_params = vals_names,
            eos = eos,
            sdf_units = mpdf.units,
            photosphere_tau = PHOTOSPHERE_TAU,
            return_as_quantity=True,
            verbose = 1 if is_verbose(verbose, 'err') else 0,
        )
        for val_name in vals_names:
            if iray == 0:
                # init
                vals_dict[val_name] = np.full((*photosphere[val_name].shape, ray_no), np.nan)
                if isinstance(photosphere[val_name], units.quantity.Quantity):
                    vals_dict[val_name] *= photosphere[val_name].unit
            # save value
            vals_dict[val_name][iray] = photosphere[val_name]

            if 'kappa_dust' in sdf.keys():
                kappa_tol = 1e-7*(units.cm**2/units.g)
                kappa_tol_val = kappa_tol.to_value(mpdf.units['opacity'])
                pts_waypts_t = np.sum((pts_on_ray - ray[0]) * ray_unit_vec, axis=-1) # the higher, the more on the pt2 side (observer)
                # find the furtherest dust-containing particle on the observer's side
                last_dust_part_ordered_indices = np.where(np.logical_and(
                    pts_waypts_t[pts_order] > 0,    # condition 1: on the observer's side (i.e. don't be further than the sink)
                    sdf.iloc[pts_order]['kappa_dust'] > kappa_tol_val,    # condition 2: dust-containing
                ))[0]
                if len(last_dust_part_ordered_indices):
                    # found the dust shell!
                    last_dust_part_ordered_ind = last_dust_part_ordered_indices[-1]
                    vals_dict['tau_dust'][iray] = np.sum(dtaus[pts_order][:last_dust_part_ordered_ind])
                    vals_dict['inner_dust_shell_rad'][iray] = mupl.set_as_quantity(
                        pts_waypts_t[pts_order][last_dust_part_ordered_ind], mpdf.units['dist'])
    vals_dict['cos_theta'] = rays_unit_vecs[:, 2]
    vals_dict['ray_unit_vec'] = rays_unit_vecs
    
    return vals_dict


# In[7]:


def get_photosphere_vals_rand_samples(
    job_nickname: str,
    file_index: int,
    ray_no: int,
    vals_names: list = ['R1', 'T'],
    cos_theta_sample_no: int|None = None,
    mpdf: MyPhantomDataFrames = None,
    verbose: int = 3,
) -> dict:
    
    job_profile = JOB_PROFILES_DICT[job_nickname]
    job_name    = job_profile['job_name']
    params      = job_profile['params']
    ieos = job_profile['ieos']
    eos  = get_eos(ieos, params, settings)
    eos_opacity = EoS_MESA_opacity(params, settings)
    
    mpdf = mpdf_read(
        job_name, file_index, eos_opacity, mpdf,
        kappa_gas = 2e-4*(units.cm**2/units.g) if file_index != 0 else 0.*(units.cm**2/units.g),
        verbose=verbose)

    if cos_theta_sample_no is None:

        rays_unit_vecs = get_rand_rays_unit_vec(ray_no)
        vals_dict = get_ph_vals(vals_names, mpdf, eos, rays_unit_vecs, verbose=verbose)
    
        if is_verbose(verbose, 'note'):
            say('note', f'{mpdf.get_filename()}', verbose,
                *[f'{val_name} = {np.average(vals_dict[val_name])} +/- {np.std(vals_dict[val_name])}' for val_name in vals_names]
            )
    else:
        # A fixed amount of rays per cos_theta interval will be generated
        #cos_theta_sample_no = 2
        ray_per_cos_theta = int(ray_no/cos_theta_sample_no)
        cos_theta_delta = 1. / cos_theta_sample_no
        cos_thetas = np.linspace(-1+cos_theta_delta, 1-cos_theta_delta, cos_theta_sample_no)
        vals_by_cos_thetas = []
    
        for i, cos_theta_mid in enumerate(cos_thetas):
            rays_unit_vecs = get_rand_rays_unit_vec(ray_per_cos_theta, cos_theta_mid, cos_theta_delta)
            vals_dict = get_ph_vals(vals_names, mpdf, eos, rays_unit_vecs, verbose=verbose)
            #vals_dict['cos_theta'] = rays_unit_vecs[:, 2]
            vals_by_cos_thetas.append(vals_dict)
        
            if is_verbose(verbose, 'note'):
                say('note', f'cos_theta_mid = {cos_theta_mid}', verbose,
                    *[f'{val_name} = {np.average(vals_dict[val_name])} +/- {np.std(vals_dict[val_name])}' for val_name in vals_names]
                )
        
        vals_dict = { key: np.concatenate([data[key] for data in vals_by_cos_thetas]) for key in vals_by_cos_thetas[0].keys()}

    return vals_dict


# In[8]:


def write_ans(job_nickname, file_index, ray_no, verbose):
    """func for parallelize."""
    mpdf = MyPhantomDataFrames()
    vals_dict = get_photosphere_vals_rand_samples(
        job_nickname, file_index, ray_no, vals_names=['R1', 'T', 'h', 'nneigh', 'vr'],
        cos_theta_sample_no=None, mpdf=mpdf, verbose=verbose)
    vals_dict['time'] = mpdf.get_time()
    with open(f"{mpdf.get_filename()}__photosphere-vals.json", 'w') as f:
        json_dump(vals_dict, f, metadata)


# .
# 
# # Main
# 
# .
# 

# In[10]:


if __name__ == '__main__':
    mpdf = MyPhantomDataFrames()

    if NPROCESSES <= 1:
        for job_nickname in job_nicknames:  # '2md', 
            for file_index in JOB_PROFILES_DICT[job_nickname]['file_indexes']: # [0, 2000, 4800, 6400, 8000, 17600]
                write_ans(job_nickname, file_index, ray_no, verbose)
    else:
        args = [
            (job_nickname, file_index, ray_no, 0)
            for job_nickname in job_nicknames
            for file_index in JOB_PROFILES_DICT[job_nickname]['file_indexes']
        ]

        with Pool(processes=NPROCESSES) as pool:
            pool.starmap(write_ans, args)
        
    if is_verbose(verbose, 'note'):
        say('note', '__main__', verbose, f"\n\n\n*** All Done. ***\n\n\n")

