#!/usr/bin/env python
# coding: utf-8

# # Main

# In[1]:


"""Scripts for analyzing of phantom outputs.

This script writes json files for each dump with info necessary to create a plot of the photosphere cross-section.
It does that by shooting a number of rays from the primary star to different directions in the observational plane
     and calculate the intersection point (defined as where tau=1)
The script also produce the plots and a movie.


Author: Chunliang Mu
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
from moviepy.editor import ImageSequenceClip
import json


# In[3]:


# import modules listed in ./lib/

from main import clmuphantomlib as mupl
from main.clmuphantomlib.settings import DEFAULT_SETTINGS as settings


# In[4]:


# parallels & optimizations


#import os
## Fixing stupid numba killing kernel
## See here https://github.com/numba/numba/issues/3016
#os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
#from numba import njit, prange


from multiprocessing import cpu_count, Pool

NPROCESSES = cpu_count()
if NPROCESSES is None:
    NPROCESSES = 1
NPROCESSES = max(NPROCESSES, 1)


# In[5]:


# settings
#
#   imported from script_input.py file

from script_PhLocCircles__input import verbose, PHOTOSPHERE_TAU, JOB_PROFILES
from script_PhLocCircles__input import ray_no, plane_axes_list, box_lim, fps, unitsOut, use_saved_jsons

for job_profile in JOB_PROFILES:
    job_profile['EoS'] = mupl.get_eos(job_profile['ieos'], job_profile['params'], settings)

# set metadata
with open("_metadata__input.json", 'r') as f:
    metadata = mupl.json_load(f)
metadata['Title'] = "Getting photosphere cross-section on a 2D plane"
metadata['Description'] = f"""Tracing {ray_no} of rays on {[plane_ax[:2] for plane_ax in plane_axes_list]} planes.
photosphere is defined as where optical depth reaches {PHOTOSPHERE_TAU}.

*   Note: ['ph_vars']['loc'] is in the order of ['ph_vars']['plane_axes'],
        I.e. if ['ph_vars']['plane_axes'] is 'xzy',
        then ['ph_vars']['loc'] is recorded in order of x, z, y coordinates!
"""


plt.rcParams.update({'font.size': 20})


# print debug info
if verbose >= 2:
    print(f"{metadata = }")
    print(f"   Note: Will use {NPROCESSES} processes for parallelization")
    


#     # Test
# 
#     photosphere_tau = PHOTOSPHERE_TAU
# 
#     job_profile = JOB_PROFILES[0]
#     job_name = job_profile['job_name']
#     file_indexes = job_profile['file_indexes']
#     plot_title_suffix = job_profile['plot_title_suffix']
#     X = job_profile['X']
#     ieos = job_profile['ieos']
# 
# 
#     file_index=file_indexes[12]
#     mpdf = mupl.MyPhantomDataFrames()
#     mpdf.read(
#         job_name, file_index,
#         calc_params=['T', 'kappa', 'R1'],
#         reset_xyz_by="R1",
#         calc_params_params={'ieos': ieos, 'X':X, 'overwrite':False, 'kappa_translate_from_cgs_units':True},
#         verbose=verbose,
#     )
#     mpdf.plot_render(plot_title_suffix=plot_title_suffix,
#         xlim=(-60000, 60000), ylim=(-60000, 60000),
#         norm=mpl.colors.LogNorm(vmin=1e-25, vmax=1e-5, clip=True),
#     )
#     if verbose:
#         print()
#         print(mpdf.get_time())
#         print(mpdf.data['gas'].keys())
#         print(mpdf.data['sink'])

# 
#     
#     
#     # units info (for plot axes title)
#     
#     unitsIn = {    # not used for reading phantom data dumps since it records units in the dump already
#         'dist': units.solRad,
#         'temp': units.K,
#         'flux': units.erg/units.cm**2/units.s,    # equivalent of units.g/units.s**3
#         'opacity': units.cm**2/units.g,
#     }
#     
#     unitsOut = {
#         'dist': units.solRad,
#         'temp': units.K,
#         'flux': units.erg/units.cm**2/units.s,    # equivalent of units.g/units.s**3
#         'opacity': units.cm**2/units.g,
#     }
#     
#     unitsOutTxt = {}
#     
#     for key in unitsOut.keys():
#         unitsOutTxt[key] = unitsOut[key].to_string('latex_inline')
# 

# # Analysis

# ## Get photosphere locations

# In[6]:


def _get_mpdf_photosphere_xsec_subprocess(
    iray, ray, sdf, hfact, mpart, kernel_radius, photosphere_tau, eos, mpdf_units
):
    #ray = rays[iray]
    ray_unit_vec = ray[1, :] - ray[0, :]
    ray_unit_vec = ray_unit_vec / np.sum(ray_unit_vec**2)**0.5

    # optimization- select only the particles affecting the ray
    #  because interpolation of m points with N particles scales with O(N*m),
    #  reducing N can speed up calc significantly
    hs = np.array(sdf['h'])
    pts = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array
    pts_on_ray = mupl.get_closest_pt_on_line(pts, ray)
    sdf_selected_indices = (np.sum((pts - pts_on_ray)**2, axis=-1) <= (kernel_radius * hs)**2)
    sdf = sdf.iloc[sdf_selected_indices]


    # get optical depth
    pts_on_ray, dtaus, pts_order = mupl.get_optical_depth_by_ray_tracing_3D(sdf=sdf, ray=ray)
    photosphere, waypts_list = mupl.get_photosphere_on_ray(
        pts_on_ray, dtaus, pts_order, sdf, ray,
        calc_params = ['loc', 'R1'],
        ray_unit_vec=ray_unit_vec,
    )

    photosphere['index'] = iray
    photosphere['xsec_loc'] = photosphere['loc'][0:2]
    if photosphere['is_found']:
        # ********** THE FOLLOWING LINE (for xsec) NEEDS UPDATE FOR GENERALIZATION **********
        #    IT ONLY WORKS FOR +z DIRECTION PLANE

        photosphere_rho = mupl.sph_interp.get_sph_interp_phantom(sdf, 'rho' , photosphere['loc'])
        photosphere_u = mupl.sph_interp.get_sph_interp_phantom(sdf, 'u' , photosphere['loc'], verbose=0)
        photosphere_h   = hfact * (mpart / photosphere_rho)**(1./3.)
        photosphere['h'] = photosphere_h
        photosphere_loc_add1h = photosphere['loc'] + photosphere_h * ray_unit_vec
        photosphere_loc_min1h = photosphere['loc'] - photosphere_h * ray_unit_vec
        photosphere['xsec_loc_add1h'] = photosphere_loc_add1h[0:2]
        photosphere['xsec_loc_min1h'] = photosphere_loc_min1h[0:2]
        try:
            photosphere['T'] = eos.get_temp(
                rho=mupl.set_as_quantity(photosphere_rho, mpdf_units['density']),
                u=mupl.set_as_quantity(photosphere_u, mpdf_units['specificEnergy']),
                return_as_quantity=False, verbose=0)
        except ValueError:
            photosphere['T'] = np.nan

    else:
        #print("*    Warning: Photosphere not found (len(taus_ordered)=", len(taus_ordered), ")")
        photosphere = None
    return photosphere


# In[7]:


# this func used to be called as get_mpdf_fig_photosphere_cross_section_R1_xyz

def get_mpdf_photosphere_xsec(
    mpdf : mupl.MyPhantomDataFrames,
    plane_orig_vec : np.ndarray = None,
    plane_axes = 'xyz',
    #plane_norm_vec : np.ndarray = np.array([0., 0., 1.]),   # NOT YET IMPLEMENTED!
    ray_no : int = 60,
    photosphere_tau : float = 1.,
    nprocesses : int = 1,
    verbose : int = 2,
) -> dict:
    """Plot a single figure of the photosphere cross-section with the plane.
    
    Plane is determined by plane_norm_vec and plane_orig_vec (sink #1 particle loc by default)
    
    *** EXPERIMENTAL - USE WITH CAUTION ***

    Parameters
    ----------
    mpdf: mupl.MyPhantomDataFrames
        With opacity 'kappa' column calculated.
    
    plot_title_suffix: str
    
    plane_orig_vec: (3,)-shaped numpy array
        determine the origin of the plane *** AFTER THE ROTATION (i.e. with observer at +z) **.
        if None, uses sink #1 particle loc
        
    #plane_norm_vec: (3,)-shaped numpy array
    #    determine the orientation of the plane.
    
    plane_axes: len==3 str / list of char
        e.g. 'xyz' means xy-plane looking from +z direction
    
    ray_no: int
        no of rays shooting from the primary star.
        I.e. determines the resolution of the plot.
        
    photosphere_tau: float
        Defines the photosphere: at what optical depth is it
        
    nprocesses: int:
        set it > 1 to enable multiprocessing.
        
    verbose : int
        How much warnings, notes, and debug info to be print on screen.

    Returns
    -------
    outfilename: str
        saved fig file name.

    """
    # settings & init
    
    if plane_orig_vec is None:
        plane_orig_vec = np.array([mpdf.data['sink'][plane_ax][0] for plane_ax in plane_axes])
    else:
        # ********** THE FOLLOWING LINE NEEDS UPDATE FOR GENERALIZATION **********
        #    IT ONLY WORKS FOR +z DIRECTION PLANE
        pass
        

    # init
    kernel = mpdf.data['gas'].kernel
    kernel_radius = kernel.get_radius()
    hfact = mpdf.params['hfact']
    mpart = mpdf.params['mass']
    
    # swaping axes to make z as the observer loc
    #   deep copy because pandas dataframe is not thread safe for reading
    #   see here https://stackoverflow.com/questions/13592618/python-pandas-dataframe-thread-safe
    sdf_gas = mpdf.data['gas'].copy(deep=True)
    for plane_ax_orig, plane_ax in zip('xyz', plane_axes):
        sdf_gas[plane_ax_orig] = mpdf.data['gas'][plane_ax].copy(deep=True)
    
    # optimization- select only the particles affecting the plane
    # ********** THE FOLLOWING LINE NEEDS UPDATE FOR GENERALIZATION **********
    #    IT ONLY WORKS FOR +z DIRECTION PLANE
    sdf_gas_selected_indices = np.array(abs(sdf_gas['z'] - plane_orig_vec[2]) <= (sdf_gas['h'] * kernel_radius))
    sdf_gas = sdf_gas.iloc[sdf_gas_selected_indices]
    
    
    # init rays
    rays = np.full((ray_no, 2, 3), np.nan)
    rays[:, :] = plane_orig_vec
    thetas = np.linspace(0., 2*pi, ray_no, endpoint=False)
    # ********** THE FOLLOWING LINE NEEDS UPDATE FOR GENERALIZATION **********
    #    IT ONLY WORKS FOR +z DIRECTION PLANE
    rays[:, 1, :] += np.column_stack((np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)))
    
    
    photospheres = {
        'dump_info': {
            'time_yr' : mpdf.get_time(),
            'nparttot': mpdf.params['nparttot'],
            'sinks_locs': {plane_ax: (mpdf.data['sink'][plane_ax]).to_list() * mpdf.units['dist'] for plane_ax in ['x', 'y', 'z']},
        },
        'units_cgs': {
            'dist' : (1.* mpdf.units['dist']).cgs.value,
        },
        'ph_vars': {
            'plane_axes': plane_axes,    # str
            'is_found': np.full(ray_no+1, False),
            'loc': np.full((ray_no+1, 3), np.nan),
            'R1': np.full((ray_no+1,), np.nan),
            'h' : np.full((ray_no+1,), np.nan),
            'T' : np.full((ray_no+1,), np.nan),
            'theta': np.full((ray_no+1,), np.nan),
            'xsec_loc': np.full((ray_no+1, 2), np.nan),
            'xsec_loc_add1h': np.full((ray_no+1, 2), np.nan),
            'xsec_loc_min1h': np.full((ray_no+1, 2), np.nan),
        },
    }
    
    args = [(iray, rays[iray], sdf_gas, hfact, mpart, kernel_radius, photosphere_tau, mpdf.eos, mpdf.units) for iray in range(ray_no)]

    if nprocesses > 1:
        with Pool(processes=NPROCESSES) as pool:
            photospheres_list = pool.starmap(
                _get_mpdf_photosphere_xsec_subprocess,
                args,
            )
    else:
        photospheres_list = []
        for arg in args:
            photospheres_list.append(_get_mpdf_photosphere_xsec_subprocess(*arg))

    for photosphere in photospheres_list:
        if photosphere is not None:
            iray = photosphere['index']
            for key in photosphere.keys():
                if key == 'index':
                    pass
                else:
                    photospheres['ph_vars'][key][iray] = photosphere[key]
            photospheres['ph_vars']['theta'][iray] = thetas[iray]

    photospheres['ph_vars']['loc']            *= mpdf.units['dist']
    photospheres['ph_vars']['h']              *= mpdf.units['dist']
    photospheres['ph_vars']['R1']             *= mpdf.units['dist']
    photospheres['ph_vars']['xsec_loc']       *= mpdf.units['dist']
    photospheres['ph_vars']['xsec_loc_add1h'] *= mpdf.units['dist']
    photospheres['ph_vars']['xsec_loc_min1h'] *= mpdf.units['dist']
    photospheres['ph_vars']['T']              *= mpdf.units['temp']
    

    # make the end point same as beginning point so the final plot looks pretty and sane.
    for key in photospheres['ph_vars'].keys():
        if key not in ['plane_axes']:
            photospheres['ph_vars'][key][-1] = photospheres['ph_vars'][key][0]


    #photospheres['xsec_locs'] = np.array(photospheres['xsec_locs'])
    #photospheres['hs']  = np.array(photospheres['hs'])
    #photospheres['xsec_locs_add1h'] = np.array(photospheres['xsec_locs_add1h'])
    #photospheres['xsec_locs_min1h'] = np.array(photospheres['xsec_locs_min1h'])
    #photospheres['R1s'] = np.array(photospheres['R1s'])
    return photospheres #, plane_orig_vec


# In[8]:


def plot_mpdf_photosphere_xsec(
    #mpdf : mupl.MyPhantomDataFrames,
    photospheres : dict,
    job_name     : str ,
    job_index    : int ,
    plot_title_suffix : str,
    #plane_axes = 'xyz',
    fig = None,
    ax  = None,
    do_legend: bool = True,
    box_lim  : float= 25000.,
    unitsOut : dict = { 'dist': units.solRad, },
    outfilename = None,
    outfilename_noext = None,
) -> str:
    """Plot a single figure of the photosphere cross-section with the plane.
    
    Plane is determined by plane_norm_vec and plane_orig_vec (sink #1 particle loc by default)
    
    *** EXPERIMENTAL - USE WITH CAUTION ***

    Parameters
    ----------
    mpdf: mupl.MyPhantomDataFrames
        With opacity 'kappa' column calculated.
        
    photospheres: dict
        data calc-ed from get_mpdf_fig_photosphere_cross_section_R1_xyz()
        Required keys:
            'xsec_locs': (:, 2)-shaped np.ndarray
            'R1s': list
    
    plot_title_suffix: str
        text to add to the title of the plot

    fig, ax: matplotlib figure and ax
        if ax is None, will generate fig and ax

    do_legend: bool
        whether or not plot legend in ax

    box_lim: float
        x & y lim of the plot.

    unitsOut: dict of units
        Output units. Assuming linear.

    outfilename: str or None
        Give either outfilename or outfilename_noext to save fig to files;
        or set outfilename_noext='' to disable saving fig.
        Name of the output figure file with file extensions.
        (Give this will save fig to one fig designated as outfilename.)
        E.g. None or "figure01.png"
        If None, will auto generate filename.

    outfilename_noext: str or None
        Give either outfilename or outfilename_noext to save fig to files;
        or set outfilename_noext='' and outfilename = '' to disable saving fig.
        outfilename_noext: Name of the output figure file without file extensions.
        (Give this will save fig to two figs: one pdf without title, one png with title.)
        E.g. None or "figure01"
        If None, will auto generate filename.
        If "", will not save fig to file.


    Returns: fig, ax, outfilename
    --------
    outfilename: str
        saved fig file name.

    """
    # plotting

    #toUnitsOut = {}
    unitsOutTxt = {}
    for key in ['dist']:
        #toUnitsOut[ key] = (1.*unitsIn[key]).to_value(unitsOut[key])
        unitsOutTxt[key] = unitsOut[key].to_string('latex')
    plane_axes = photospheres['ph_vars']['plane_axes']
    

    xlim = (-box_lim, box_lim)
    ylim = xlim
    x = photospheres['ph_vars']['xsec_loc'][:, 0].to(unitsOut['dist'])
    y = photospheres['ph_vars']['xsec_loc'][:, 1].to(unitsOut['dist'])
    time = photospheres['dump_info']['time_yr'  ].to(unitsOut['time'])

    outfilename_vectxt = f"R1-{plane_axes[0]}{plane_axes[1]}{plane_axes[2]}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(left=0.18, bottom=0.1, right=0.88, top=0.8)
    
    ax.scatter(x, y, s=1, color='C0', label="photosphere")
    
    # plot sink particles
    ax.scatter(
        photospheres['dump_info']['sinks_locs'][plane_axes[0]].to(unitsOut['dist']),
        photospheres['dump_info']['sinks_locs'][plane_axes[1]].to(unitsOut['dist']),
        marker='.', color='red', label="sinks",
    )
    
    # drawing h error area
    xerr_combined = np.concatenate(
        (photospheres['ph_vars']['xsec_loc_min1h'][:, 0], photospheres['ph_vars']['xsec_loc_add1h'][::-1, 0])).to(unitsOut['dist'])
    yerr_combined = np.concatenate(
        (photospheres['ph_vars']['xsec_loc_min1h'][:, 1], photospheres['ph_vars']['xsec_loc_add1h'][::-1, 1])).to(unitsOut['dist'])
    ax.fill(xerr_combined, yerr_combined, color='C0', alpha=0.15)


    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"{plane_axes[0]} / {unitsOutTxt['dist']}")
    ax.set_ylabel(f"{plane_axes[1]} / {unitsOutTxt['dist']}")
    ax.text(
        0.98, 0.98,
        f"Time = {time:.1f}\n" + \
        f" $R_{{\\rm ph}} = {(np.average(photospheres['ph_vars']['R1'][:-1]**2)**0.5).to_value(unitsOut['dist']):.0f}" + \
        f" \\pm {np.std(photospheres['ph_vars']['R1']).to_value(unitsOut['dist']):.0f} $ {unitsOutTxt['dist']}",
        color = "black", ha = 'right', va = 'top',
        transform=ax.transAxes,
    )
    if do_legend:
        ax.legend(loc="lower right")


    if outfilename_noext is None and outfilename is None:
        # default - output as two fig
        jobfilename = mupl.get_filename_phantom_dumps(job_name=job_name, job_index=job_index)
        outfilename_noext = f"{jobfilename}__photosphere-xsec__{outfilename_vectxt}"
    if outfilename_noext or outfilename:
        if outfilename is None:
            # giiven outfilename_noext -> output fig in both pdf and png
            outfilename = f"{outfilename_noext}.pdf"
            fig.savefig(outfilename)
            outfilename = f"{outfilename_noext}.png"
        # output fig for either given outfilename_noext or outfilename
        ax.set_title(
            f"Photosphere cross-section in {plane_axes[0]}{plane_axes[1]}-plane\n" + \
            f"resolution = {photospheres['dump_info']['nparttot']:.2e}\n{plot_title_suffix}",
        )
        fig.savefig(outfilename)
        
    #plt.close(fig)
    return fig, ax, outfilename


# In[9]:


# main process


if __name__ == '__main__':
    
    mpdf = mupl.MyPhantomDataFrames()
    
    for job_profile in JOB_PROFILES:
        job_name = job_profile['job_name']
        ieos = job_profile['ieos']
        X    = job_profile['X']

        outfilenames_dict = {}
        for plane_axes in plane_axes_list:
            outfilenames_dict[plane_axes] = []
        
        for file_index in job_profile['file_indexes']:
            # read data
            mpdf.read(job_name, file_index, reset_xyz_by='CoM', verbose=verbose)
            mpdf.eos = job_profile['EoS']
            if 'Tdust' in mpdf.data['gas'].columns:
                mpdf.data['gas']['T'] = mpdf.data['gas']['Tdust']
            elif 'temperature' in mpdf.data['gas'].columns:
                mpdf.data['gas']['T'] = mpdf.data['gas']['temperature']
            mpdf.calc_sdf_params(
                calc_params=['kappa',], #'R1',
                calc_params_params={'ieos': ieos, 'X':X, 'overwrite':False, 'kappa_translate_from_cgs_units':True},
                verbose=verbose,
            )
        
            for plane_axes in plane_axes_list:
                outfilename_noext = f"{mpdf.get_filename()}__photosphere-xsec__R1-{plane_axes}"
                if use_saved_jsons:
                    with open(f"{outfilename_noext}.json", 'r') as f:
                        photospheres = mupl.json_load(f)
                else:
                    photospheres = get_mpdf_photosphere_xsec(
                        mpdf=mpdf,
                        ray_no=ray_no,
                        plane_axes=plane_axes,
                        photosphere_tau = PHOTOSPHERE_TAU,
                        nprocesses = NPROCESSES,
                        verbose=0,
                    )
                    with open(f"{outfilename_noext}.json", 'w') as f:
                        mupl.json_dump(photospheres, f, metadata)
                    
    
                #fig, ax, outfilename = plot_mpdf_photosphere_xsec(
                #    photospheres= photospheres,
                #    job_name    = job_name,
                #    job_index   = file_index,
                #    plot_title_suffix=job_profile['plot_title_suffix'],
                #    box_lim     = box_lim,
                #    unitsOut    = unitsOut,
                #    outfilename_noext = outfilename_noext,
                #)
                #plt.close(fig)
                #outfilenames_dict[plane_axes].append(outfilename)
                #print(f"Done: {outfilename};")
    
        # Make movie
    
        for plane_axes in plane_axes_list:
            outfilename_vectxt = f"R1-{plane_axes}"
            #with ImageSequenceClip(outfilenames_dict[plane_axes], fps=fps) as vid:
            #    moviefilename = f"{job_profile['job_name']}__photosphere-xsec__{outfilename_vectxt}__movie.mp4"
            #    vid.write_videofile(moviefilename)

    print("\n\n\n*** All Done. ***\n\n\n")


# In[ ]:




