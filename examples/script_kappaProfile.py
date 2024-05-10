#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Scripts for analyzing of phantom outputs.

This script analyze the dump files about the opacity based on density and temperature.

"""


# # Main

# ## Imports & Settings

# In[2]:


import numpy as np
from astropy import units
import matplotlib.pyplot as plt
import matplotlib as mpl
from moviepy.editor import ImageSequenceClip
from os import path


# In[3]:


# import my modules listed in ./main/

from main import clmuphantomlib as mupl
#from main.clmuphantomlib.readwrite import json_load
from main.clmuphantomlib.log import is_verbose, say
from main.clmuphantomlib.settings   import DEFAULT_SETTINGS as settings
from main.clmuphantomlib.units_util import get_val_in_unit #set_as_quantity, get_units_field_name, get_units_cgs
from main.clmuphantomlib.eos_mesa   import EoS_MESA_opacity
from multiprocessing import cpu_count, Pool #Process, Queue
NPROCESSES = 1 if cpu_count() is None else max(cpu_count(), 1)


# In[4]:


# settings
#
#   imported from script_input.py file


from script_kappaProfile__input import verbose, fps, unitsOut, JOB_PROFILES_DICT

unitsOutTxt = {  key  : unitsOut[key].to_string('latex_inline') for key in unitsOut.keys() }


plt.rcParams.update({'font.size': 20})
if __name__ == '__main__' and is_verbose(verbose, 'note'):
    # remember to check if name is '__main__' if you wanna say anything
    #    so when you do multiprocessing the program doesn't freak out
    say('note', "script", verbose, f"Will use {NPROCESSES} processes for parallelization")


# In[5]:


# functions

# plot_kappaProfile(job_name, file_index, eos_opacity, xlims, ylim, verbose)
def plot_kappaProfile(
    job_name: str, file_index: int, eos_opacity: EoS_MESA_opacity,
    xlims: dict,
    ylim: tuple,
    verbose: int,
) -> str:
    """Plot kappa Profile of a dump.

    Warning: assume a hard-coded constant kappa_gas as 2e-4 cm2/g

    Returns outfilename
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.subplots_adjust(wspace=0.0)
    mpdf = mupl.MyPhantomDataFrames().read(job_name, file_index, verbose=verbose) # reset_xyz_by_CoM=True, 
    #jobfilename = mupl.get_filename_phantom_dumps(job_name, file_index)
    jobfilename = mpdf.get_filename()
    # get temperature column label (one of the elem in the set below)
    temp_key = {'T', 'temperature', 'Tdust'}.intersection(mpdf.data['gas'].keys()).pop()
    mpdf.data['gas']['T'    ] = mpdf.data['gas'][temp_key]
    mpdf.data['gas']['kappa'] = get_val_in_unit(mpdf.data['gas']['kappa'], units.cm**2/units.g, mpdf.units['opacity'])

    y_orig = mpdf.get_val('kappa').to(unitsOut['opacity'])
    y_mesa = eos_opacity.get_kappa(mpdf.get_val('rho'), mpdf.get_val('T'), do_extrap=False).to(unitsOut['opacity'])
    y_extrap_indexes = np.where(~np.isfinite(y_mesa))[0]
    y_mesa_extrap = eos_opacity.get_kappa(
        mpdf.get_val('rho')[y_extrap_indexes], mpdf.get_val('T')[y_extrap_indexes], do_extrap=True).cgs

    # setting the switch between mesa opacity from phantom and nucleation opacity from luis
    # mesa opacity table from phantom uses Ferguson-2005-1 for opacity calc at low T (T < 1e4 K),
    #    which includes opacity from grains forming.
    #    We don't want that becaues we have our own carbon dust nucleation opacity.
    #    according to Ferguson-2005-1 fig9, grains dominates at T < 1450K (for logR=-3, X=0.7, Z=0.02)
    T_0, T_delta = 1450 * units.K, 50 * units.K
    x = mpdf.get_val('T').to(unitsOut['temp'], equivalencies=units.equivalencies.temperature())
    kappa_gas = (2e-4*(units.cm**2/units.g)).to(unitsOut['opacity'])
    y_comb = np.where(x < T_0, y_orig, y_orig - kappa_gas + y_mesa)

    
    # kappa vs temp
    ax = axes[0]
    x = mpdf.get_val('T').to(unitsOut['temp'], equivalencies=units.equivalencies.temperature())
    x_extrap = x[y_extrap_indexes]
    ax.loglog(x, y_orig, '.')
    ax.loglog(x, y_mesa, '.')
    ax.loglog(x_extrap, y_mesa_extrap, '.')
    ax.loglog(x, y_comb, '.', label='Blended')
    ax.set_xlim(xlims['T'])
    ax.set_ylim(ylim)
    ax.set_xlabel(f"$T$ / {x.unit.to_string('latex_inline')}")
    ax.set_ylabel(f"$\\kappa$ / {y_orig.unit.to_string('latex_inline')}")
    ax.text(
        0.02, 0.98,
        f"Time = {mpdf.get_time(unitsOut['time']):.1f}",
        color = "black", ha = 'left', va = 'top',
        transform=ax.transAxes,
    )

    # kappa vs rho
    ax = axes[1]
    x = mpdf.get_val('rho').to(unitsOut['density'])
    x_extrap = x[y_extrap_indexes]
    ax.loglog(x, y_orig, '.', label='Nucleation')
    ax.loglog(x, y_mesa, '.', label='MESA')
    ax.loglog(x_extrap, y_mesa_extrap, '.', label='MESA extrap')
    ax.loglog(x, y_comb, '.', label='Blended')
    ax.set_xlim(xlims['rho'])
    ax.set_ylim(ylim)
    ax.set_xlabel(f"$\\rho$ / {x.unit.to_string('latex_inline')}")
    ax.legend(loc='lower right')


    fig.suptitle(
        f"Opacity of all particles in the dump (different calculation method)\n"
        #f"resolution = {mpdf.params['nparttot']:.2e}\n"
        f"{job_profile['plot_title_suffix']}"
    )

    outfilename = f"{jobfilename}__kappaProfile.png"
    fig.savefig(outfilename)
    plt.close(fig)
    del mpdf

    return outfilename


# In[63]:


do_debug = False


# In[75]:


if __name__ == '__main__' and do_debug:
    ylim = (1e-6, 1e4)
    xlims= {
        'T'  : (  5.,  2e6),
        'rho': (2e-20, 1e-3),
    }
    
    key = '2md'
    file_index  = 17600
    job_profile = JOB_PROFILES_DICT[key]
    job_name    = job_profile['job_name']
    params      = job_profile['params']
    eos_opacity = EoS_MESA_opacity(params, settings)
    
    outfilename = plot_kappaProfile(job_name, file_index, eos_opacity, xlims, ylim, verbose)
    print(outfilename)


# In[127]:


if __name__ == '__main__' and do_debug:
    params = JOB_PROFILES_DICT['2md']['params']
    eos_opacity = EoS_MESA_opacity(params, settings)
    Ts = (10**np.arange(3., 3.3, 0.005))*units.K
    fig, ax = plt.subplots(figsize=(10, 8))
    for log10_rho in range(-16, -11, 1):
        rho = (10**log10_rho)*(units.g/units.cm**3)
        kappas = eos_opacity.get_kappa(rho=rho, T=Ts)
        ax.semilogy(Ts.cgs, kappas.cgs, '-', label=f"$\\log_{{10}} \\rho = {log10_rho}$")

    ax.legend()
    ax.set_title("MESA opacity table $\\kappa$ vs temperature $T$")
    ax.set_ylabel(f"$\\kappa$ / {kappas.cgs.unit.to_string('latex_inline')}")
    ax.set_xlabel(f"$T$ / $K$")
    fig.savefig(f"main/test_mesa-opacity_kappa-T-rho.png")


# In[ ]:


# plotting kappa vs Temp and kappa vs rho
if __name__ == '__main__':

    ylim = (1e-6, 1e4)
    xlims= {
        'T'  : (  5.,  2e6),
        'rho': (2e-20, 1e-3),
    }
    
    #mpdf = mupl.MyPhantomDataFrames()
    for key in ['2md', '4md']:
        job_profile = JOB_PROFILES_DICT[key]
        job_name    = job_profile['job_name']
        params      = job_profile['params']
        eos_opacity = EoS_MESA_opacity(params, settings)

        
        if NPROCESSES <= 1:

            # single process
            if __name__ == '__main__' and is_verbose(verbose, 'note'):
                say('note', "script_kappaProfile", verbose, f"Using single process.")
            
            outfilenames = []
            for file_index in job_profile['file_indexes']:
                outfilename = plot_kappaProfile(job_name, file_index, eos_opacity, xlims, ylim, verbose)
                outfilenames.append(outfilename)
        else:

            # multi-process
            if __name__ == '__main__' and is_verbose(verbose, 'note'):
                say('note', "script_kappaProfile", verbose, f"Using {NPROCESSES} processes.")
                
            args = [(job_name, file_index, eos_opacity, xlims, ylim, 0) for file_index in job_profile['file_indexes']]
            with Pool(processes=NPROCESSES) as pool:
                outfilenames = pool.starmap(plot_kappaProfile, args)


        # define job_folder_prefix
        for i in range(len(job_name)-1, -1, -1):
            if job_name[i] == path.sep:
                job_folder_prefix = job_name[:i]
                break
            else:
                job_folder_prefix = job_name
        with ImageSequenceClip(outfilenames, fps=fps) as vid:
            moviefilename = f"{job_folder_prefix}__kappaProfile__movie.mp4"
            vid.write_videofile(moviefilename)

    print("\n\n\n*** All Done. ***\n\n\n")


# if __name__ == '__main__':
#     
#     plt.close(fig)
#     
#     fig, ax = plt.subplots(figsize=(8, 8))
#     x = mpdf.get_val('T').to(unitsOut['temp'], equivalencies=units.equivalencies.temperature())
#     y = mpdf.get_val('rho').to(unitsOut['density'])
#     ax.loglog(x, y, '.')
#     ax.text(
#         0.95, 0.5,
#         f"Time = {mpdf.get_time(unitsOut['time']):.1f}",
#         color = "black", ha = 'right', va = 'top',
#         transform=ax.transAxes,
#     )
#     ax.set_xlim(xlims['T'])
#     ax.set_ylim(xlims['rho'])
# 
# 
# 
# 

# In[ ]:




