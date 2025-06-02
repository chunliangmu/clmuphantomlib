#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Deprecated.
Old main library- Will be disassembled into different files in the future *sometime*.

Owner: Chunliang Mu

--------
original description:
--------

Chunliang Mu's phantom data analysis library

Assuming temperature unit being K. Reads & handles other units from phantom data dumps.
"""


# In[2]:


## Jupyter magic widget for interactive plots
#%matplotlib widget



# my libs
from .log import error, warn, note, debug_info
from .geometry import *
from .units_util import DEFAULT_UNITS, set_as_quantity, set_as_quantity_temperature, get_units_field_name
from .sph_interp import get_sph_interp
from .mpdf import *


# In[3]:


import math
#from warnings import warn
import numpy as np
from numpy import pi
from scipy import optimize
from astropy import units
from astropy import constants as const
import sarracen
import matplotlib.pyplot as plt
import matplotlib as mpl
# importing moviepy libraries
try:
    from moviepy.editor import ImageSequenceClip
except RuntimeError as e:
    print(e)
    ImageSequenceClip = None



# # Defines

# In[4]:


# CONSTANTS

# hydrogen atom mass
CONST_M_H = const.m_p.cgs
# radiation constant from SPLASH globaldata.f90
CONST_RADCONST = (4*const.sigma_sb/const.c).cgs
# kb_on_mh
CONST_KB_ON_MH = (const.k_B / CONST_M_H).cgs



# # Functions

# In[6]:





# In[7]:


# EoS

def eos_adiabatic(gamma, rho, u):
    """Adiabatic EOS. Return Pressure P"""
    return (gamma - 1) * rho * u


# In[8]:





# In[9]:


# helper tools


# Geometry



# Others


def get_supersonic_frac(sdf):
    """Return the fraction of supersonic particles in sdf."""
    return np.count_nonzero(sdf['mach'] > 1) / len(sdf['mach'])


# In[10]:


def get_sph_close_pts_indices(
    loc : np.ndarray,
    sdf : sarracen.SarracenDataFrame = None,
    pts : np.ndarray = None,
    hs  : np.ndarray = None,
    kernel: sarracen.kernels.BaseKernel = None,
):
    """Return indices of the sph particles close to loc.
    
    Need to supply either sdf or all of the (pts, hs, kernels)
    If any of the (pts, hs, kernels) is None, will get them from sdf
    If all (sdf, pts, hs, kernels) is given, will ignore sdf.
    """
    
    if sdf is not None:
        if pts    is None:    pts = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array
        if hs     is None:    hs  = np.array(sdf['h'])    # npart-shaped array
        if kernel is None:    kernel = sdf.kernel
    else:
        if (pts is None) or (hs is None) or (kernel is None):
            raise TypeError("Need to supply either sdf or all of the (pts, hs, kernels) as parameters")
    
    kernel_radius = kernel.get_radius()
    r2_range = (kernel_radius * hs)**2
    r2s = np.sum((pts - loc)**2, axis=-1)
    indices = r2s < r2_range
    return indices



# In[12]:


# get temperature
# ref: SPLASH lightcurve_utils.f90

def get_temp_from_u(
    rho, u, mu, ieos:int,
    rho_unit=None, u_unit=None,
    verbose: int = 0,
) -> np.ndarray:
    """Calculate temperature from internal energy, assuming it being a mix of gas & radiation pressure (only if ieos!=2).
    
    I.e. it solves this equation:
        rho*u = 3/2*rho*kb*T/(mu*mH) + a*T^4
        
    If ieos ==2, it calc T by:
        rho*u = 3/2*rho*kb*T/(mu*mH)
    (i.e. ignoreing rad pressure)
    
    It is same function from lightcurve_utils.f90 in the source code of SPLASH (2023-05-17)
    
    rho & u must be all positive element-wise.
    
    Parameters
    ----------
    rho: float, 1D array, or astropy.units.quantity.Quantity
        Density  (in cgs units if not astropy.units.quantity.Quantity).
        Not needed if ieos==2 (you can set it as None)
    u: float, 1D array, or astropy.units.quantity.Quantity
        Internal energy  (in cgs units if not astropy.units.quantity.Quantity).
    mu: float or None or "adapt"
        mean molecular weight.
        if None or "adapt", will assume to be 0.6 if temp > 10000K and 2.38 if temp < 6000K.
            If inbetween or otherewise, will use the average of both.
    ieos: int
        Equation of state (see Phantom doc: https://phantomsph.readthedocs.io/en/latest/eos.html).
        Use 2 for ideal gas EoS (adiabatic), which will calc T assuming no rad pressure
        Use 10/12 for MESA / (ideal+rad) EoS, which will calc T assuming ideal gas + rad for u
        Use anything else will still calc T assuming ideal gas + rad for u, but it will also give a warning.
    rho_unit, u_unit:
        if None and rho, u are not astropy quantities, *** WILL ASSUME CGS UNITS ***.
    
    Returns
    -------
    temp: np.ndarray
        Temperature in K
    """
    
    # Constants & inputs in cgs units
    
    if mu is None or (issubclass(type(mu), str) and mu == 'adapt'):
        temp_ionized    = get_temp_from_u(rho, u, 0.6     , ieos, rho_unit=rho_unit, u_unit=u_unit, verbose=verbose,)
        temp_recombined = get_temp_from_u(rho, u, 2.380981, ieos, rho_unit=rho_unit, u_unit=u_unit, verbose=verbose,)
        temp = np.where(
            np.logical_and(temp_ionized > 10000., temp_recombined > 6000.),
            temp_ionized,
            np.where(
                np.logical_and(temp_ionized < 10000., temp_recombined < 6000.),
                temp_recombined,
                np.nan
            )
        )
        note("get_temp_from_u()", verbose,
             f"Using average temp for {np.count_nonzero(np.isnan(temp))} out of {len(temp)} particles.\n" + \
             f"                   {np.count_nonzero(np.logical_and(temp_ionized > 10000., temp_recombined < 6000.))}" + \
             "out of which are contradicting each other (T > 10000K if mu=0.6, T < 6000K if mu=2.38)"
        )
        temp = np.where(np.isnan(temp), (temp_ionized + temp_recombined) / 2, temp)
        return temp
    
    
    
    kB_on_mH = CONST_KB_ON_MH.cgs.value
    radconst = CONST_RADCONST.cgs.value
    tol = 1.e-8
    maxiter=500
        
        
    
    if issubclass(type(rho), units.quantity.Quantity):
        rho = rho.cgs.value
    elif rho_unit is not None:
        # using rho = rho * instead of rho *= to avoid overwriting original data
        rho = rho * (1. * rho_unit).cgs.value
        
    if issubclass(type(u), units.quantity.Quantity):
        u = u.cgs.value
    elif u_unit is not None:
        # using u = u * instead of u *= to avoid overwriting original data
        u = u * (1. * u_unit).cgs.value
        
        
    # calc T
    if ieos == 2:
        # ideal gas only
        temp = u * mu / (1.5 * kB_on_mH)
        return temp
    elif ieos != 10 and ieos != 12:
        # give warning
        print(f"*** WARNING: Unrecognized EoS {ieos=}. Calc T from u assuming ideal gas + rad pressure.  ***")
        
    
    # initial guess of temperature
    
    temp = np.minimum(
        u * mu / (1.5 * kB_on_mH),
        (rho * u / radconst)**0.25,
    )
    
    rho_i = rho
    u_i = u
    def func(T_i, rho_i, u_i):
        """Energy density function for solving T."""
        return 1.5 * rho_i * kB_on_mH / mu * T_i + radconst * T_i**4 - rho_i * u_i
    
    def func_prime(T_i, rho_i, u_i):
        """The derivative of T of func."""
        return 1.5 * rho_i * kB_on_mH / mu + 4 * radconst * T_i**3

    
    if temp.ndim == 0:
        temp = optimize.root_scalar(
            func, x0=temp, args=(rho, u), method='newton', fprime=func_prime, maxiter=maxiter, rtol=tol).root
    elif temp.ndim == 1:
        # Newton-Raphson as implemented in SPLASH
        #dt=np.inf
        for i in range(maxiter):
            temp_on_5 = 0.2 * temp
            dt = -func(temp, rho, u) / func_prime(temp, rho, u)
            if np.all(abs(dt) < tol * temp): break
            dt[dt >  temp_on_5] = temp_on_5[dt >  temp_on_5]
            dt[dt < -temp_on_5] = temp_on_5[dt < -temp_on_5]
            temp = temp + dt

        else:
            warn('get_temp_from_u()', verbose, f"temperature not converging- max rtol = {np.max(abs(dt/temp))}")
        note('get_temp_from_u()', verbose, f"max rtol = {np.max(abs(dt/temp))} after {i} iters.")
    else:
        raise TypeError(f"Unexpected dimension of input data rho & u:\n{rho=}\n{u=}")
    return temp






# #### Equations from External sources- remember to cite!

# In[14]:


def get_roche_lobes_radius_eggleton1983(q):
    """
    Return the approximated Roche Lobes radii using Eq2 from Eggleton-1983-1, assuming orbital separation is 1.
    
    *** Remember to cite! (Eggleton-1983-1)
    
    Source paper suggests that the error should be within 1%.
    
    Parameters
    ----------
    q: float
        mass ratio.
    """
    r_L = ( 0.49 * q**(2./3.) ) / ( 0.6 * q**(2./3.) + np.log(1 + q**(1./3.)) )
    return r_L


# In[15]:


def get_opacity_dust_bowen1988(Teq, kmax, Tcond, delta, kappa_gas=0.*(units.cm**2/units.g)):
    """
    Return the Bowen opacity from Bowen-1988-1 eq5.
    Returns dust opacity only by default (as Default kappa_gas is zero).
    Set kappa_gas to get total opacity (dust+gas).
    
    *** Remember to cite! (Bowen-1988-1)
    
    Parameters
    ----------
    Teq:  astropy.units.quantity.Quantity
        dust equilibrium temperature (K)
        (In Siess-2022-1 we use gas temperature to approximate this- see sec5.2)
        
    kmax: astropy.units.quantity.Quantity
        maximum dust opacity (cm²/g)
        
    Tcond: astropy.units.quantity.Quantity
        dust condensation temperature (K)
        
    delta: astropy.units.quantity.Quantity
        condensation temperature range (K)
        
    kappa_gas: astropy.units.quantity.Quantity
        gas opacity (cm2/g). Default is 0. cm2/g.
    
    Returns
    -------
    kappa: astropy.units.quantity.Quantity
        Bowen parameterization dust opacity (cm2/g)
    """
    Teq = set_as_quantity_temperature(Teq, copy=False)
    Tcond = set_as_quantity_temperature(Tcond, copy=False)
    delta = set_as_quantity_temperature(delta, copy=False)
    kmax = set_as_quantity(kmax, unit=(units.cm**2/units.g), copy=False)
    kappa_gas = set_as_quantity(kappa_gas, unit=(units.cm**2/units.g), copy=False)
    
    kappaDust = kmax / ( 1 + np.exp((Teq - Tcond)/delta) )
    kappa = kappaDust + kappa_gas
    return kappa


# # Classes

# In[16]:





# ## Functions for MyPhantomDataFrames

# In[17]:


# Rendering

def plot_mpdf_movies_xsec(
    job_name, plot_title_suffix,
    file_range : (int, int, int) = None,
    file_indexes : list = None,
    figsize=(8, 8),
    xyz_axis='xyz', xsec=0.0, rendered = 'rho',
    mpdf_read_kwargs : dict = {},
    calc_rendered_func=None, calc_rendered_func_params={},
    xlim=(-10000, 10000), ylim=(-10000, 10000),
    norm=mpl.colors.LogNorm(vmin=None, vmax=None, clip=True),
    unit_time=units.year, unit_time_print_utime_too=False,
    fps=20,
    verbose=1,
    **kwargs
):
    """
    Plot Movies of rendered at xsec=xsec.
    Saves individual plots & a movie.
    Returns movie file name.

    Parameters
    ----------
    job_name: str
        Name of the job. e.g. 'binary'
        
    file_range: (int, int, int)
        Range of the number. Overrides file_indexes. Give either file_range or file_indexes.
        e.g. file_range=(0, 100+1, 1)
            combined with job_name='binary' will plot everything from binary_00000 to binary_00100.
            
    plot_title_suffix: str
        Text to add to the end of the plot title. Use '' if you don't need to add anything.
        
    figsize: (int, int)
        figure size.

    xyz_axis: str (len 3)
        set x, y axis of the plot & the cross-sec direction z.
        e.g. 'yzx' will plot a slice of x=y, y=z plot at xsec of x=xsec.
        
    xsec: float
        cross-section location.
        Setting the plot to plot a slice at xyz_axis[2]=xsec.
    
    rendered: str
        the variable to be rendered in the plot.
        
    mpdf_read_kwargs: dict
        dict of Keyword arguments to pass to mpdf.read. 
        
    calc_rendered_func: function
        function used to calculate sdf[rendered] after reading each phantom data file.
        it needs to return sdf[rendered].
        if None, do nothing.
        
    calc_rendered_func_params: dict
        params passed to calc_rendered_func after the first param.
        The first param passed to calc_rendered_func is always sdf.
        
    xlim, ylim: (float, float) or astropy.units.quantity.Quantity (len==2)
        lim of x & y axis of the plot.
        If quantity, the unit of x/y axis will be set to its unit. (*** To Be Implemented***)
        
    norm: mpl.colors.Normalize
        Normalization for matplotlib (mpl) color bar.
        Use either mpl.colors.Normalize or mpl.colors.LogNorm.
        
    unit_time: astropy.units.Unit
        Unit for the time stamp.

    unit_time_print_utime_too: bool
        If true, will print an additional line of time stamp in code unit time.
        
    fps: int
        fps for the movie.
        
    verbose: int
        how verbose should this func be. 0~2.
        
    **kwargs: other keyword arguments
        Keyword arguments to pass to mpdf.plot_render.
        

    Returns
    -------
    moviefilename: str
        filename of the movie generated
    """
    
    # init
    
    print_progress = False
    if verbose >= 1:
        print_progress = True
        print("Working on:")
    if file_range is not None:
        file_indexes = range(*file_range)

    mpdf = MyPhantomDataFrames()
    outfilenames = []

    # set x,y,z axis to be 'x', 'y', or 'z'
    x_axis = xyz_axis[0]
    y_axis = xyz_axis[1]
    z_axis = xyz_axis[2]
        
    # running
    j = 0
    for i in file_indexes:
        
        fig, ax = plt.subplots(figsize=figsize)

        mpdf.read(job_name, i, verbose=verbose, **mpdf_read_kwargs)
        sdf = mpdf.data['gas']
        
        # calculate rendered par
        if calc_rendered_func is not None:
            sdf[rendered] = calc_rendered_func(sdf, **calc_rendered_func_params)
            
        outfilename = mpdf.plot_render(
            fig=fig, ax=ax, savefilename=None, plot_title_suffix=plot_title_suffix,
            xyz_axis=xyz_axis, xsec=xsec, rendered = rendered,
            xlim=xlim, ylim=ylim, norm=norm,
            unit_time=unit_time, unit_time_print_utime_too=unit_time_print_utime_too,
            **kwargs,
        )
        outfilenames.append(outfilename)
        
        plt.close(fig)
        
        if print_progress and j % 10 == 0:
            print(f'{i}', end=' ')
        j += 1

    with ImageSequenceClip(outfilenames, fps=fps) as vid:
        moviefilename = f'{job_name}__{x_axis}-{y_axis}-{rendered}__{z_axis}={xsec:.1f}__movie.mp4'
        vid.write_videofile(moviefilename)
        
    return moviefilename


# In[18]:


def plot_mpdf_movies_xsec_opacity_bowendust(kmax, Tcond, delta, kappa_gas, **kwargs):
    """
    Plot Movies of bowen dust opacity at xsec=xsec.
    See plot_mpdf_movies_xsec for more info about params.
    """    
    def get_opacity_bowendust_sdf_bowen1988(sdf, kmax, Tcond, delta, kappa_gas):
        if 'Tdust' in sdf.keys():
            Tdust = sdf['Tdust']
        else:
            print("*   Warning: Tdust not found in sdf.keys().")
            if 'temperature' in sdf.keys():
                Tdust = sdf['temperature']
            elif 'T' in sdf.keys():
                Tdust = sdf['T']
            else:
                raise Exception(f"No Temperature-like columns found in sdf.keys():\n{sdf.keys()}")
        return get_opacity_dust_bowen1988(Tdust, kmax, Tcond, delta, kappa_gas=kappa_gas).to_value(units.cm**2/units.g)
    calc_rendered_func = get_opacity_bowendust_sdf_bowen1988
    
    calc_rendered_func_params={'kmax': kmax, 'Tcond': Tcond, 'delta': delta, 'kappa_gas': kappa_gas}
    
    result = plot_mpdf_movies_xsec(
        calc_rendered_func = calc_rendered_func,
        calc_rendered_func_params=calc_rendered_func_params,
        **kwargs,
    )
    return result









### TESTING ###




# In[39]:


# Making a movie of photosphere temperature profile
# *   Warning: Assuming Bowen dust opacity - see BOWEN_DUST_PARS

def plot_mpdf_movies_single_ray_trace_profiles__TMP_FUNC_IGNORE_ME(
    job_name, file_range, plot_title_suffix,
    ray_dir_vec=np.array([0., 0., 100.]),
    use_idealgas_temperature=True,
    mu=2.380981,    # mean molecular weight, used if use_idealgas_temperature
    photosphere_tau=1.,
    ieos=10,
    X=0.686,
    fps=1,
):
    """Plot Movies of single ray tracing (from primary star) & temperature & density profiles.
    
    *** EXPERIMENTAL - USE WITH CAUTION ***
    
    """
    # settings
    #use_idealgas_temperature = True
    #mu = 2.380981    # mean molecular weight, used if use_idealgas_temperature
    xlim = (10., 100000.)
    ylim_tau = (1e-1, 1e5)
    ylim_T = (1e1, 5e6)
    ylim_rho = (1e-20, 1e-5)
    sample_size_T = 1000
    unit_dist_text = units.solRad.to_string('latex')
    unit_temp_text = units.K.to_string('latex')
    #unit_rho_text  = mpdf.units['density'].cgs.to_string('latex')
    unit_rho_text  = (units.solMass/units.solRad**3).to_string('latex')
    
    outfilenames = []
    times = []
    photospheres = []
    lum_effs = []
    mpdf = MyPhantomDataFrames()
    # read files
    for i in range(*FILE_RANGE):
        if use_idealgas_temperature: calc_params=[]
        else: calc_params=['T', 'kappa']
        mpdf.read(
            job_name, i,
            calc_params=calc_params,
            calc_params_params={'ieos': ieos, 'X':X, 'overwrite':False, 'kappa_translate_from_cgs_units':True},
        )
        if use_idealgas_temperature:
            mpdf.data['gas']['T'] = \
            (2./3. * mu * (set_as_quantity(mpdf.data['gas']['u'], unit=mpdf.units['specificEnergy']) \
                          ) / CONST_KB_ON_MH).cgs.value
        filename = mpdf.get_filename()
        print(f"\n\nWorking on: {filename}\n")

        star1_x = mpdf.data['sink'].x[0]
        star1_y = mpdf.data['sink'].y[0]
        star1_z = mpdf.data['sink'].z[0]
        ray = np.array([
            [star1_x, star1_y, star1_z],
            [star1_x, star1_y, star1_z],
        ])
        ray[1] += ray_dir_vec
        ray_unit_vec = ray[1, :] - ray[0, :]
        ray_unit_vec = ray_unit_vec / np.sum(ray_unit_vec**2)**0.5


        # optimization- first select only the particles affecting the ray
        #  because interpolation of m points with N particles scales with O(N*m),
        #  reducing N can speed up calc significantly
        sdf = mpdf.data['gas']
        kernel_radius = sdf.kernel.get_radius()
        hs = np.array(sdf['h'])
        pts = np.array(sdf[['x', 'y', 'z']])    # (npart, 3)-shaped array
        pts_on_ray = get_closest_pt_on_line(pts, ray)
        sdf_selected_indices = (np.sum((pts - pts_on_ray)**2, axis=-1) <= (kernel_radius * hs)**2)
        print(f"{np.count_nonzero(sdf_selected_indices)} particles are close enough to the ray to have effects.")    # debug
        sdf = sdf.iloc[sdf_selected_indices]
    
        # get optical depth
        #sdf['kappa'] = (0.35 * (units.cm**2/units.g)).to_value(mpdf.units['opacity'])
        #sdf['kappa'] = get_opacity_dust_bowen1988(Teq=sdf['T'], **BOWEN_DUST_PARS).to_value(mpdf.units['opacity'])
        #print(f"\n{sdf = }\n")
        print(f"{ray = }\n")
        pts_on_ray, dtaus, pts_order = get_optical_depth_by_ray_tracing_3D(sdf=sdf, ray = ray)
        pts_order_nonzero = np.where(dtaus[pts_order])[0]
        print(f"{pts_order_nonzero.size = }\n")
        pts_on_ray_ordered = pts_on_ray[pts_order]
        dist_to_ray0_ordered = np.sum((pts_on_ray_ordered - ray[0]) * ray_unit_vec, axis=-1)
        taus_ordered = np.cumsum(dtaus[pts_order])


        # get photosphere
        photosphere_loc_index = np.searchsorted(taus_ordered, photosphere_tau) - 1
        photosphere_found = photosphere_loc_index <= len(sdf) - 2
        if photosphere_found:
            if photosphere_loc_index == -1:
                # if first particle blocks everything: estimate it to be where that particle is
                photosphere_loc_index = 0
                photosphere_loc = pts_on_ray_ordered[0]
            else:
                # intepolate to find loc
                photosphere_taus = taus_ordered[photosphere_loc_index : photosphere_loc_index+2]
                photosphere_dtau = photosphere_taus[1] - photosphere_taus[0]
                photosphere_dtau0_frac = (photosphere_taus[1] - photosphere_tau) / photosphere_dtau
                photosphere_loc = \
                    pts_on_ray_ordered[photosphere_loc_index] * photosphere_dtau0_frac + \
                    pts_on_ray_ordered[photosphere_loc_index+1] * (1 - photosphere_dtau0_frac)
            photosphere_dist_to_ray0 = np.sum((photosphere_loc - ray[0]) * ray_unit_vec, axis=-1)
        else:
            print(f"*    Warning: Photosphere not found ({max(taus_ordered)=})")


        # calc temperature profiles
        dist_to_ray0_samples = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), sample_size_T)
        pts_on_ray_samples = ray[0] + dist_to_ray0_samples.reshape((-1, 1)) * ray_unit_vec
        T_samples = get_sph_interp(sdf, 'T', pts_on_ray_samples)
        rho_samples = get_sph_interp(sdf, 'rho', pts_on_ray_samples)
        photosphere = {}
        if photosphere_found:
            # get lum & other photosphere quantities
            photosphere['loc'] = photosphere_loc
            photosphere['R1'] = photosphere_dist_to_ray0
            for col in ['T', 'rho', 'h']:
                photosphere[col] = get_sph_interp(sdf, col, photosphere_loc)
            area_eff_quantity = 4 * pi * (photosphere['R1'] * mpdf.units['dist'])**2
            lum_eff_quantity = (const.sigma_sb * (photosphere['T'] * units.K)**4 * area_eff_quantity).to(units.solLum)

            # get lum & T errors
            dist_to_ray0_samples_ph = np.logspace(
                np.log10(photosphere['R1'] - photosphere['h']), np.log10(photosphere['R1'] + photosphere['h']),
                sample_size_T,
            )
            pts_on_ray_samples_ph = ray[0] + dist_to_ray0_samples_ph.reshape((-1, 1)) * ray_unit_vec
            T_samples_ph = get_sph_interp(sdf, 'T', pts_on_ray_samples_ph)
            area_eff_quantities_ph = 4 * pi * (dist_to_ray0_samples_ph * mpdf.units['dist'])**2
            lum_eff_quantities_ph = (const.sigma_sb * (T_samples_ph * units.K)**4 * area_eff_quantities_ph).to(units.solLum)
            photosphere['T_m'] = min(T_samples_ph) - photosphere['T']
            photosphere['T_p'] = max(T_samples_ph) - photosphere['T']
            lum_eff_quantity_m = min(lum_eff_quantities_ph) - lum_eff_quantity
            lum_eff_quantity_p = max(lum_eff_quantities_ph) - lum_eff_quantity

            
        times.append(mpdf.get_time().to_value(units.year))
        photospheres.append(photosphere)
        lum_effs.append([
            lum_eff_quantity.to_value(units.solLum),
            (lum_eff_quantity + lum_eff_quantity_m).to_value(units.solLum),
            (lum_eff_quantity + lum_eff_quantity_p).to_value(units.solLum),
        ])

        
        # plotting

        # plot1 - optical depth
        ylim = ylim_tau
        fig, axes = plt.subplots(3, figsize=(10, 12), sharex=True)
        #fig.tight_layout()
        fig.subplots_adjust(hspace=0.0)
        ax = axes[0]
        ax.semilogy(dist_to_ray0_ordered, taus_ordered)
        ax.set_ylim(ylim)
        ax.set_ylabel("Optical depth $\\tau$")
        ax.tick_params(which='both', labelbottom=False)
        ax.axhline(1., color='grey', linestyle='dashed')
        if photosphere_found:
            ax.axvline(photosphere_dist_to_ray0, color='green', linestyle='dashed')
            ax.fill_betweenx(ylim, photosphere['R1'] + photosphere['h'], photosphere['R1'] - photosphere['h'], alpha=0.1)
            ax.text(photosphere['R1'] + photosphere['h'], ylim[1],
                    #f" Photosphere distance to primary star:\n" + \
                    f" $R_{{1, \\rm ph}} = {photosphere['R1']:.1f} $ {unit_dist_text}", va='top')
        ax.text(
            0.98, 0.87, f"Time = {mpdf.get_time(unit=units.year):.2f}\n" + \
            #"\nSmoothing Length\nat the photosphere:\n" + \
            f"\n$h_{{\\rm ph}} = {photosphere['h']:.1f} $ {unit_dist_text}\n" + \
            #"\nEffective Luminosity\nfrom $R_{{1, \\rm ph}}$ and $T_{{\\rm ph}}$:\n" + \
            f"$L_{{\\rm eff}}$ = {lum_eff_quantity.value:.1e} {lum_eff_quantity.unit.to_string('latex')}\n" + \
            f"$\\Delta L_{{\\rm eff}}$ = $^{{{lum_eff_quantity_p.value:+.1e}}}_{{{lum_eff_quantity_m.value:+.1e}}}$" + \
            f"{lum_eff_quantity.unit.to_string('latex')}\n",
            color = "black", ha = 'right', va = 'top',
            transform=ax.transAxes,
        )
        ax.set_title(
            "Temperature & Density profile along the ray\n" + \
            f"ray originated from primary star, with direction of {ray_unit_vec}\n" + \
            f"resolution = {mpdf.params['nparttot']:.2e}{plot_title_suffix}",
        )

        # plot2 - Temperature profile
        ax = axes[1]
        ylim = ylim_T
        ax.loglog(dist_to_ray0_samples, T_samples)
        ax.set_ylim(ylim)
        ax.set_ylabel(f"Temperature $T$ / {unit_temp_text}")
        #ax.set_yticks(ax.get_yticks()[:-2])    # remove top-most one tick to avoid overlap
        if photosphere_found:
            ax.axvline(photosphere_dist_to_ray0, color='green', linestyle='dashed')
            ax.axhline(photosphere['T'], color='green', linestyle='dashed')
            ax.fill_betweenx(ylim, photosphere['R1'] + photosphere['h'], photosphere['R1'] - photosphere['h'], alpha=0.1)
            ax.fill_between(
                dist_to_ray0_samples,
                y1=photosphere['T'] + photosphere['T_m'],
                y2=photosphere['T'] + photosphere['T_p'],
                alpha=0.1,# color='red',
            )
            ax.text(xlim[0], photosphere['T'] + photosphere['T_m'],
                    #f" Photosphere temperature:\n" + \
                    f" $T_{{\\rm ph}} = {photosphere['T']:.0f}" + \
                    f"^{{{photosphere['T_p']:+.0f}}}_{{{photosphere['T_m']:+.0f}}} $" + \
                    f" {unit_temp_text}",
                    va='top')
        #ax.set_xlim(xlim)
        #ax.set_xlabel(f"Distance to Primary Star $R_1$ on ray / {unit_dist_text}")


        # plot3 - Density profile
        ax = axes[2]
        ylim = ylim_rho
        ax.loglog(dist_to_ray0_samples, rho_samples)
        ax.set_ylim(ylim)
        ax.set_ylabel(f"Density $\\rho$ / {unit_rho_text}")
        #ax.set_yticks(ax.get_yticks()[:-2])    # remove top-most one tick to avoid overlap
        if photosphere_found:
            ax.axvline(photosphere_dist_to_ray0, color='green', linestyle='dashed')
            ax.axhline(photosphere['rho'], color='green', linestyle='dashed')
            ax.fill_betweenx(ylim, photosphere['R1'] + photosphere['h'], photosphere['R1'] - photosphere['h'], alpha=0.1)
            ax.text(xlim[0], photosphere['rho'],
                    #f" Photosphere density:\n" + \
                    f" $\\rho_{{\\rm ph}} = {photosphere['rho']:.2e} $ {unit_rho_text}", va='top')
        ax.set_xlim(xlim)
        ax.set_xlabel(f"Distance to Primary Star $R_1$ / {unit_dist_text}")
        
        
        # saving
        outfilename_vectxt = f"R1-x{ray_unit_vec[0]:.1f}y{ray_unit_vec[1]:.1f}z{ray_unit_vec[2]:.1f}"
        outfilename = f"{filename}__photosphere-tau+T+rho-profile__{outfilename_vectxt}.png"
        fig.savefig(outfilename)
        
        outfilenames.append(outfilename)
        plt.close(fig)
        

    # Make movie
    with ImageSequenceClip(outfilenames+outfilenames[0:1]*4, fps=fps) as vid:
        moviefilename = f'{job_name}__photosphere-tau+T+rho-profile__{outfilename_vectxt}__movie.mp4'
        vid.write_videofile(moviefilename)
    
    times = times * units.year
    lum_effs = lum_effs*units.solLum
    
    return moviefilename, times, photospheres, lum_effs


