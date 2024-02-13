#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module that adds a class to handle phantom data.

It's a wrapper of the sarracen dataframe with some additional functions for my common envelope analysis.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import say, is_verbose
from .units_util import DEFAULT_UNITS, complete_units_dict, set_as_quantity, get_units_field_name
from .geometry import get_norm_of_vec

#  import (general)
import numpy as np
from astropy import units
from astropy import constants as const
import sarracen
import matplotlib.pyplot as plt
import matplotlib as mpl
from moviepy.editor import ImageSequenceClip




# Functions and global constants


# File handling

def get_filename_phantom_dumps(job_name: str, file_index: int) -> str:
    """Return phantom dumps filename."""
    return f'{job_name}_{file_index:05}'




# Class

class MyPhantomDataFrames:
    """An object using Sarracen data frames to deal with Phantom outputs."""
    
    def __init__(self, verbose:int=3):
        self.sdfs = ()
        self.data = {}
        self.job_name   = ''
        self.file_index = -1
        self.params = {}    # gas particles params
        self.time = 0.
        self.gamma = 0.
        self.ieos  = -1
        self.total_mass = 0.
        self.loc_CoM    = np.full(3, np.nan)     # location of the Center of Mass
        self.loc_star1  = np.full(3, np.nan)   # location of the primary star
        #self.i_star1 = np.nan    # index of the primary star in self.data['sink']
        self.units = DEFAULT_UNITS # dict of astropy quantities, with keys ['mass', 'dist', 'time']
        self.const = {}    # constants values in self.units
        self._update_units(verbose=verbose)

        
    def _update_units(self, verbose:int=3):
        """Update self.units based on mass, dist, time, & temp in self.units"""
        self.units = complete_units_dict(self.units)
        
        self.const['G'] = const.G.to_value(self.units['G'])
        self.const['sigma_sb'] = const.sigma_sb.to_value(self.units['sigma_sb'])
        if is_verbose(verbose, 'debug'):
            say(
                'debug', "mupl.MyPhantomDataFrames._update_units()", verbose,
                f"{self.units=}\n",
                f"G={self.const['G']} {self.units['G']}\n",
                f"sigma_sb={self.const['sigma_sb']} {self.units['sigma_sb']}\n",
            )
        
    
    def get_filename(self) -> str:
        """Return filename from self.job_name and self.file_index."""
        return get_filename_phantom_dumps(job_name=self.job_name, file_index=self.file_index)
    
    def get_time(self, unit=units.year) -> units.Quantity:
        """Return time of the dump as an astropy Quantity."""
        return (self.time*self.units['time']).to(unit)
    
    
    def read(
        self,
        job_name='', file_index=0,
        calc_params : list = [],
        calc_params_params : dict = {},
        reset_xyz_by : str = "",
        verbose : int = 3,
        reset_xyz_by_CoM : bool = False,
    ):
        """
        Read phantom data.
        
        Parameters
        ----------
        job_name: str
            the job name of the dump (e.g. for ../binary_00123, it is "../binary")
            
        file_index: int
            the number of the dump (e.g. for ../binary_00123, it is 123)
            
        calc_params: list of str.
            Additional parameters to calculate. passed to self.calc_sdf_params().
            See self.calc_sdf_params() for more info.
            
        calc_params_params: dict
            Parameters for calc_params. Passed to self.calc_sdf_params().
            See self.calc_sdf_params() for more info.
        
        reset_xyz_by_CoM: bool (deprecated)
            whether or not to overwrite xyz column to move origin of the coordinate system to CoM.

        reset_xyz_by: str
            whether or not to add a vector to xyz column to move origin of the coordinate system to designated places.
            acceptable input:
                "CoM": Center of Mass
                "R1" or "primary": primary star (i.e. first entry in sink, a.k.a. data['sink'].iloc[0])

        verbose: int
            How much warnings, notes, and debug info to be print on screen. 
            
        Returns self.
        """
        
        # get filename
        if len(job_name) > 0:
            self.job_name = job_name
        elif len(self.job_name) > 0:
            pass
        else:
            raise ValueError(f"Read failed- please supply job_name")
            return self
        self.file_index = file_index
        filename = self.get_filename()
        if is_verbose(verbose, 'warn'):
            say('note', 'MyPhantomDataFrames.read()', verbose, f"\n\n\tReading {filename=}\n\n")
            
        # read
        self.sdfs = sarracen.read_phantom(filename)
        if isinstance(self.sdfs, (tuple, list)):
            self.sdfs = tuple(self.sdfs)
        elif isinstance(self.sdfs, sarracen.sarracen_dataframe.SarracenDataFrame):
            self.sdfs = (self.sdfs, )
        else:
            say('warn', 'MyPhantomDataFrames.read()', verbose,
                f"\n\n\tUnexpected type of data just read: {type(self.sdfs)=}, please check code.\n\n")
        
        # set alias
        self.data = {
            'gas' : self.sdfs[0],
            #'sink': self.sdfs[1],
        }
        if len(self.sdfs) >= 2:
            self.data['sink'] = self.sdfs[1]
        self.params = self.data['gas'].params
        self.time = self.params['time']
        self.gamma = self.params['gamma']
        self.ieos = int(self.params['ieos'])
        #self.total_mass = self.params['nparttot'] * self.params['mass'] + self.data['sink']['m'].sum()
        
        # get mass & CoM
        self.data['gas'].create_mass_column()
        self.total_mass = np.sum([sdf['m'].sum() for sdf in self.sdfs])
        self.loc_CoM = self.get_loc_CoM()

            
        
        # set units
        self.units = {
            'dist': units.def_unit('udist', self.params['udist'] * units.cm),
            'mass': units.def_unit('umass', self.params['umass'] * units.g),
            'time': units.def_unit('utime', self.params['utime'] * units.s),
            'temp': units.K,
        }
        self._update_units()
        if is_verbose(verbose, 'debug'):
            say(
                'debug', 'MyPhantomDataFrames.read()', verbose,
                *[f"{self.units[i]} = {(self.units[i]/DEFAULT_UNITS[i]).decompose()} {DEFAULT_UNITS[i]}" for i in ['dist', 'mass', 'time']],
                f"{self.time = }\n{self.gamma = }\n{self.ieos = }\n{self.total_mass = }\n",
                f"Center of mass location: {self.loc_CoM = }\n",
            )

                    
        if reset_xyz_by_CoM:
            reset_xyz_by = "CoM"
        
        if not reset_xyz_by:
            if is_verbose(verbose, 'note') and get_norm_of_vec(self.loc_CoM) > 1:
                say(
                    'note', 'MyPhantomDataFrames.read()', verbose,
                    f"CoM significantly deviates from the origin with distance of {get_norm_of_vec(self.loc_CoM)}.",
                    "Consider use reset_xyz_by_CoM=True option when read?",
                )
        self.reset_xyz_by(reset_xyz_by, verbose=verbose)
        
        
        # safety tests
        if 'kappa' in self.data['gas'].columns:
            do_warn = True
            if 'kappa' in calc_params or 'opacity' in calc_params:
                if 'kappa_translate_from_cgs_units' in calc_params_params.keys():
                    if calc_params_params['kappa_translate_from_cgs_units']:
                        do_warn = False
            # warn
            if do_warn and is_verbose(verbose, 'warn'):
                say(
                    'warn', 'MyPhantomDataFrames.read()', verbose,
                    "kappa column exists.",
                    f"We here assume kappa is in phantom units {self.units['opacity']=} ",
                    "However in phantom kappa is assumed to be in cgs unit.",
                    "If so, please CONVERT KAPPA MANNUALLY into PHANTOM units BEFORE proceeding, e.g.:",
                    "\tmpdf.data['gas']['kappa'] = mupl.units_util.get_val_in_unit(", 
                    "\tmpdf.data['gas']['kappa'], units.cm**2/units.g, mpdf.units['opacity'])",
                )
        

            
        # calculate additional params
        self.calc_sdf_params(calc_params=calc_params, calc_params_params=calc_params_params, verbose=verbose)
        
        return self


    def reset_xyz_by(
        self,
        what: str = '',
        verbose: int = 3,
    ):
        """
        Reset coordinates to center on the specified thing.

        Parameters
        ----------
        
        """
        # do reset xyz
        if not what:
            # do nothing
            return self
        elif what in {'CoM'}:
            self.loc_CoM = self.get_loc_CoM()
            reset_xyz_by_arr = self.loc_CoM
        elif what in {'R1', 'primary'}:
            reset_xyz_by_arr = np.array(self.data['sink'][['x', 'y', 'z']].iloc[0])
        else:
            if is_verbose(verbose, 'err'):
                say('err', 'MyPhantomDataFrames.reset_xyz_by()', verbose,
                    f"Unknown coordinates center reseting center str {what = }",
                    "Action Cancelled.")
            return self
            
        if is_verbose(verbose, 'note'):
            say('note', 'MyPhantomDataFrames.reset_xyz_by()', verbose,
                f"Reseting Origin to {what} ({reset_xyz_by_arr})...")
        
        for sdf in self.sdfs:
            sdf['x'] -= reset_xyz_by_arr[0]
            sdf['y'] -= reset_xyz_by_arr[1]
            sdf['z'] -= reset_xyz_by_arr[2]
        self.loc_CoM = self.get_loc_CoM()
        
        if is_verbose(verbose, 'note'):
            say('note', 'MyPhantomDataFrames.reset_xyz_by()', verbose, f"CoM location is now {self.loc_CoM}")
        if is_verbose(verbose, 'warn') and what in {'', "CoM"} and get_norm_of_vec(self.loc_CoM) > 1e-5:
            say('warn', 'MyPhantomDataFrames.reset_xyz_by()', verbose,
                f"CoM is not close to origin {get_norm_of_vec(self.loc_CoM) = }")
    
    
    def calc_sdf_params(
        self,
        calc_params: list|set = [],
        calc_params_params : dict = {
            'ieos': None,
            'overwrite': False,
            'kappa_translate_from_cgs_units': False,
        },
        verbose: int = 3,
    ):
        """
        Calculate density, mass, velocity, and more for self.data['gas'].
        
        Assume "xyzh" columns in self.data['gas'].
        
        Parameters
        ----------
        calc_params: list of str
            Additional parameters to calculate.
            Allowed str:
                'pressure' or 'P'  (will get pressure, sound speed & mach number)
                'temperature' or 'T'
                'opacity' or 'kappa' (a simplified formula for gas opacity: 2e-4 cm2/g for T<6000, 0.2(1+X) cm2/g for T>6000)
                'R1': distance to the primary star (self.data['sink'].iloc[0])
                'vr': speed relative to the origin of the coordinate system (positive for expansion/escaping, negative for contracting) 
                
        calc_params_params: dict
            Parameters for calc_params.
            Used parameters:
                'ieos': if None, will use self.ieos
                'overwrite': if True, will overwrite existing column
                For calc 'T' (temperature):
                    'mu': float (mean molecular weight)
                For calc 'kappa' (opacity):
                    'X': float (Hydrogen mass fraction)
                    'kappa_translate_from_cgs_units': if True, will interpret existing kappa column in cgs unit,
                        and attempt to translate its unit into the (correct) phantom units.
                        Won't do anything is 'kappa' column not present.

        verbose: int
            How much warnings, notes, and debug info to be print on screen. 
            
        
        Returns self.
        """
        
        sdf = self.data['gas']
        
        ieos = None
        if 'ieos' in calc_params_params.keys():
            ieos = calc_params_params['ieos']
        if ieos is None:
            ieos = self.ieos
            
        overwrite = False
        if 'overwrite' in calc_params_params.keys():
            overwrite = calc_params_params['overwrite']
            

        ## get mass
        #sdf.create_mass_column()

        # get density
        if 'rho' not in sdf.columns:
            sdf.calc_density()
        else:
            if verbose >= 2: print(f"    Note: Density column rho already exist in {self.time = }.")
                
        # get speed if velocity presents
        if all([key in sdf.columns for key in ('vx', 'vy', 'vz')]):
            sdf['v'] = (sdf['vx']**2 + sdf['vy']**2 + sdf['vz']**2)**0.5
            
            
        # get temperature
        if 'T' in calc_params or 'temperature' in calc_params:
            # safety check
            do_this = True
            if 'T' in sdf.columns and np.any(sdf['T']):
                if overwrite:
                    if verbose >= 1: print(f"**  Warning: Overwriting non-zero temperature column 'T' already in the datafile.")
                else:
                    do_this = False
                    if verbose >= 2: print(f"*   Note: non-zero temperature column 'T' already in the datafile. Calc Cancelled.")
            elif 'temperature' in sdf.columns and np.any(sdf['temperature']):
                sdf['T'] = sdf['temperature']
                do_this = False
                if verbose >= 2: print(f"*   Note: Using non-zero temperature column 'temperature' as 'T' column.")
            if 'u' not in sdf.columns:
                raise ValueError(f"No column for specific internal energy u found in {sdf.columns = }")
            if 'mu' in calc_params_params.keys():
                mu = calc_params_params['mu']
            else:
                if 'mu' in sdf.columns:
                    mu = sdf['mu']
                else:
                    mu = None
                    
            if do_this:
                if verbose >= 3: print(f"    Info: Using {ieos= } for temperature calc.")
                # calc T
                sdf['T'] = get_temp_from_u(
                    rho= sdf['rho'], rho_unit= self.units['density'],
                    u  = sdf['u']  ,   u_unit= self.units['specificEnergy'],
                    mu = mu, ieos = ieos,
                    verbose=verbose,
                )
        
        
        
        # get opacity
        if 'kappa' in calc_params or 'opacity' in calc_params:
            # safety check
            kappa_translate_from_cgs_units = False
            if 'kappa_translate_from_cgs_units' in calc_params_params.keys():
                kappa_translate_from_cgs_units = calc_params_params['kappa_translate_from_cgs_units']
            do_this = True
            if 'kappa' in sdf.columns and np.any(sdf['kappa']):
                if kappa_translate_from_cgs_units:
                    do_this = False
                    sdf['kappa'] = set_as_quantity(sdf['kappa'], units.cm**2/units.g).to_value(self.units['opacity'])
                    if verbose >= 2: print(f" Translating kappa from cgs units to phantom units {self.units['opacity'].cgs =}")
                elif overwrite:
                    if verbose >= 1: print(f"**  Warning: Overwriting non-zero opacity column 'kappa' already in the datafile.")
                else:
                    do_this = False
                    if verbose >= 2: print(f"*   Note: non-zero opacity column 'kappa' already in the datafile. Calc Cancelled.")
            if do_this:
                if 'T' not in sdf.columns:
                    raise ValueError(f"No column for temperature 'T' found in {sdf.columns = }. Try calc 'T' as well?")
                if 'X' in calc_params_params.keys():
                    X = calc_params_params['X']
                else:
                    X = 0.7389
                # estimate kappa
                sdf['kappa'] = set_as_quantity(
                    np.where(sdf['T'] < 6000., 2e-4, 0.2*(1+X)), # in cgs unit
                    units.cm**2/units.g,
                ).to_value(self.units['opacity'])
        
        

        # get pressure
        if 'P' in calc_params or 'pressure' in calc_params:
            # safety check
            do_this = True
            if 'P' in sdf.columns and np.any(sdf['P']):
                if overwrite:
                    if verbose >= 1: print(f"**  Warning: Overwriting non-zero pressure column 'P' already in the datafile.")
                else:
                    do_this = False
                    if verbose >= 2: print(f"*   Note: non-zero pressure column 'P' already in the datafile. Calc Cancelled.")
            if do_this:
                if ieos == 2:
                    # adiabatic (or polytropic - *** not implemented!!!) eos
                    if 'u' in sdf.columns:
                        sdf['P'] =  eos_adiabatic(gamma=self.gamma, rho=sdf['rho'], u=sdf['u'])
                    else:
                        raise NotImplementedError(
                            f"No u column found in {self.time = } !\n" + \
                            "Are you using Polytropic eos? Polytropic EoS has NOT yet been implemented.")
                elif ieos == 10:
                    # mesa eos
                    sdf['P'] = sdf['pressure']
                else:
                    if verbose >= 1: print(f"*   Warning: Unrecognizaed ieos when calc P: {ieos = } !")
            # sanity check
            if any(sdf['P'] <= 0):
                if verbose >= 1: print(f"*   Warning: In {self.time = } there exists non-positive pressure!")
            else:
                if verbose >= 2: print(f"    Note: All pressure in {self.time = } are positive.")
                    
            # get sound speed & mach number
            if overwrite or 'c_s' not in sdf.columns:
                sdf['c_s'] = (self.gamma * sdf['P'] / sdf['rho'])**0.5
            if 'v' in sdf.columns and (overwrite or 'mach' not in sdf.columns):
                sdf['mach'] = sdf['v'] / sdf['c_s']
                
        # get distance to the primary star
        if 'R1' in calc_params:
            data_star = self.data['sink'].iloc[0]
            sdf['R1'] = ((sdf['x'] - data_star['x'])**2 + (sdf['y'] - data_star['y'])**2 + (sdf['z'] - data_star['z'])**2)**0.5

        if 'vr' in calc_params:
            # speed relative to the origin of the coordinate system
            v_vecs = np.array(sdf[['vx', 'vy', 'vz']])
            r_vecs = np.array(sdf[[ 'x',  'y',  'z']])
            sdf['vr'] = np.sum(v_vecs * r_vecs, axis=1) / np.sum(r_vecs**2, axis=1)**0.5
        return self
    
    
    def get_dyn_timescale_star(self, star_radius, star_mass=None, unit_time=DEFAULT_UNITS['time']):
        """
        Get dynamical timescale for star.
        
        By default uses total mass of the star.
        
        Returns
        -------
        tau_dyn: astropy.units.quantity.Quantity
        """
        
        # set mass to default if None; set mass & radius units
        if star_mass is None:
            star_mass = self.total_mass * self.units['mass']
        else:
            star_mass = set_as_quantity(star_mass, self.units['mass'], copy=False)
            
        star_radius = set_as_quantity(star_radius, self.units['dist'], copy=False)
        
        # get tau_dyn
        tau_dyn = ((star_radius**3 / (2*const.G*star_mass))**0.5).to(unit_time, copy=False)
        return tau_dyn
    
    
    def get_loc_CoM(self, recalc_total_mass=False) -> np.ndarray:
        """Get the locaton of the Center of Mass.
        
        Returns a 3D numpy array (NO UNITS!).
        """
        if recalc_total_mass:
            total_mass = np.sum([sdf['m'].sum() for sdf in self.sdfs])
        else:
            total_mass = self.total_mass
        return np.sum([[(sdf[axis]*sdf['m']).sum() for axis in 'xyz'] for sdf in self.sdfs], axis=0) / total_mass
    
    def get_orb_sep(self, unit=units.au, verbose:int=3) -> units.Quantity:
        """
        Get orbital separation between the two sink particles.
        
        Requires that exactly two sink particles to be in data frame.
        
        Returns
        -------
        orb_sep: float
            orbital separation
        """
        # sanity check
        if len(self.data['sink']) != 2:
            if len(self.data['sink']) <= 1:
                if is_verbose(verbose, 'err'):
                    say(
                        'err', 'MyPhantomDataFrames.get_orb_sep()', verbose,
                        f"In {self.time = } Less than two sink particles detected. Cannot calc orb_sep.")
                return np.nan
            else:
                if is_verbose(verbose, 'warn'):
                    say(
                        'warn', 'MyPhantomDataFrames.get_orb_sep()', verbose,
                        f"In {self.time = } More than two sink particles detected. Using first 2 to calc orb_sep.")
        # calc
        sinks = self.data['sink']
        orb_sep = np.sum([(sinks[axis][0] - sinks[axis][1]) ** 2 for axis in 'xyz'])**0.5
        orb_sep = set_as_quantity(orb_sep, self.units['dist']).to(unit)
        return orb_sep

    
    def get_val(self, val_name:str, dataset_name:str='gas', as_quantity:bool=True, copy:bool=True):
        """Get value (as astropy quantity by default) from dataset"""
        ans = self.data[dataset_name][val_name]
        if as_quantity:
            ans = set_as_quantity(ans, self.units[get_units_field_name(val_name)], copy=copy)
        return ans
    
    
    def plot_render(
        self,
        fig=None, ax=None, do_ax_titles=True,
        savefilename=False, plot_title_suffix='',
        xyz_axis='xyz', xsec=0.0, rendered = 'rho',
        box_lim: None|float|units.quantity.Quantity = None,
        xlim=None, #(-10000, 10000)*DEFAULT_UNITS['dist'],
        ylim=None, #(-10000, 10000)*DEFAULT_UNITS['dist'],
        cbar=True,
        norm=mpl.colors.LogNorm(vmin=None, vmax=None, clip=True),
        unit_time=units.year, unit_time_print_utime_too=False,
        **kwargs
    ):
        """
        Plot a sarracen render plot.
        
        Very few safety checks are performed- make sure you enter sane params.
        
        Parameters
        ----------
        fig, ax: mpl.figure.Figure, mpl.axes._axes.Axes
            fig & ax for plotting. if either set to None, will get a new fig & ax
            
        do_ax_titles: bool
            If True, will set ax titles and x/y labels. If False, will skip these (useful for subplots).
            
        savefilename: False, None, or Str
            If False, will not save and will return ax.
            If None, will save with default name and return out filename.
            If str, will save to this filename and return it.
            
        plot_title_suffix: str
            Text to add to the end of the plot title. Use '' if you don't need to add anything.

        xyz_axis: str (len==3)
            set x, y axis of the plot & the cross-sec direction z.
            e.g. 'yzx' will plot a slice of x=y, y=z plot at xsec of x=xsec.
            Warning: There is no safety check for this par. Make sure you enter something sane.

        xsec: float
            cross-section location.
            Setting the plot to plot a slice at xyz_axis[2]=xsec.

        rendered: str
            the variable to be rendered in the plot.

        box_lim: float or astropy.units.quantity.Quantity
            if supplied, will overwrite xlim and ylim.
            Quantity (*** To Be Implemented***)
        
        xlim, ylim: (float, float) or astropy.units.quantity.Quantity (len==2)
            lim of x & y axis of the plot.
            If Quantity, the unit of x/y axis will be set to its unit. (*** To Be Implemented***)
            
        norm: mpl.colors.Normalize
            Normalization for matplotlib (mpl) color bar.
            Use either mpl.colors.Normalize or mpl.colors.LogNorm,
            e.g. norm = mpl.colors.LogNorm(vmin=None, vmax=None, clip=True),
            
        cbar: bool
            True if a colorbar should be drawn.

        unit_time: astropy.units.Unit
            Unit for the time stamp.
            
        unit_time_print_utime_too: bool
            If true, will print an additional line of time stamp in code unit time.
            
        **kwargs: other keyword arguments
            Keyword arguments to pass to sdf.render.
        

        Returns
        -------
        savefilename: str
            filename of the plot.
            returns this if savefilename is not False.
            
        fig, ax: mpl.figure.Figure, mpl.axes._axes.Axes
            returns this if savefilename is False.
        """
        # get/clear ax
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax.clear()
        
        # set x,y,z axis to be 'x', 'y', or 'z'
        x_axis = xyz_axis[0]
        y_axis = xyz_axis[1]
        z_axis = xyz_axis[2]
        
        # set data
        sdf = self.data['gas']
        
        # set x/y axis unit
        if box_lim is not None:
            if isinstance(box_lim, units.quantity.Quantity):
                xlim = set_as_quantity((-box_lim.value, box_lim.value), box_lim.unit)
                ylim = xlim
                raise NotImplementedError()
            else:
                xlim = (-box_lim, box_lim)
                ylim = (-box_lim, box_lim)
        
        if isinstance(xlim, units.quantity.Quantity):
            xunit = xlim.unit
            xunit_txt = f" / {xunit.to_string('latex_inline')}"
            xlim = xlim.value
            raise NotImplementedError()
        else:
            xunit = None
            xunit_txt = ''
        if isinstance(ylim, units.quantity.Quantity):
            yunit = ylim.unit
            yunit_txt = f" / {yunit.to_string('latex_inline')}"
            ylim = ylim.value
            raise NotImplementedError()
        else:
            yunit = None
            yunit_txt = ''
        
        # render
        ax = sdf.render(
            rendered, x=x_axis, y=y_axis,
            xsec=xsec,
            ax=ax, cbar=cbar,
            xlim=xlim, ylim=ylim,
            norm=norm,
            **kwargs,
        )
        if do_ax_titles:
            ax.set_xlabel(f"{x_axis}{xunit_txt}")
            ax.set_ylabel(f"{y_axis}{yunit_txt}")
            ax.set_title(
                f"Cross-Section at {z_axis} = {xsec}\n" + \
                f"resolution = {sdf.params['nparttot']:.2e}{plot_title_suffix}",
                fontsize=10,
            )
        
        # set time stamp text
        if unit_time is None:
            unit_time = self.units['time']
        time_txt = f"Time = {self.get_time(unit=unit_time).value:.2f} {unit_time.to_string('latex_inline')}"
        if unit_time_print_utime_too:
            time_txt += f"\n (= {self.time:.0f} unit)"
        ax.text(
            0.98, 0.98, time_txt,
            color = "white", ha = 'right', va = 'top',
            transform=ax.transAxes,
        )
        
        if savefilename is False:
            # plot & exit
            return fig, ax
        elif savefilename is None:
            filename = self.get_filename()
            savefilename = f"{filename}__{x_axis}-{y_axis}-{rendered}__{z_axis}={xsec:.1f}.png"
        fig.savefig(savefilename)
        fig.clf()
        return savefilename
