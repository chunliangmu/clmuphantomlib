"""
Input Parameters for my scripts.
"""

# imports and internal settings
import numpy as np
from astropy import units
Msun_str = units.Msun.to_string('latex_inline')


# script input parameters

verbose = 3

PHOTOSPHERE_TAU = 1.0

ray_no = 600

plane_axes_list = ['xyz', 'xzy']

box_lim = 199

fps = 10

unitsOut = {
    'dist': units.au,
    'time': units.year,
    'temp': units.K,
}

use_saved_jsons = False


JOB_PROFILES_LIST = (
    {
        'job_name': '../photosphere/luis_2md/light',
        'file_indexes': np.arange(0, 17600+1, 100),
        'plot_title_suffix': f" for 1.7{Msun_str} primary star with Nucleation Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'X': 0.691,
        'name': f"Dusty 1.7{Msun_str}",
        #'fmt': '-',
        'color': 'blue',
    },
    {
        'job_name': '../photosphere/luis_4md/light',
        'file_indexes': np.arange(0, 17600+1, 100),
        'plot_title_suffix': f" for 3.7{Msun_str} primary star with Nucleation Dust",
        'ieos': 10,
        'params': {
            'X' : 0.686,
            'Z' : 0.024,
            'mu': 2.381,
        },
        'X': 0.686,
        'name': f"Dusty 3.7{Msun_str}",
        #'fmt': '-',
        'color': 'blue',
    },
    {
        'job_name': '../photosphere/miguel_2m/binary',
        'file_indexes': np.arange(0, 5000+1, 100),
        'plot_title_suffix': f" for 1.7{Msun_str} primary star without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'X': 0.691,
        'name': f"Non-dusty 1.7{Msun_str}",
        #'fmt': '--',
        'color': 'orange',
    },
    {
        'job_name': '../photosphere/miguel_4m/binary',
        'file_indexes': np.arange(0, 5000+1, 100),
        'plot_title_suffix': f" for 3.7{Msun_str} primary star without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.686,
            'Z' : 0.024,
            'mu': 2.381,
        },
        'X' : 0.686,
        'name': f"Non-dusty 3.7{Msun_str}",
        #'fmt': '--',
        'color': 'orange',
    },
    {
        'job_name': '../photosphere/miguel_2m_2022/binary',
        'file_indexes': np.arange(0, 6000+1, 100),
        'plot_title_suffix': f" for 1.7{Msun_str} primary star without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'X': 0.691,
        'name': f"Non-dusty 1.7{Msun_str}",
        #'fmt': '--',
        'color': 'orange',
    },
)

JOB_PROFILES = JOB_PROFILES_LIST[:]

JOB_PROFILES_GROUPS = {
    '2m': (JOB_PROFILES_LIST[0], JOB_PROFILES_LIST[2],),
    '4m': (JOB_PROFILES_LIST[1], JOB_PROFILES_LIST[3],),
}
