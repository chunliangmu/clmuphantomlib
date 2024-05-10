#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input Parameters describing sims saved in ../photosphere/
"""

# imports and internal settings
import numpy as np
from astropy import units
Msun_str = units.Msun.to_string('latex_inline')

# output directory- remember to add the final sep '/' at the end
interm_dir = '../interm/'
output_dir = '../fig/'
out_dir    = output_dir  # deprecated keyword


fps = 6

unitsOut = {
    'dist': units.au,
    'time': units.year,
    'temp': units.K,
    'density': units.g / units.cm**3,
    'opacity': units.cm**2 / units.g,
    'dimless': units.dimensionless_unscaled,
    'speed': units.km/units.s,
}


PHOTOSPHERE_TAU = 1.0


# script input parameters
JOB_PROFILES_LIST = (
    {
        'raw_dir' : '../raw/luis_2md/',
        'file_prefix': 'light',
        #'job_name': '../photosphere/luis_2md/light', # deprecated keyword- will still be added automatically later
        #'file_indexes': np.arange(0, 17600+1, 100),
        'file_indexes': np.concatenate((np.arange(0, 3400-1, 20), np.arange(3400, 17600+1, 100))),
        'plot_title_suffix': f" for 1.7{Msun_str} primary with Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'name': f"Dusty 1.7{Msun_str}",
        'nickname': '2md',
        #'fmt': '-',
        'color': 'blue',
    },
    {
        'raw_dir' : '../raw/luis_4md/',
        'file_prefix': 'light',
        #'job_name': '../photosphere/luis_4md/light',
        'file_indexes': np.arange(0, 17600+1, 100),
        'plot_title_suffix': f" for 3.7{Msun_str} primary with Dust",
        'ieos': 10,
        'params': {
            'X' : 0.686,
            'Z' : 0.024,
            'mu': 2.381,
        },
        'name': f"Dusty 3.7{Msun_str}",
        'nickname': '4md',
        #'fmt': '-',
        'color': 'blue',
    },
    {
        'raw_dir' : '../raw/miguel_2m/',
        'file_prefix': 'binary',
        #'job_name': '../photosphere/miguel_2m/binary',
        'file_indexes': np.arange(0, 5000+1, 100),
        'plot_title_suffix': f" for 1.7{Msun_str} primary without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'name': f"Non-dusty 1.7{Msun_str}",
        'nickname': '2m',
        #'fmt': '--',
        'color': 'orange',
    },
    {
        'raw_dir' : '../raw/miguel_4m/',
        'file_prefix': 'binary',
        #'job_name': '../photosphere/miguel_4m/binary',
        'file_indexes': np.arange(0, 5000+1, 100),
        'plot_title_suffix': f" for 3.7{Msun_str} primary without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.686,
            'Z' : 0.024,
            'mu': 2.381,
        },
        'name': f"Non-dusty 3.7{Msun_str}",
        'nickname': '4m',
        #'fmt': '--',
        'color': 'orange',
    },
    {
        'raw_dir' : '../raw/miguel_2m_2022/',
        'file_prefix': 'binary',
        #'job_name': '../photosphere/miguel_2m_2022/binary',
        'file_indexes': np.arange(0, 6000+1, 100),
        'plot_title_suffix': f" for 1.7{Msun_str} primary without Dust",
        'ieos': 10,
        'params': {
            'X' : 0.691,
            'Z' : 0.021,
            'mu': 2.381,
        },
        'name': f"Non-dusty 1.7{Msun_str}",
        'nickname': '2m_2022',
        #'fmt': '--',
        'color': 'orange',
    },
)

for job_profile in JOB_PROFILES_LIST:
    # deprecated keyword generation
    job_profile['job_name'] = job_profile['raw_dir'] + job_profile['file_prefix']
    job_profile['X'] = job_profile['params']['X']

JOB_PROFILES_GROUPS = {
    '2m': (JOB_PROFILES_LIST[0], JOB_PROFILES_LIST[2],),
    '4m': (JOB_PROFILES_LIST[1], JOB_PROFILES_LIST[3],),
}

JOB_PROFILES_DICT = { job_profile['nickname']: job_profile for job_profile in JOB_PROFILES_LIST }