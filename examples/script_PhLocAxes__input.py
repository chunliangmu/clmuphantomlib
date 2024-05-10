"""
Input Parameters for my scripts.
"""

# imports and internal settings
from _photosphere_jobProfiles__input import out_dir, fps, unitsOut, PHOTOSPHERE_TAU, JOB_PROFILES_LIST, JOB_PROFILES_GROUPS, JOB_PROFILES_DICT
from clmuphantomlib.units_util import DEFAULT_UNITS


# script input parameters

verbose = 3


# units used in the dump file
# *** WARNING: script does not check this yet!
#      Change it if you are not using Rsun & Msun for phantom dumps!!!
unitsIn = DEFAULT_UNITS

JOB_PROFILES = JOB_PROFILES_LIST[:]
