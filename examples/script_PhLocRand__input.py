"""
Input Parameters for my scripts.
"""

# imports and internal settings
from _photosphere_jobProfiles__input import out_dir, unitsOut, PHOTOSPHERE_TAU, JOB_PROFILES_DICT #, fps


job_nicknames = ['4md', '2md', '4m', '2m_2022']

verbose = 3

ray_no = 2000

# use even number
#    for binning purposes (since we are plotting abs(cos) figs so the plot is "folded" in half)
cos_theta_sample_no = 20

fps = 1
