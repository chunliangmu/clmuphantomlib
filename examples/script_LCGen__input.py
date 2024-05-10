"""
Input Parameters for my scripts.
"""

# imports and internal settings
from _photosphere_jobProfiles__input import interm_dir, output_dir, unitsOut, PHOTOSPHERE_TAU, JOB_PROFILES_DICT #, fps


verbose = 6

job_nicknames = ['2md', '4md']#, '4m', '2m_2022']
xyzs_list  = ['xyz', 'xzy']
no_xy=(16, 16)
no_xy_txt = 'x'.join([f'{i}' for i in no_xy])
output_dir = f'../fig/20240222_LCGen/{no_xy_txt}/test_'
verbose_loop = 0
