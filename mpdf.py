#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module that adds a class to handle phantom data.

It's a wrapper of the sarracen dataframe with some additional functions for my common envelope analysis.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from .log import error, warn, note, debug_info

#  import (general)
from astropy import units
from astropy import constants as const
import sarracen
import matplotlib.pyplot as plt
import matplotlib as mpl
from moviepy.editor import ImageSequenceClip




# *** TBD ***