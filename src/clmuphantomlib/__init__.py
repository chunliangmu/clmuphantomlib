#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Chunliang Mu's phantom data analysis library


Author: Chunliang Mu (at Macquarie University, expected duration 2023-2026)
Principal Supervisor: Professor Orsola De Marco

This library is based on the wonderful python package 'sarracen'.
It is developped as part of my PhD project
"Non-adiabatic common envelope simulation of massive stars"
(This project title is provisional as of 2023-10-16.)

Work on this project started on 2023-01-25.

Assuming temperature unit being K.
Reads & handles other units from phantom data dumps.


Note:
MyPhantomDataFrames column names:
    rho: density
    m  : mass
    h  : smoothing length
    u  : specificInternalEnergy
    T  : temperature


-------------------------------------------------------------------------------

Side note: Remember to limit line length to 79 characters according to PEP-8
https://peps.python.org/pep-0008/#maximum-line-length
which is the length of below line of '-' characters.

-------------------------------------------------------------------------------

"""


from . import sph_interp


from .settings  import DEFAULT_SETTINGS
from .geometry  import *
#from .readwrite import *
from .io        import *
from .light     import get_photosphere_on_ray
from .eos       import get_eos, get_eos_opacity
from .mpdf      import get_filename_phantom_dumps, MyPhantomDataFrames

# .main is deprecrated but we will go with this for now.
from .main      import *
