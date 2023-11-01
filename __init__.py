#!/usr/bin/env python


"""
Chunliang Mu's phantom data analysis library


Author: Chunliang Mu (at Macquarie University, expected duration 2023-2026)
Principal Supervisor: Professor Orsola De Marco

This library is based on the wonderful python package 'sarracen'.
It is developped as part of my PhD project "Non-adiabatic common envelope simulation of massive stars"
(This project title is provisional as of 2023-10-16.)

Work on this project started on 2023-01-25.

Assuming temperature unit being K. Reads & handles other units from phantom data dumps.
"""



from .geometry import *
from .readwrite import *
from .light import *

from .main  import *