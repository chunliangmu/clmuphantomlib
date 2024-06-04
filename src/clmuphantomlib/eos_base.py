#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[Deprecated]


A sub-module for handling Equation of State (EoS) related stuff.
Defines the base class for EoS.

Owner: Chunliang Mu
"""



from .log import say

say('warn', None, True, "clmuphantomlib.eos_base module is Deprecated. Use clmuphantomlib.eos.base instead.")


from .eos.base import *
