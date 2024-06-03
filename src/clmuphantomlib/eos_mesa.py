#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[Deprecated]


A sub-module to deal with MESA EoS (from its table).

Owner: Chunliang Mu
"""



# Init

from .log import say

say('warn', None, True, "clmuphantomlib.eos_mesa module is Deprecated. Use clmuphantomlib.eos.mesa instead.")


from eos.mesa import *
