#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for reading fortran files.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from ..log   import say, is_verbose
#from .shared import _add_metadata, get_str_from_astropyUnit, get_compress_mode_from_filename


#  import (general)
import struct
import io






# Functions










# ----------------------------- #
# - Fortran-related read func - #
# ----------------------------- #



def fortran_read_file_unformatted(
    fp: io.BufferedReader,
    t: str,
    no: None|int = None,
    verbose: int = 3,
) -> tuple:
    """Read one record from an unformatted file saved by fortran.

    Because stupid fortran save two additional 4 byte int before and after each record respectively when saving unformatted data.
    (Fortran fans please don't hit me)

    Parameters
    ----------
    fp: io.BufferedReader:
        File object you get with open(), with read permission.

    t: str
        Type of the data. Acceptable input:
            'i' | 'int'    | 'integer(4)': 4-byte integer
            'f' | 'float'  | 'real(4)'   : 4-byte float
            'd' | 'double' | 'real(8)'   : 8-byte float
    no: int|None
        Number of data in this record. if None, will infer from record.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.
    """

    if t in ['i', 'int', 'integer(4)']:
        t_format = 'i'
        t_no_bytes = 4
    elif t in ['f', 'float', 'real(4)']:
        t_format = 'f'
        t_no_bytes = 4
    elif t in ['d', 'double', 'real(8)']:
        t_format = 'd'
        t_no_bytes = 8
    else:
        say('err', None, verbose,
            f"Unrecognized data type t={t}."
        )
        if is_verbose(verbose, 'fatal'):
            raise NotImplementedError("Unrecognized data type. Please add it to the code.")
    
    rec_no_bytes = struct.unpack('i', fp.read(4))[0]
    no_in_record = int(rec_no_bytes / t_no_bytes)
    if no is None:
        no = no_in_record
        rec_no_bytes_used = rec_no_bytes
    else:
        rec_no_bytes_used = no * t_no_bytes
        if no != no_in_record:
            say('warn', None, verbose,
                f"Supplied no={no} does not match the record no_in_record={no_in_record}.",
                "Incorrect type perhaps?",
                "will continue to use supplied no regardless."
            )

    data = struct.unpack(f'{no}{t_format}', fp.read(rec_no_bytes_used))
    rec_no_bytes_again = struct.unpack('i', fp.read(4))[0]
    if rec_no_bytes != rec_no_bytes_again:
        say('warn', None, verbose,
            "The no of bytes recorded in the beginning and the end of the record did not match!",
            f"Beginning is {rec_no_bytes}, while end is {rec_no_bytes_again}.",
            "This means something is seriously wrong.",
            "Please Check if data sturcture is correct and file is not corrupted.",
        )
    return data

