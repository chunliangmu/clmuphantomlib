#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for reading / writing intermedian hdf5 files.

Owner: Chunliang Mu
"""



# Init


#  import (my libs)
from ..log   import say, is_verbose
from ._shared import _add_metadata, get_str_from_astropyUnit, get_compress_mode_from_filename


#  import (general)
import sys
import h5py
import gzip
import numpy as np
from astropy import units
import io
import os
import shutil





HDF5_ATTRS_ACCEPTABLE_TYPES : tuple = (
    int, float, str,
    bool, np.bool_,
    np.float32, np.float64,
    np.int32, np.int64,
)





# Functions










# ---------------------------------- #
# - HDF5-related read / write func - #
# ---------------------------------- #




def _hdf5_dump_metadata(
    metadata: dict,
    grp     : h5py.Group,
    add_data: bool = True,
    verbose : int = 3,
) -> None:
    """Dump metadata to grp.attrs.

    Parameters
    ----------
    metadata: dict
        data to be written.
        If not dict, will raise Error if verbose, or do NOTHING if otherwise.

    grp: h5py.File | h5py.Group
        hdf5 data file, where data will be written to.

    add_data: bool
        Add additional metadata info.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.
    """
    if isinstance(metadata, dict):
        # safety check
        if is_verbose(verbose, 'warn') and sys.getsizeof(metadata) + sys.getsizeof(grp.attrs) >= 65535:
            say('warn', None, verbose,
                "Potentially large metadata size:",
                f"(Adding {sys.getsizeof(metadata)/1024:.1f}KB to {sys.getsizeof(grp.attrs)/1024:.1f}KB).",
                "Should be less than 64KB.",
                sep=' ',
            )
        _add_metadata(grp.attrs, add_data=add_data, verbose=verbose)
        for key in metadata.keys():
            if   isinstance(metadata[key], dict):
                # add metadata to individual datasets
                if key in grp.keys():
                    _hdf5_dump_metadata(metadata[key], grp[key], add_data=False, verbose=verbose)
                elif is_verbose(verbose, 'err'):
                    say('err', None, verbose, f"{key=} in {metadata.keys()=}, but not in {grp.keys()}.")
            elif isinstance(metadata[key], HDF5_ATTRS_ACCEPTABLE_TYPES):
                grp.attrs[key] = metadata[key]
            elif is_verbose(verbose, 'err'):
                say('err', None, verbose, f"Unexpected metadata[{key=}] type: {type(metadata[key])}.")
    else:
        if is_verbose(verbose, 'fatal'):
            raise TypeError(f"metadata {type(metadata)=} should be of type 'dict'.")
    return



def _hdf5_dump_sub(
    data    : dict,
    grp     : h5py.Group,
    metadata: dict|None = {},
    add_metadata: bool = True,
    verbose : int = 3,
) -> None:
    """Encode the data and dump to grp.

    Suitable for storing medium/large machine-readable files.

    Do NOT put weird characters like '/' in obj.keys().
    obj['_meta_'] will be stored as metadata in grp.attrs.

    Parameters
    ----------
    data: dict
        data to be written.

    grp: h5py.File | h5py.Group
        hdf5 data file, where data will be written to.
        
    metadata: dict | None
        meta data to be added to file. The code will also save some of its own metadata.
        set it to None to disable this feature.

    add_metadata: bool
        Add additional metadata info.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.
    """
    
    # write data to file
    if isinstance(data, dict):
        
        
        for key in data.keys():
            obj = data[key]
            
            # sanity check
            if is_verbose(verbose, 'fatal') and not isinstance(key, str):
                # must be in str because it's the folder path within hdf5 files
                raise TypeError(f"key={key} of dict 'data' should be of type 'str', but it is of type {type(key)}.")

            # hold for metadata
            if key in {'_meta_'}:
                # wait till after to write in case we want to write metadata to some of the datasets too
                pass
            else:
                # parse into data and dump
                
                if   isinstance( obj, type(None) ):
                    sav = grp.create_dataset(key, dtype='f')
                    sav.attrs['_type_'] = 'None'
                    
                elif isinstance( obj, HDF5_ATTRS_ACCEPTABLE_TYPES ):
                    if ('_meta_' in data.keys() and key in data['_meta_'].keys()) or (isinstance(metadata, dict) and key in metadata.keys()):
                        sav = grp.create_dataset(key, dtype='f')
                        sav.attrs['_data_'] = obj
                        sav.attrs['_type_'] = False
                    else:
                        sav = grp['_misc_'] if '_misc_' in grp.keys() else grp.create_dataset('_misc_', dtype='f')
                        sav.attrs[  key   ] = obj

                elif isinstance( obj, (tuple, list) ):
                    obj_elem_type = type(obj[0])
                    np_array_like = issubclass(obj_elem_type, (float, int))
                    # check type coherence
                    for obj_elem in obj:
                        if obj_elem_type != type(obj_elem):
                            np_array_like = False
                    if np_array_like:
                        sav = grp.create_dataset(key, data=np.array(obj))
                        sav.attrs['_type_'] = 'numpy.ndarray'
                    else:
                        sav = grp.create_group(key)
                        _hdf5_dump_sub({str(i): iobj for i, iobj in enumerate(obj)}, sav, metadata=None, add_metadata=False, verbose=verbose)
                        sav.attrs['_type_'] = 'tuple'
                        
                elif type(obj) is np.ndarray:
                    if len(obj.shape):    # array-like
                        sav = grp.create_dataset(key, data=obj)
                    else:                 # scalar
                        sav = grp.create_dataset(key, dtype='f')
                        sav.attrs['_data_'] = obj.item()
                    sav.attrs['_type_'] = 'numpy.ndarray'
                        
                elif type(obj) is units.Quantity:
                    if len(obj.shape):    # array-like
                        sav = grp.create_dataset(key, data=obj.value)
                    else:                 # scalar
                        sav = grp.create_dataset(key, dtype='f')
                        sav.attrs['_data_'] = obj.value
                    sav.attrs['_type_'] = 'astropy.units.Quantity'
                    sav.attrs['_unit_'] = get_str_from_astropyUnit(obj.unit)

                elif isinstance(obj, type):
                    sav = grp.create_dataset(key, dtype=obj)
                    sav.attrs['_type_'] = 'type'

                elif isinstance(obj, dict):
                    sav = grp.create_group(key)
                    _hdf5_dump_sub(obj, sav, metadata=None, add_metadata=False, verbose=verbose)
                    sav.attrs['_type_'] = 'dict'
                    
                else:
                    # Not yet implemented
                    if is_verbose(verbose, 'fatal'):
                        raise NotImplementedError(
                            f"I haven't yet implemented storing data type {type(obj)} in hdf5 for data['{key}'] = {obj}")

                
        if '_meta_' in data.keys():
            # write meta data as of the data
            _hdf5_dump_metadata(data['_meta_'], grp, add_data=add_metadata, verbose=verbose)
            
    else:
        if is_verbose(verbose, 'fatal'):
            raise TypeError(f"Incorrect input type of data: {type(data)}. Should be dict.")

    # write more metadata
    if metadata is not None:
        _hdf5_dump_metadata(metadata, grp, add_data=add_metadata, verbose=verbose)

    return





def _hdf5_load_sub(
    data    : dict,
    grp     : h5py.Group,
    load_metadata : bool = True,
    verbose : int = 3,
) -> dict:
    """load from grp, decode and put into data.

    Suitable for storing medium/large machine-readable files.

    Do NOT put weird characters like '/' in obj.keys().
    obj['_meta_'] will be stored as metadata in grp.attrs.

    Parameters
    ----------
    data: dict
        dict to be load into.

    grp: h5py.File | h5py.Group
        hdf5 data file, where data will be load from.
        
    load_metadata : bool
        Do NOT load meta data from loaded dict.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.

        
    Returns
    -------
    data: original data
    """
    
    # re-construct data from file
    if isinstance(data, dict):
        
        if load_metadata:
            data['_meta_'] = dict(grp.attrs)

        for key in grp.keys():
            
            obj = grp[key]

            if isinstance(obj, h5py.Group):    # is dict

                if load_metadata:
                    data['_meta_'][key] = dict(obj.attrs)
                
                data[key] = {}
                _hdf5_load_sub(data[key], obj, load_metadata=load_metadata, verbose=verbose)


                if '_type_' in obj.attrs.keys() and obj.attrs['_type_'] in {'tuple'}:
                    try:
                        data_temp = {k: v for k, v in data[key].items() if k not in {'_meta_'}}
                        data[key] = tuple([data_temp[i] for i in sorted(data_temp, key=lambda x: int(x))])
                    except ValueError:
                        if is_verbose(verbose, 'err'):
                            say('err', None, verbose,
                                f"Unexpected input: cannot convert {key=} from dict to tuple,",
                                f"because {data[key].keys()=} cannot each be converted to integers.")
                
            elif isinstance(obj, h5py.Dataset):    # is data


                if load_metadata and key not in {'_misc_'}:   # load metadata
                    data['_meta_'][key] = dict(obj.attrs)

                
                if key in {'_misc_'}: # is small pieces of data
                    for k in obj.attrs.keys():
                        data[k] = obj.attrs[k]
                        
                elif obj.shape:       # is array

                    data[key] = obj
                    if '_type_' in obj.attrs.keys():
                        if   obj.attrs['_type_'] in {'numpy.ndarray'}:
                            data[key] = np.array(data[key])
                        elif obj.attrs['_type_'] in {'astropy.units.Quantity'} and '_unit_' in obj.attrs.keys():
                            data[key] = units.Quantity(value=data[key], unit=obj.attrs['_unit_'])
                        elif is_verbose(verbose, 'err'):
                            say('err', None, verbose, f"Unexpected input {dict(obj.attrs)=}")
                    elif is_verbose(verbose, 'err'):
                        say('err', None, verbose, f"Unexpected input {dict(obj.attrs)=}")

                else:    # is scalar
                    
                    # load data
                    data[key] = None
                    if   '_data_' in obj.attrs.keys():
                        data[key] = obj.attrs['_data_']
                    elif is_verbose(verbose, 'err') and '_type_' not in obj.attrs.keys():
                        say('err', None, verbose, f"Unexpected input {dict(obj.attrs)=}")

                    # re-construct data
                    if '_type_' in obj.attrs.keys() and obj.attrs['_type_']:
                        if   obj.attrs['_type_'] in {'None'}:
                            data[key] = None
                        if   obj.attrs['_type_'] in {'type'}:
                            data[key] = obj.dtype
                        elif obj.attrs['_type_'] in {'numpy.ndarray'}:
                            data[key] = np.array(data[key])
                        elif obj.attrs['_type_'] in {'astropy.units.Quantity'} and '_unit_' in obj.attrs.keys():
                            data[key] = units.Quantity(value=data[key], unit=obj.attrs['_unit_'])
                        elif is_verbose(verbose, 'err'):
                            say('err', None, verbose, f"Unexpected input {dict(obj.attrs)=}")
                
            elif is_verbose(verbose, 'err'):
                say('err', None, verbose, f"Unexpected input type {type(obj)=}")


    return data



def hdf5_open(
    filename: str,
    filemode: str = 'a',
    metadata: None|dict = None,
    compress: None|bool = None,
    verbose : int = 3,
) -> h5py.File:
    """Open a hdf5 file.

    Remember to close it with the .close() function!
    Alternatively you can put this in a with group.

    You can write to sub groups within one file by running
        hdf5_dump(obj, fp.create_group([group_name]))


    Parameters
    ----------
    compress: None | bool | 'gzip'
        if the file is compressed.
        if None, will guess from file name.
        if is True, will use 'gzip'.
        Will do nothing if fp is not of type str.
    """
    # compression
    if compress is None and isinstance(filename, str):
        compress = get_compress_mode_from_filename(filename)
    if compress:
        if   filemode in {'r'}:
            filename = gzip.open(filename, f'{filemode[0]}b')
        elif filemode in {'a'}:
            # decompress whole file before writing
            filename_root, ext = os.path.splitext(filename)
            if ext not in {'.gz'} and is_verbose(verbose, 'fatal'):
                raise ValueError(f"{filename=} should end with extension '.gz'.")
            try:
                with gzip.open(filename, 'rb') as f_in, open(filename_root, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            except FileNotFoundError:
                pass
            filename = filename_root
            if is_verbose(verbose, 'note'):
                say('note', None, verbose, f"Remember to manually compress the file, or use hdf5_close()")
        elif filemode in {'w', 'x'}:
            filename_root, ext = os.path.splitext(filename)
            if ext in {'.gz'}:
                filename = filename_root
            if is_verbose(verbose, 'note'):
                say('note', None, verbose, f"Remember to manually compress the file, or use hdf5_close()")
        elif is_verbose(verbose, 'fatal'):
            raise ValueError(f"Unrecognized {filemode=}")
    
    
    fp = h5py.File(filename, mode=filemode)
    if metadata is not None:
        _hdf5_dump_sub({}, fp, metadata, add_metadata=True, verbose=verbose)
    return fp





def hdf5_close(
    fp      : h5py.File,
    compress: None|bool = None,
    verbose : int = 3,
) -> None:
    fp.close()
    if compress is None or compress:
        raise NotImplementedError("Compression in this func not yet implemented")





def hdf5_subgroup(
    fp       : h5py.File | h5py.Group,
    grp_name : str,
    metadata : None|dict = None,
    overwrite: bool= False,
    verbose  : int = 3,
) -> h5py.Group:
    """Create / get a subgroup from fp.
    
    Remember to set overwrite=True at the dump-level.
    """
    
    if overwrite and grp_name in fp.keys():
        del fp[grp_name]
        
    fp_subgrp = fp[grp_name] if grp_name in fp.keys() else fp.create_group(grp_name)
    
    if metadata is not None:
        _hdf5_dump_sub({}, fp_subgrp, metadata, add_metadata=True, verbose=verbose)
        
    return fp_subgrp



def hdf5_dump(
    obj     : dict,
    fp      : str | h5py.File | h5py.Group | io.BufferedReader,
    metadata: None| dict = None,
    compress: None| bool | str = None,
    verbose : int = 3,
) -> None:
    """Dump obj to file-like fp as a hdf5 file in my custom format with support of numpy arrays etc.

    *** WILL OVERWRITE EXISTING FILES ***

    Suitable for storing medium/large machine-readable files.

    Do NOT put weird characters like '/' in obj.keys().

    DO NOT INCLUDE THE FOLLOWING KEYWORDS IN INPUT UNLESS YOU KNOW WHAT YOU ARE DOING
        '_meta_' : # meta data
            Note: You can add additional metadata for each datasets, in format of e.g.
            data = {
                'x1': ...,
                'x2': ...,
                ...,
                '_meta_': {
                    'x1': { 'Description': "Description of x1.", },
                    'x2': { 'Description': "Description of x2.", },
                },
            }
        '_data_' : # actual data
        '_type_' : # type of the data stored
            Supported type:
                 None|False  (or other False-equivalent things): return '_data_' as is
                'None'     : None.
                'str'      : Sting
                'dict'     : dict
                'np.bool_' : stored as bool (Will NOT read back as np.bool_ !)
                'tuple': tuple stored as list
                'numpy.ndarray': numpy array stored as list by default
                'astropy.units.Quantity': astropy Quantity stored as list (value) and string (unit)
        '_unit_' : unit of the astropy.units.Quantity, if that is the type
        '_misc_' : small pieces of data
        

    Parameters
    ----------
    obj: dict
        data to be written.

    fp: str | h5py.File | h5py.Group | io.BufferedReader:
        Binary file object with write permission.
        
    metadata: dict | None
        meta data to be added to file. The code will also save some of its own metadata.
        set it to None to disable this feature.

    compress: None | bool | 'gzip'
        if the file is compressed.
        if None, will guess from file name.
        if is True, will use 'gzip'.
        Will do nothing if fp is not of type str.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.
    """
    if metadata is None:
        metadata = {}
    if isinstance(fp, str):
        
        # init
        if compress is None:
            compress = get_compress_mode_from_filename(fp)
        filename_root, ext = os.path.splitext(fp)
        if not compress or ext not in {'.gz'}:
            filename_root = fp
        if is_verbose(verbose, 'note'):
            say('note', None, verbose,
                f"Writing to {filename_root}  (will OVERWRITE if file already exist.; {compress=})")
        # open & dump
        with h5py.File(filename_root, mode='w') as f:
            _hdf5_dump_sub(obj, f, metadata, add_metadata=True, verbose=verbose)
            
        # compress
        if compress:
            #if compress in {'gzip'}:
            if is_verbose(verbose, 'note'):
                say('note', None, verbose,
                    f"Compressing and saving to {filename_root}.gz;",
                    f"Deleting {filename_root}",
                )
            with open(filename_root, 'rb') as f_in, gzip.open(f"{filename_root}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(filename_root)
            #else:
            #    if is_verbose(verbose, 'warn'):
            #        say('warn', None, verbose, f"Unrecognized compress mode {compress=}, will read as if compress=False")

                
    elif isinstance(fp, h5py.Group):
        _hdf5_dump_sub(obj, fp, metadata, add_metadata=True, verbose=verbose)
    elif is_verbose(verbose, 'fatal'):
        raise TypeError(f"Unexpected input fp type {type(fp)=}")
    return





def hdf5_load(
    fp      : str | h5py.File | h5py.Group | io.BufferedReader,
    load_metadata : bool = False,
    compress: None| bool | str = None,
    verbose : int = 3,
) -> None:
    """Load data from h5py file in my custom format.


    Parameters
    ----------
    fp: str | h5py.File | h5py.Group | io.BufferedReader:
        Binary File object with read permission.
        
    load_metadata : bool
        Do NOT load meta data from loaded dict.

    compress: None | bool | 'gzip'
        if the file is compressed.
        if None, will guess from file name.
        if is True, will use 'gzip'.
        Will do nothing if fp is not of type str.
        
    verbose: int
        How much erros, warnings, notes, and debug info to be print on screen.

    Returns
    -------
    obj: original data
    """
    if isinstance(fp, str):
        
        # init
        if compress is None:
            compress = get_compress_mode_from_filename(fp)
        if is_verbose(verbose, 'note'):
            say('note', None, verbose, f"Reading from {fp}  ({compress=})")

        # open & read
        do_compress = False
        if compress:
            #if compress in {'gzip'}:
            do_compress = True
            with h5py.File(gzip.open(fp, 'rb'), mode='r') as f:
                obj = _hdf5_load_sub({}, f, load_metadata=load_metadata, verbose=verbose)
            #else:
            #    if is_verbose(verbose, 'warn'):
            #        say('warn', None, verbose, f"Unrecognized compress mode {compress=}, will read as if compress=False")
        if not do_compress:
            # no compression
            with h5py.File(fp, mode='r') as f:
                obj = _hdf5_load_sub({}, f, load_metadata=load_metadata, verbose=verbose)
                
    elif isinstance(fp, h5py.Group):
        obj = _hdf5_load_sub({}, fp, load_metadata=load_metadata, verbose=verbose)
    elif is_verbose(verbose, 'fatal'):
        raise TypeError(f"Unexpected input fp type {type(fp)=}")

    return obj
