#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for reading / writing intermedian files.

"""



# Init


import json
import numpy as np
from datetime import datetime
from astropy import units



CURRENT_VERSION = '0.1'



# Functions


# JSON-related read / write func
#  suitable for small human-readable files


def _json_encode(
    obj : dict,
    metadata            : dict = {},
    overwrite_obj       : bool = False,
    overwrite_obj_kwds  : bool = False,
    ignore_unknown_types: bool = False,
    iverbose            : int  = 1,
) -> dict:
    """Encode the obj to add meta data and do type convertion.

    Recursive. Note:

    1. DO NOT PUT NON-SERIALIZABLE THINGS IN LIST (NOT INPLEMENTED)! USE DICT INSTEAD.
    2. DO NOT INCLUDE THE FOLLOWING KEYWORDS IN INPUT: (they will be added by this func)
        '_meta_' : # meta data   (if top_level)
        '_data_' : # actual data (if top_level)
        '_type_' : # type of the data stored
            Supported type:
                 None  (or other False-equivalent things): return '_data_' as is
                'tuple': tuple stored as list
                'numpy.ndarray': numpy array stored as list by default
                'astropy.units.Quantity': astropy Quantity stored as list (value) and string (unit)
    
    Parameters
    ----------
    obj: dict
        data to be serialized.

    metadata: dict or None
        meta data to be added to file. The code will also save some of its own metadata.
        set it to None to disable this feature.
        
    overwrite_obj: bool
        If False, will copy the obj before modifying to avoid changing the raw data

    overwrite_obj_kwds: bool
        if to overwrite used keywords (see above) if it already exists.
        if False, may raise ValueError if used keywords already exists.

    ignore_unknown_types: bool
        If a data is not in the known list,
            replace the data with a message ("-NotImplemented-")
            instead of raising a NotImplementedError.
        
    iverbose: int
        How much warnings, notes, and debug info to be print on screen. 
        
    Returns
    -------
    obj: (as dict) serializable data
    """
    # first, make a copy
    if not overwrite_obj and isinstance(obj, dict):
        obj = obj.copy()

    # then, write metadata
    if metadata is not None:
        if isinstance(obj, dict):
            if '_meta_' in obj.keys() and obj['_meta_'] and not overwrite_obj_kwds:
                # safety check
                raise ValueError
            obj['_meta_'] = {
                '_version_myformatter_': CURRENT_VERSION,
                '_datetime_utcnow_': datetime.utcnow().isoformat(),
            }
            # note: no need to parse data since we will do it anyway in the next step
            if isinstance(metadata, dict):
                for key in metadata.keys():
                    obj['_meta_'][key] = metadata[key]
            else:
                obj['_meta_']['_data_'] = metadata
        else:
            return _json_encode(
                {'_type_': None, '_data_': obj}, metadata=metadata,
                overwrite_obj=overwrite_obj, overwrite_obj_kwds=overwrite_obj_kwds,
                ignore_unknown_types=ignore_unknown_types, iverbose=iverbose,)
    
    # now, parse regular data
    if isinstance(obj, dict):
        # safety check
        if '_type_' in obj.keys() and obj['_type_']:
            if overwrite_obj_kwds:
                del obj['_type_']
                if iverbose >= 2:
                    print("*   Note: _json_encode(...):" + \
                          "there are '_type_' keyword inside the input dict." + \
                          "The data stored there will be removed to avoid issues.")
            elif iverbose:
                print("*   Warning: _json_encode(...):" + \
                      "there are '_type_' keyword inside the input dict. These could cause issues when reading data.")
        # recursively format whatever is inside the dict
        for key in obj.keys():
            obj[key] = _json_encode(
                obj[key], metadata=None,
                overwrite_obj=overwrite_obj, overwrite_obj_kwds=overwrite_obj_kwds,
                ignore_unknown_types=ignore_unknown_types, iverbose=iverbose,)
    else:
        # meaning this func is being recursively called- return the obj
        if isinstance( obj, (list, str, int, float, bool, type(None),) ):
            # native types
            pass
        # custom formatting
        #  *** Add new type here! ***
        elif isinstance( obj, tuple ):
            obj = {'_type_': 'tuple', '_data_': list(obj)}
        elif type(obj) is np.ndarray :
            obj = {'_type_': 'numpy.ndarray', '_data_': obj.tolist()}
        elif type(obj) is units.Quantity :
            unit = obj.unit
            # first test if the unit is a custom-defined unit that might not be parseable
            try:
                units.Unit(unit.to_string())
            except ValueError:
                unit = unit.cgs
            obj = {
                '_type_': 'astropy.units.Quantity',
                '_data_': obj.value.tolist(),
                '_unit_': unit.to_string(),
            }
        else:
            if ignore_unknown_types:
                return "-NotImplemented-"
            else:
                raise NotImplementedError
    return obj





def _json_decode(
    obj : dict,
    overwrite_obj   : bool = False,
    remove_metadata : bool = True,
    iverbose        : int  = 1,
) -> dict:
    """Decode the obj obtained from json_load(...) to its original state.

    Recursive.

    Parameters
    ----------
    obj: dict
        data to be serialized.

    overwrite_obj: bool
        If False, will copy the obj before modifying to avoid changing the raw data
        
    remove_metadata: bool
        Remove meta data from loaded dict (top level only).
        
    iverbose: int
        How much warnings, notes, and debug info to be print on screen. 
        
    Returns
    -------
    obj: original data
    """


    if isinstance(obj, dict):
        
        # first, make a copy
        if not overwrite_obj and isinstance(obj, dict):
            obj = obj.copy()
    
        # then, remove metadata
        if remove_metadata and isinstance(obj, dict) and '_meta_' in obj.keys():
            del obj['_meta_']
    
        # parse back to original data type
        if '_type_' in obj.keys():

            if not obj['_type_']:    # None
                if '_data_' in obj.keys():
                    return _json_decode(
                        obj['_data_'],
                        overwrite_obj=overwrite_obj,
                        remove_metadata=False, iverbose=iverbose)
            elif obj['_type_'] == 'tuple':
                if '_data_' in obj.keys():
                    return tuple(obj['_data_'])
            elif obj['_type_'] == 'numpy.ndarray':
                if '_data_' in obj.keys():
                    return np.array(obj['_data_'])
            elif obj['_type_'] == 'astropy.units.Quantity':
                if '_data_' in obj.keys() and '_unit_' in obj.keys():
                    return units.Quantity(value=obj['_data_'], unit=obj['_unit_'], copy=(not overwrite_obj))
            else:
                if iverbose:
                    print("*   Warning: _json_decode(...):" + \
                          f"Unrecognized obj['_type_']= {obj['_type_']}" + \
                          "type convertion for this is cancelled."
                         )
                    
            if iverbose:
                print("*   Warning: _json_decode(...):" + \
                      "Found '_type_' keyword, but read failed." + \
                      "This could imply save file corruption." + \
                      " obj['_type_'] data ignored."
                     )
        for key in obj.keys():
            obj[key] = _json_decode(
                obj[key],
                overwrite_obj=overwrite_obj,
                remove_metadata=False, iverbose=iverbose)

    return obj





def json_dump(
    obj: dict, fp,
    metadata: dict = {},
    overwrite_obj = False,
    overwrite_obj_kwds = False,
    ignore_unknown_types: bool = False,
    indent=4,
    iverbose: int = 1,
):
    """Dump obj to file-like fp as a json file in my custom format with support of numpy arrays etc.

    Suitable for storing small human-readable files.


    Parameters
    ----------
    obj: dict
        data to be serialized.

    fp:
        File object you get with open().
        
    metadata: dict or None
        meta data to be added to file. The code will also save some of its own metadata.
        set it to None to disable this feature.
        
    overwrite_obj: bool
        If False, will copy the obj before modifying to avoid changing the raw data

    overwrite_obj_kwds: bool
        if to overwrite used keywords (see above) if it already exists.
        if False, may raise ValueError if used keywords already exists.
        
    ignore_unknown_types: bool
        If a data is not in the known list,
            replace the data with a message ("-NotImplemented-")
            instead of raising a NotImplementedError.
        
    indent: int
        indentation in the saved json files.
        
    iverbose: int
        How much warnings, notes, and debug info to be print on screen.
    """
    obj = _json_encode(
        obj, metadata=metadata,
        overwrite_obj=overwrite_obj, overwrite_obj_kwds=overwrite_obj_kwds,
        ignore_unknown_types=ignore_unknown_types, iverbose=iverbose,)
    return json.dump( obj, fp, indent=indent, )



def json_load(
    fp,
    remove_metadata: bool = True,
    iverbose: int = 1,
):
    """Read obj from a json file (saved by json_dump(...) in this submodule).

        
    remove_metadata: bool
        remove meta data from loaded dict.
        
    iverbose: int
        How much warnings, notes, and debug info to be print on screen.
    """
    return _json_decode( json.load(fp), overwrite_obj=True, remove_metadata=True, iverbose=iverbose, )

