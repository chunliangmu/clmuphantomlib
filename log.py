#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for warning / error reporting / messaging things.

Owner: Chunliang Mu
"""



def _is_verbose(iverbose: int, iverbose_req: {int, None}) -> bool:
    """Test if we should be verbose."""
    return (iverbose_req is None and iverbose) or iverbose >= iverbose_req



def error(
    iverbose: int,
    orig: str,
    msg : str,
    iverbose_req: int = None,
):
    """Show an error message.

    Parameters
    ----------
    iverbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        
    orig: str
        Origins of this message (typically function name).

    msg : str
        The message to put up.

    iverbose_req: int
        Required minimum iverbose to do anything.
        If None, will treat iverbose as a bool and print even if iverbose < 0 !
    """
    if _is_verbose(iverbose, iverbose_req):
        print(f"*** Error  :    {orig}:\n\t{msg}")
    return




def warn(
    iverbose: int,
    orig: str,
    msg : str,
    iverbose_req: int = 2,
):
    """Show a warning message.

    Parameters
    ----------
    iverbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        
    orig: str
        Origins of this message (typically function name).

    msg : str
        The message to put up.

    iverbose_min_for_showing: int
        Required minimum iverbose to do anything.
        If None, will treat iverbose as a bool and print even if iverbose < 0 !
    """
    if _is_verbose(iverbose, iverbose_req):
        print(f"**  Warning:    {orig}:\n\t{msg}")
    return




def note(
    iverbose: int,
    orig: str,
    msg : str,
    iverbose_req: int = 3,
):
    """Show a note message.

    Parameters
    ----------
    iverbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        
    orig: str
        Origins of this message (typically function name).

    msg : str
        The message to put up.

    iverbose_min_for_showing: int
        Required minimum iverbose to do anything.
    """
    if _is_verbose(iverbose, iverbose_req):
        print(f"*   Note   :    {orig}:\n\t{msg}")
    return




def debug_info(
    iverbose: int,
    orig: str,
    msg : str,
    iverbose_req: int = 4,
):
    """Show a debug info message.

    Parameters
    ----------
    iverbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        
    orig: str
        Origins of this message (typically function name).

    msg : str
        The message to put up.

    iverbose_min_for_showing: int
        Required minimum iverbose to do anything.
    """
    if _is_verbose(iverbose, iverbose_req):
        print(f"Debug Info :    {orig}:\n\t{msg}")
    return