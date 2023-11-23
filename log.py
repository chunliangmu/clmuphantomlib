#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A sub-module for warning / error reporting / messaging things.

Owner: Chunliang Mu
"""



def _is_verbose(verbose: int|bool, verbose_req: int|None) -> bool:
    """Test if we should be verbose."""
    if verbose_req is None or isinstance(verbose, bool):
        return verbose
    else:
        return verbose >= verbose_req



def error(
    orig: str,
    verbose: int|bool,
    *msgs: str,
    verbose_req: int|None = None,
):
    """Show an error message.

    Parameters
    ----------
    orig: str
        Origins of this message (typically function name).

    verbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        * Note: Input type maybe exppand to accepting (int, file stream) as well in the future,
            to output to log files.

    msgs: str
        The messages to put up.

    verbose_req: int
        Required minimum verbose to do anything.
        If None, will treat verbose as a bool and print even if verbose < 0 !
    """
    if _is_verbose(verbose, verbose_req):
        msgs_txt = '\n\t'.join(msgs)
        print(f"*** Error  :    {orig}:\n\t{msgs_txt}")
    return




def warn(
    orig: str,
    verbose: int|bool,
    *msgs: str,
    verbose_req: int|None = 2,
):
    """Show a warning message.

    Parameters
    ----------
    orig: str
        Origins of this message (typically function name).

    verbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        * Note: Input type maybe exppand to accepting (int, file stream) as well in the future,
            to output to log files.

    msgs: str
        The messages to put up.

    verbose_min_for_showing: int
        Required minimum verbose to do anything.
        If None, will treat verbose as a bool and print even if verbose < 0 !
    """
    if _is_verbose(verbose, verbose_req):
        msgs_txt = '\n\t'.join(msgs)
        print(f"**  Warning:    {orig}:\n\t{msgs_txt}")
    return




def note(
    orig: str,
    verbose: int|bool,
    *msgs: str,
    verbose_req: int|None = 3,
):
    """Show a note message.

    Parameters
    ----------
    orig: str
        Origins of this message (typically function name).

    verbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        * Note: Input type maybe exppand to accepting (int, file stream) as well in the future,
            to output to log files.

    msgs: str
        The messages to put up.

    verbose_min_for_showing: int
        Required minimum verbose to do anything.
    """
    if _is_verbose(verbose, verbose_req):
        msgs_txt = '\n\t'.join(msgs)
        print(f"*   Note   :    {orig}:\n\t{msgs_txt}")
    return




def debug_info(
    orig: str,
    verbose: int|bool,
    *msgs: str,
    verbose_req: int|None = 4,
):
    """Show a debug info message.

    Parameters
    ----------
    orig: str
        Origins of this message (typically function name).

    verbose: int
        How much errors, warnings, notes, and debug info to be print on screen.
        * Note: Input type maybe exppand to accepting (int, file stream) as well in the future,
            to output to log files.
        
    msgs: str
        The messages to put up.

    verbose_min_for_showing: int
        Required minimum verbose to do anything.
    """
    if _is_verbose(verbose, verbose_req):
        msgs_txt = '\n\t'.join(msgs)
        print(f"Debug Info :    {orig}:\n\t{msgs_txt}")
    return