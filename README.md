# clmuphantomlib

[![DOI](https://zenodo.org/badge/756653213.svg)](https://doi.org/10.5281/zenodo.22912863)

A python library for analyzing Phantom SPH data (including lightcurves calculations), using sarracen as backend.

> Author: Chunliang Mu  
> Requrie python 3.10+ (since I am using the | operator for type hints)
> 
> This library is written by me for my PhD project **Radiative Transfer (RT) in Common Envelope Evolution (CEE)** (a.k.a. "Non-adiabatic Common Envelope Simulation of Massive Stars").  
> Creator: ***Chunliang Mu*** (PhD student at Macquarie University 2023-2026)  
> Principal Supervisor: Professor Orsola De Marco  
> Associate Supervisor: Professor Mark Wardle  

For examples for running the code, see `examples/` folder, which are directly copied from my scripts on 2024-05-10.
You may want to change the input parameters (the `*__input.py` file and the `_*.py` files) and put them into the src/ directory (or alternatively put a symbolic link in the `examples/` directory to `src/clmuphantomlib`, so the package can be loaded correctly) before using them.
See <https://github.com/chunliangmu/RTinCEE-scripts-2024> for all of my scripts using this library in 2024 so far.

**Note: Please cite the sarracen paper if you use this code (see below link for the sarracen repository description), since this code uses sarracen behind the scene.**

## Dependencies

See `requirements.txt`.

- Python libraries:
	- `python3` (version >= 3.10)
	- `numpy scipy astropy h5py numba matplotlib ipympl moviepy`
	- [`sarracen`](https://github.com/ttricco/sarracen)

## Externel files

- `.gitignore`: obtained from https://github.com/github/gitignore/blob/main/Python.gitignore under CC0-1.0 license.

## Useful links

- `phantom` [GitHub](https://github.com/danieljprice/phantom)
- `sarracen` [GitHub](https://github.com/ttricco/sarracen)
- More example scripts on [GitHub](https://github.com/chunliangmu/RTinCEE-scripts-2024)
