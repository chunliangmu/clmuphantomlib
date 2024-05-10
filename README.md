# clmuphantomlib

> A python library for analyzing Phantom SPH data using Sarracen


## Meta

Author: Chunliang Mu

Requrie python 3.10+ (since I am using the | operator for type hints)


This library is written by me for my PhD project:

Project20230125: **Radiative Transfer (RT) in Common Envelope Evolution (CEE)**


Creator: ***Chunliang Mu*** (PhD student at Macquarie University 2023-2026(expected))

Principal Supervisor: Professor Orsola De Marco

Associate Supervisor: Professor Mark Wardle



There are some examples in the examples/ folder for running the code.
They are directly copied from my scripts,
and you may want to change the input parameters (the `*__input.py` file and the `_*.py` files)
before using them.

Have fun!


## Disclaimer

This project is a *work in progress*.
No guarrantees whatsoever.
Use it at your own risk.

**Note: Please cite the sarracen paper if you use this code (see below link for the sarracen repository description), since this code uses sarracen behind the scene.**




## Dependencies

- Python libraries:
	- `python3` (version >= 3.10)
	- `numpy scipy astropy h5py numba matplotlib ipympl moviepy`
	- `sarracen`

(
I think that's all.
If that doesn't work, try install all of those with anaconda:
`numpy scipy astropy sympy h5py numba pandas seaborn matplotlib ipympl ipynbname pylint moviepy jupyter jupyterlab`
and then install `sarracen` with pip.
)




## Externel files

- `.gitignore`: obtained from https://github.com/github/gitignore/blob/main/Python.gitignore under CC0-1.0 license.




## Useful links

- `phantom`: GitHub https://github.com/danieljprice/phantom

- `sarracen`: GitHub https://github.com/ttricco/sarracen
