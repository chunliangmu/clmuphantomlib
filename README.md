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


----------------


## Disclaimer

Please note that this library is still under development.
I am putting the code online in the hope that it would be helpful; but I do NOT guarantee it will work.
Use it at your own risk!
I hope it helps.

**Note: Please cite the sarracen paper if you use this code (see below link for the sarracen repository description), since this code uses sarracen behind the scene.**


----------------


## Dependencies

- Python libraries:
	- `python` (version >= 3.10)
	- `numpy scipy astropy numba matplotlib ipympl moviepy`
	- `sarracen`

(
In the future I might add h5py as well.
I think that's all.
If that doesn't work, try install all of those with anaconda:
`numpy scipy astropy sympy h5py numba pandas seaborn matplotlib ipympl ipynbname pylint moviepy jupyter jupyterlab`
and then install `sarracen` with pip.
)


----------------


## Externel files

- .gitignore
	- obtained from https://github.com/github/gitignore/blob/main/Python.gitignore under CC0-1.0 license.


----------------


## Useful links

- Phantom
	- GitHub: https://github.com/danieljprice/phantom

- Sarracen
	- GitHub: https://github.com/ttricco/sarracen
