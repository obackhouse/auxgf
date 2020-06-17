auxgf
=====

The auxgf package implements Green's function methods for problems in molecular quantum chemistry using auxiliary space representations, written in Python.
Methods are built as extensions to the existing functionality of the PySCF package.

For more information on the auxiliary representation and the AGF2 method, please refer to our paper:

* O. J. Backhouse, M. Nusspickel and G. H. Booth, *J. Chem. Theory Comput.*, 2020, 16, 1090-1104 (https://doi.org/10.1021/acs.jctc.9b01182).

Requirements
------------

* NumPy
* SciPy
* PySCF
* MPI4Py (optional)
* TBLIS (optional)

Features
--------

* Object-oriented implementation of an auxiliary space
* Auxiliary second-order Green's function perturbation theory for restricted (RAGF2) and unrestricted (UAGF2) references
* Auxiliary second-order Møller-Plesset perbturbation theory for restricted (RAMP2) and unrestricted (UAMP2) references
* Optimised MPI Parallel, density-fitted RAGF2(None,0) calculations (OptRAGF2)
* Spin-component-scaled SCS-RAGF2 and SCS-UAGF2 calculations, along with SCS-MP2 via zeroth iteration
* Second-order algebraic diagrammatic construction for restricted (RADC2) references
* Spin-component-scaled SCS-RADC2 calculations
* Auxiliary GW calculations for restricted (RAGWA) references, with a number of self-consistency options
* Interface to many other PySCF methods

Installation
------------

 - `git clone https://github.com/obackhouse/auxgf.git`
 - `cd auxgf`
 - `python tests/test.py`

Examples
--------

The `examples` directory contains a number of examples for calculations using the auxgf package. 

One may obtain a list of descriptions for each script using `head examples/*.py --lines=1`.

