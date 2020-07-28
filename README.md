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

* Object-oriented implementation of auxiliary space.
* Auxiliary second-order Green's function perturbation theory for restricted (RAGF2) and unrestricted (UAGF2) references.
* Auxiliary second-order Møller-Plesset perturbation theory for restricted (RAMP2) and unrestricted (UAMP2) references.
* Second-order algebraic diagrammatic construction for restricted (RADC2) references.
* Auxiliary GW approximation for restricted (RAGWA) references, with a number of self-consistency options.
* Optimised AGF2(None,0) codes for density-fitted restricted (OptRAGF2) and unrestricted (OptUAGF2) references.
* Spin-component scaling (SCS) support for all auxiliary methods.
* MPI parallel codes for OptRAGF2, OptUAGF2 and RADC2.
* Interface to many other PySCF methods.

Installation
------------

 - `git clone https://github.com/obackhouse/auxgf.git`
 - `cd auxgf`
 - `python tests/test.py`

Examples
--------

The `examples` directory contains a number of examples for calculations using the auxgf package. 

One may obtain a list of descriptions for each script using `head examples/*.py --lines=1`.

