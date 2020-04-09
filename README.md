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

Installation
------------

 - `git clone https://github.com/obackhouse/auxgf.git`
 - `cd auxgf`
 - `python tests/test.py`

Examples
--------

The `examples` directory contains a number of examples for calculations using the auxgf package. 

One may obtain a list of descriptions for each script using `head examples/*.py --lines=1`.

