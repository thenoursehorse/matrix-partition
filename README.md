# matrix-partition

Automatically partition a matrix into a block diagonal form with respect to 
symmetries.

Dependencies
-------------
* numpy
* scipy
* (Optional) networkx, which makes the merge list sort faster.

Installation
---------------
```
python -m pip install --upgrade setuptools
```


To do
---------------
1. docstrings for member functions.
1. The connections, sectors, and permutations need to be stored for each 
symmetry that is applied.
1. Implement function to rotate a state into the final permutation and basis 
after applying symmetry.
1. Other useful types of plots.

Examples
---------------
* Ising chain into parity sectors.
* XXZ spin-1/2 chain into SU(2), and parity sectors.

References
---------------
1. The automatic partitioning is inspired from 
[TRIQS](https://triqs.github.io)
in the reference 
[P. Seth, I. Krivenko, M. Ferrero, and O. Parcollet, Comp. Phys. Comm. 200, 274–284 (2016)](http://dx.doi.org/10.1016/j.cpc.2015.10.023).

Sections not under GPL-3.0
---------------
1. The algorithms to merge lists that share common elements are from:
<https://stackoverflow.com/a/4843408> by user Jochen Ritzel
and
<https://stackoverflow.com/a/4842897> by user Howard.

1. Hinton plot is copied from 
<https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html>
and the [QuTip](https://qutip.org/).