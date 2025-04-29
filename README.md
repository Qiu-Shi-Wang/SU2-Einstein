# SU2-Einstein
Computer-assisted construction of SU(2)-invariant negative Einstein metrics.

Arbitrary-precision or interval-arithmetic computations are done using [FLINT](https://python-flint.readthedocs.io/en/latest/), the main required package besides the standard Python packages.

Run the code, with the parameters set up to match appendix B of the paper, using

```
python3 su2metric.py
```

The code is contained in 4 files.

- `su2metric.py`: Main script. Contains the arbitrary-precision Taylor series solver, as well as the class `eta` of the approximate solution $\hat\eta
- `estimates_inf.py`: contains functions used for the estimates at hyperbolic infinity, implemented using interval arithmetic.
- `arb_cheby.py`: contains the class of Chebyshev polynomials `arb_chebyT` taken from [Buttsworth--Hodgkinson](https://github.com/liamhodg/O3O10), as well as a new class `T_rational` of rational functions of Chebyshev polynomials.
- `sturm.py`: contains an arbitrary-precision interval-arithmetic implementation of Sturm's theorem.
