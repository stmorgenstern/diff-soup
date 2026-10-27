# Third-party notices

diff_soup is licensed under BSD-3-Clause (see `LICENSE`). Portions of
its algorithms and code are derived from or modeled on the following
permissively-licensed projects, reproduced here with their required
notices. Vendored binary dependencies (SuiteSparse/KLU and the GCC runtime
libraries they need) carry their own license files under
`diff_soup/_internal/klu_ffi/licenses/`.

## OrdinaryDiffEq.jl (MIT "Expat" License)

The NDF engine (`diff_soup/timestepping/vsvo/engines/ndf/`) follows the
QNDF solver of OrdinaryDiffEq.jl and its simplified-Newton layer: the
backward-difference-table recurrences (`U!`, `calc_R`, `update_D!`), the
corrector equation, its convergence law and default constants
(`nlsolve.jl`/`newton.jl`/`type.jl`), and the accept/reject step-and-order
controllers (`controllers.jl`), which OrdinaryDiffEq.jl in turn attributes
to E. Alberdi Celaya, J. J. Anza Aguirrezabala & P. Chatzipantelidis,
"Implementation of an Adaptive BDF2 Formula and Comparison with the MATLAB
Ode15s" (2014).

> Copyright (c) 2016-2020: ChrisRackauckas, Yingbo Ma, Julia Computing Inc, and
> other contributors:
>
> https://github.com/SciML/OrdinaryDiffEq.jl/graphs/contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.

## SciPy (BSD-3-Clause License)

The NDF engine's dense output
(`diff_soup/timestepping/vsvo/engines/ndf/dense.py`) evaluates the Newton
backward-difference interpolant in the cumulative-product form used by
SciPy's BDF dense output (`scipy/integrate/_ivp/bdf.py`). The engine also
adopts that module's difference-table conventions (the state as row 0 of
the table, the `h * f` first-difference seed, and the fold carried through
row 0), its minimum-step law (a small multiple of the floating-point
spacing at the current time), and its factor-of-two step cut after a
nonlinear-convergence failure.

The bounded trust region of `ds.inference.TrustRegion`
(`diff_soup/inference/hierarchical/_bounds.py`) reproduces the box-bound
handling of SciPy's `least_squares` with `method="trf"`
(`scipy/optimize/_lsq/common.py` and `trf.py`): the Coleman-Li scaling,
`make_strictly_feasible`, `intersect_trust_region`, and `select_step`, under
SciPy's names and with SciPy's semantics.

> Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above
>    copyright notice, this list of conditions and the following
>    disclaimer in the documentation and/or other materials provided
>    with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived
>    from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
> FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
> COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
> INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
> BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
> LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
> ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## SUNDIALS (BSD-3-Clause License)

Parts of the VSVO engines' step machinery derive
from SUNDIALS CVODE and IDA source expression: the automatic initial-step-size
search (the bracketed second-derivative estimate with its bias and bound
constants, `diff_soup/timestepping/vsvo/engines/bdf/dense.py`, also used
by the NDF engine), the endpoint snap/clamp rounding fuzz, the
error-weight refresh cadence, and the step-count windows of the
linear-solver reuse gates. The IDA-class engine
(`diff_soup/timestepping/vsvo/engines/ida/`) follows the conventions of
SUNDIALS IDA, developed at Lawrence Livermore National Laboratory: the `id`
mask of differential rows, the `2/(1+cjratio)` correction of a corrector
taken under a stale leading coefficient, the solution interpolation of
`ida_get_solution`, and the modes of its consistent initialization. SUNDIALS is Copyright (c) 2002-2024, Lawrence
Livermore National Security and Southern Methodist University, released
under BSD-3-Clause; its license text follows.

```text
BSD 3-Clause License

Copyright (c) 2025-2026, Lawrence Livermore National Security, University of Maryland Baltimore County, and the SUNDIALS contributors.
Copyright (c) 2013-2025, Lawrence Livermore National Security and Southern Methodist University.
Copyright (c) 2002-2013, Lawrence Livermore National Security.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SuiteSparse (KLU and BTF: LGPL-2.1-or-later; AMD, COLAMD and SuiteSparse_config: BSD-3-Clause)

The KLU extension links SuiteSparse 7.10.1 (KLU, BTF, AMD, COLAMD and
SuiteSparse_config) dynamically, as unmodified shared libraries that ship in
`diff_soup/_internal/klu_ffi/lib/` of the wheel. KLU and BTF are
LGPL-2.1-or-later; AMD, COLAMD and SuiteSparse_config are BSD-3-Clause. Their
license texts, including the full GNU Lesser General Public License version
2.1 (`libquadmath-COPYING.LIB.txt`), ship in
`diff_soup/_internal/klu_ffi/licenses/`; the per-library notices are in
`diff_soup/_internal/klu_ffi/THIRD_PARTY_NOTICES.md`.

## SuperLU_MT (BSD-3-Clause License)

The SuperLU_MT extension links SuperLU_MT 4.0.2 and the reference BLAS
routines distributed with it, statically. It is built only when a
SuperLU_MT installation is found. SuperLU_MT is Copyright (c) 2003, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory. Its license text and the exact source revision are in
`diff_soup/_internal/superlu_ffi/THIRD_PARTY_NOTICES.md`.

## GCC runtime libraries (GPL-3.0-or-later WITH GCC-exception-3.1, LGPL-2.1-or-later)

The KLU extension is compiled with GCC, and the SuiteSparse libraries it
links against need the GCC runtime at load time. A bare Python environment
does not carry those libraries, so the wheel bundles them, unmodified as
built by the GCC project, beside the vendored SuiteSparse binaries in
`diff_soup/_internal/klu_ffi/lib/`:

- `libgcc_s.so.1`, `libgfortran.so.5`, and `libgomp.so.1` are
  GPL-3.0-or-later WITH GCC-exception-3.1 — the GNU General Public License
  version 3 together with the GCC Runtime Library Exception version 3.1,
  which is the term under which these runtime libraries are redistributed
  with software built by GCC.
- `libquadmath.so.0` is LGPL-2.1-or-later — the GNU Lesser General Public
  License version 2.1.

None of the four is modified; diff_soup neither patches nor statically
absorbs them, and each remains a separate shared library in the wheel.
Their license texts ship in
`diff_soup/_internal/klu_ffi/licenses/`:

- `GCC-COPYING3.txt` — GNU General Public License, version 3
- `GCC-COPYING.RUNTIME.txt` — GCC Runtime Library Exception, version 3.1
- `libquadmath-COPYING.LIB.txt` — GNU Lesser General Public License,
  version 2.1

## Published-method references

The NDF coefficient family (the `kappa` corrections, harmonic-sum `gamma`
vector, and per-order error constants) implements L. F. Shampine &
M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1), 1997,
and R. W. Klopfenstein, "Numerical differentiation formulas for stiff
systems of ordinary differential equations", RCA Review 32, 1971.
The other published methods the library implements:

- K. E. Brenan, S. L. Campbell & L. R. Petzold, *Numerical Solution of
  Initial-Value Problems in Differential-Algebraic Equations*, SIAM, 1996
  (the BDF family of DAE integrators behind the IDA-class engine).
- E. Hairer, S. P. Nørsett & G. Wanner, *Solving Ordinary Differential
  Equations I: Nonstiff Problems*, Springer, Ch. III.1 (the
  backward-difference interpolant of the dense output).
- J. J. Moré & D. C. Sorensen, "Computing a trust region step", SIAM J. Sci.
  Stat. Comput. 4(3), 1983 (the dense trust-region subproblem).
- T. Steihaug, "The conjugate gradient method and trust regions in large
  scale optimization", SIAM J. Numer. Anal. 20(3), 1983 (the matrix-free
  trust-region subproblem).
- M. A. Branch, T. F. Coleman & Y. Li, "A subspace, interior, and conjugate
  gradient method for large-scale bound-constrained minimization problems",
  SIAM J. Sci. Comput. 21(1), 1999 (the bounded trust region).
- S. Sun & J. Nocedal, "A trust region method for noisy unconstrained
  optimization", Math. Program. 202, 2023 (the noise-relaxed gain ratio and
  the radius floor).
- G. W. Recktenwald, "The control-volume finite-difference approximation to
  the diffusion equation", 2012 (the width-weighted harmonic mean of face
  diffusivities).

Published algorithms and mathematical formulas are facts, not copied
expression; they are cited here for scholarly attribution.
