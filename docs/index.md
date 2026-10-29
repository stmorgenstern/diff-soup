# diff-soup

diff-soup is a differentiable, GPU-native JAX library for solving large
systems of ordinary differential equations and differential algebraic
equations arising from the discretization of partial differential
equations. We employ specialized numerical methods for stiff systems that
utilize sparse data structures and algorithms for efficient performance.

Features include:

- fully differentiable DAE solvers built on variable-step, variable-order
  adaptive implicit integration;
- flexible PDE domains, from structured grids for regular geometries to
  unstructured meshes for irregular ones;
- full compatibility with JAX's automatic differentiation, `jit`
  compilation, and `vmap` batching primitives;
- sparse direct linear algebra that stays on the GPU (cuDSS), with KLU,
  SuperLU_MT, and dense backends on CPU;
- optimization, data assimilation, and uncertainty quantification built on
  the differentiable solve.

diff-soup draws inspiration from [Diffrax](https://docs.kidger.site/diffrax/)
and [Equinox](https://docs.kidger.site/equinox/) in the JAX ecosystem, and
from the Julia [SciML](https://sciml.ai/) ecosystem.

## Explore the docs

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Installation**

    ---

    Install the package and set it up on CPU or GPU.

    [:octicons-arrow-right-24: Installation](installation/index.md)

-   :material-compass-outline:{ .lg .middle } **Overview**

    ---

    Understand diff-soup's essential modules for modeling, simulation,
    and optimization.

    [:octicons-arrow-right-24: Overview](overview/index.md)

-   :material-school-outline:{ .lg .middle } **Tutorials**

    ---

    Step-by-step guides to specific features and workflows.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-flask-outline:{ .lg .middle } **Worked examples**

    ---

    Textbook-like examples for real modeling workflows.

    [:octicons-arrow-right-24: Worked examples](examples/index.md)

-   :material-code-braces:{ .lg .middle } **API reference**

    ---

    The curated public surface, documented from its docstrings.

    [:octicons-arrow-right-24: API reference](reference/index.md)

</div>

## Citing diff-soup

The methods behind diff-soup are described in the preprint
[arXiv:2610.37333](https://arxiv.org/abs/2610.37333v1).

If you use diff-soup in your work, please cite:

```bibtex
@article{degnanmorgenstern2026diffsoup,
  title   = {diff-soup: {GPU}-native Differentiable Simulation and Inference for Large-scale Stiff Dynamical Systems},
  author  = {Samuel Degnan-Morgenstern and Martin Z. Bazant},
  journal = {arXiv preprint arXiv:2610.37333},
  year    = {2026},
}
```

## Acknowledgments

diff-soup builds on the following projects:

- SUNDIALS IDA and CVODE (Lawrence Livermore National Laboratory), whose
  step-size, error-weight, and solver-reuse rules the time steppers follow.
- OrdinaryDiffEq.jl (SciML), whose QNDF solver the NDF engine follows.
- SciPy, whose BDF dense output and bounded least-squares trust region
  diff-soup reproduces.
- SuiteSparse KLU (Timothy A. Davis), SuperLU_MT (Lawrence Berkeley
  National Laboratory), and NVIDIA cuDSS, the sparse direct solvers.
- JAX and Equinox, which diff-soup is built on, and Diffrax, whose
  interfaces for saving output and events diff-soup mirrors.
- BlackJAX, which runs posterior sampling.

Their licenses and notices are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/stmorgenstern/diff-soup/blob/main/THIRD_PARTY_NOTICES.md).
