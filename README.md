# diff-soup: differentiable, sparse-oriented unified PDE suite

<p align="left">
  <a href="https://github.com/stmorgenstern/diff-soup/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg" alt="License: BSD 3-Clause"></a>
  <a href="https://stmorgenstern.github.io/diff-soup/"><img src="https://img.shields.io/badge/Documentation-online-blue.svg" alt="Docs"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?style=flat" alt="arXiv"></a>
</p>

diff-soup is a software package written in JAX for modeling, simulation, and inference of (stiff) dynamical systems governed by ODEs, DAEs, and PDEs.

For modeling:

- a Recipe API that states equations in math-like notation on structured and unstructured domains;
- finite difference, finite volume and finite element discretizations;
- sparse Jacobians assembled by automatic differentiation with graph coloring.

For simulation:

- three variable-step, variable-order stiff integrators written in JAX, `JNDF`, `JBDF` and `JIDA`, for ODEs and DAEs up to index one;
- sparse direct solves with NVIDIA cuDSS on the GPU, and KLU or dense solves on the CPU;
- forward and reverse mode differentiation of every solve, to second order;
- batched solves, with data and model parallelism across GPUs and cluster nodes.

For inference:

- parameter estimation using least squares estimation, maximum likelihood estimation (MLE), and maximum a posteriori (MAP) estimation;
- physically-constrained deep learning using differentiable simulation, with neural networks embedded inside the physical model.

![The three modules of diff-soup](https://raw.githubusercontent.com/stmorgenstern/diff-soup/main/.github/assets/software_overview.png)

The modeling, simulation and inference modules of diff-soup, and the JAX and GPU features they are built on.

## Installation

```bash
pip install "diff-soup[cu13]"   # CUDA 13, driver 580 or newer
pip install "diff-soup[cu12]"   # CUDA 12, driver 525 or newer
pip install diff-soup           # CPU only
```

Requires Linux x86_64 and Python 3.12 or 3.13. The GPU backend needs an NVIDIA GPU of compute capability 8.0 or newer.

Turn on JAX's 64-bit mode with `jax.config.update("jax_enable_x64", True)` before building a problem.

Building from source is covered on the [installation page](https://stmorgenstern.github.io/diff-soup/installation/).

## Documentation

Available at [https://stmorgenstern.github.io/diff-soup/](https://stmorgenstern.github.io/diff-soup/).

## Quick example

The two-dimensional Brusselator, a stiff reaction-diffusion system, on a 32 by 32 finite-volume grid with its walls held at the reaction steady state:

```python
import diff_soup as ds
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

n = 32
A, B, alpha = 1.0, 3.0, 1.0 / 50.0

spec = ds.problem("brusselator")
spec.domain("square", ds.domains.cartesian_grid_2d(n, n, 1.0, 1.0, method="fvm"))
spec.field("u", mass="differential",
           init=lambda x, y: A + jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y))
spec.field("v", mass="differential", init=lambda x, y: jnp.full_like(x, B / A))

@spec.equations(writes=("u", "v"), reads=("u", "v"))
def brusselator(q):
    uuv = q.u**2 * q.v
    du = A + uuv - (B + 1.0) * q.u + alpha * q.ops.laplacian(q.u)
    dv = B * q.u - uuv + alpha * q.ops.laplacian(q.v)
    return {"u": du, "v": dv}

i, j = jnp.indices((n, n))
wall = ((i == 0) | (i == n - 1) | (j == 0) | (j == n - 1)).ravel()
spec.bc.dirichlet("u", on=wall, value=A)
spec.bc.dirichlet("v", on=wall, value=B / A)

problem = spec.build()
y0 = spec.assemble_state({})

sol = ds.pdesolve(problem, ds.JIDA(), (0.0, 10.0), y0)
u_end = sol.ys[-1][: n * n]
print(float(u_end.mean()))   # the average of u over the square at t = 10
```

![The Brusselator solution](https://raw.githubusercontent.com/stmorgenstern/diff-soup/main/.github/assets/brusselator_2d.gif)

u and v of the same model on a 128 by 128 grid, from t = 0 to t = 10.

## Acknowledgments

diff-soup builds on the following projects.

**Time stepping**  
[SUNDIALS](https://computing.llnl.gov/projects/sundials): IDA and CVODE, whose step-size, error-weight and solver-reuse rules the integrators follow (Hindmarsh et al. 2005).  
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl): the QNDF method the NDF integrator follows (Rackauckas and Nie 2017).  
[SciPy](https://scipy.org): BDF dense output and the bounded least-squares trust region.  

**Sparse direct solvers**  
[SuiteSparse KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse): the CPU sparse LU (Davis and Palamadai Natarajan 2010).  
[SuperLU_MT](https://github.com/xiaoyeli/superlu_mt): the threaded CPU sparse LU for source builds (Li 2005).  
[NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/): the GPU sparse direct solver.  

**JAX ecosystem**  
[JAX](https://github.com/jax-ml/jax): the array library and compiler diff-soup is written in.  
[Equinox](https://github.com/patrick-kidger/equinox): modules and PyTree tools.  
[Diffrax](https://github.com/patrick-kidger/diffrax): the interfaces for saving output and events, which diff-soup mirrors.  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): shape and dtype annotations.  
[Optax](https://github.com/google-deepmind/optax): first-order optimisers for the embedded neural networks.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): posterior sampling.  

diff-soup is released under the BSD 3-Clause License, see [LICENSE](https://github.com/stmorgenstern/diff-soup/blob/main/LICENSE). Third-party licences and notices are listed in [THIRD_PARTY_NOTICES.md](https://github.com/stmorgenstern/diff-soup/blob/main/THIRD_PARTY_NOTICES.md).

## Citation

If you use diff-soup, please cite the paper (preprint forthcoming).

```bibtex
@article{degnanmorgenstern2026diffsoup,
    title={diff-soup: {GPU}-native Differentiable Simulation and Inference for Large-scale Stiff Dynamical Systems},
    author={Samuel Degnan-Morgenstern and Martin Z. Bazant},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2026},
}
```
