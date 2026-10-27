# Linear solvers

This page provides an overview of the linear backends behind the
corrector, the automatic backend selection, and the `Layout`
device-placement policy. The model problem is the heat-equation solve
from [running a simulation](running-a-simulation.md).

Every Newton iterate solves one linear system against the factored
iteration matrix ([nonlinear solvers](nonlinear.md)). The backends
follow an analyze, factor, solve lifecycle: `analyze` records the
sparsity pattern once, `factor` builds a numeric factor from it, and
the factor then serves many cheap solves. The engines keep the factor
of the iteration matrix across steps and refactor it only when the
matrix is refreshed ([linear solve reference](../../reference/simulation/linear.md)).

## The backends

`linear=` on the solver constructor selects the backend
([ODE/DAE solvers](solvers.md)). It takes one of two forms.

- A name selects a backend at its default settings, one of `"auto"`,
  `"cudss"`, `"klu"`, `"superlu"`, or `"dense"`.
- A backend object from `ds.linear` carries settings or the arguments
  the backend needs, as in `ds.linear.KLU(ordering="colamd")` or
  `ds.linear.Krylov(ds.linear.CuDSS())`.

A name and the object built without arguments are the same backend, so
`linear="klu"` and `linear=ds.linear.KLU()` run the same solve.

The direct backends factor the iteration matrix and come in both forms.

- `"cudss"` is NVIDIA's cuDSS sparse direct solver on CUDA GPUs. It
  factors a batch of independent matrices in one call, holds its
  numeric factor across steps, and serves the reduced-precision solve
  modes of the problem's precision policy ([precision](precision.md)).
- `"klu"` is SuiteSparse KLU on the host CPU, on float64 values. A batch
  of independent matrices is factored one matrix after another. It is
  the CPU sparse solver in the released wheels.
- `"superlu"` is SuperLU_MT on the host CPU, a threaded sparse LU on
  float64 values. It is not in the wheels and comes with a
  [source build](../../installation/index.md#install-from-source).
- `"dense"` is the XLA dense LU on CPU and GPU. It holds the LU of the
  assembled matrix and batches natively under `jax.vmap`.
- `"auto"`, the default, resolves at solve time: dense for blocks of
  at most ten unknowns, otherwise cuDSS on GPU and KLU on CPU. When
  the resolved sparse backend is unavailable the solve raises with a
  build hint, and `linear="dense"` opts into the dense cost.

!!! warning "SuperLU_MT on more than one thread"
    SuperLU_MT's threaded factorization can return a wrong solution and
    report success, an upstream defect. A source build uses every core
    the process may run on unless `OMP_NUM_THREADS` caps it. Set
    `OMP_NUM_THREADS=1`, or use `"klu"`.

The composed backends take arguments, so they are objects only.

- `ds.linear.Krylov(inner)` is an iterative solver (GMRES) that uses a
  factorization from `inner` as its preconditioner, for example
  `ds.linear.Krylov(ds.linear.CuDSS())`.
- `ds.linear.BlockTridiagonal(block_size)` is a pure-JAX solver for
  block-tridiagonal matrices, such as one field on a one-dimensional
  grid with `block_size=1`.

Settings are keyword arguments of the object and keep the library's own
names. The fill-reducing ordering is `ordering=` on KLU and SuperLU_MT
and `reorder=` on cuDSS, and the pivoting threshold is
`pivot_tolerance=` on KLU and `pivot_threshold=` on SuperLU_MT. The
[linear solve reference](../../reference/simulation/linear.md#direct-backends) lists every
setting with its default.

KLU and SuperLU_MT run on the CPU only. A program that uses one of
them and is compiled for a GPU, for example on a host whose default
device is a GPU, raises a `FactorContractError` that names the platform.
Run such a program on the CPU device, with
`jax.default_device(jax.devices("cpu")[0])` or with `JAX_PLATFORMS=cpu`.

Reading batched solves and the per-backend batch rules are covered in
[batched solves](batched-solves.md).

## Status reporting

A factorization that meets a zero pivot reports `SolveStatus.SINGULAR`,
and one whose values or factor hold a non-finite number reports
`SolveStatus.NONFINITE`. Solves with such a factor return `nan`, so a
failed factorization never yields a finite answer, and the Newton
corrector treats it as a failed iterate. Each backend decides the status
from its pivots the same way on every device. cuDSS and Dense also flag
a pivot that is zero to working precision, and KLU and SuperLU_MT report
the singularity their factorization detects. The status of a single solve
is available from `diff_soup.linear.solve_checked`.

A failure inside a native solver call (an out-of-memory error of cuDSS,
a CUDA error) raises from the call that ran it. The device state of the
process may not survive such a fault, so later programs that factor or
solve in the same process can fail as well. Restart the process after
such an error.

## Device placement

`spec.build(layout=ds.Layout(...))` declares a problem's
device-placement policy, read by the linear layer.

- `reorder=` selects the fill-reducing ordering the sparse
  factorization applies before it factors, `"amd"` (approximate
  minimum degree, the default) or `"metis"` (nested dissection). The
  ordering sets how much fill the factor acquires, and nested
  dissection suits three-dimensional meshes, where fill otherwise
  dominates. A backend that cannot select an ordering refuses a
  non-default value.
- `hybrid_memory=True` spills the cuDSS factor to host memory when it
  outgrows the GPU, and `hybrid_device_memory_limit_mb=` caps the
  device share.
- `mesh=` is a device mesh from `ds.global_mesh`, with a `data` axis, a
  `model` axis, or both. Experiments of a fit are split along the `data`
  axis, and one factorization is spread over the devices of the `model`
  axis, where the cuDSS backend selects its single-process multi-GPU or
  multi-node mode from which process owns each device. Such a spread
  factorization runs in double precision only and is rebuilt at every
  solve, because held factors live on one device. The default `None`
  factors on one device ([multi-GPU
  parallelism](../inference/multi-gpu.md)). A `data` axis over several
  devices in one process takes the backends whose factors are arrays
  (`"dense"`, `BlockTridiagonal`, and `Krylov` over one of them); cuDSS,
  KLU, and SuperLU_MT keep native factor storage and are refused there,
  so run one process per device of the axis.
- `balance_steps=True` moves experiments between the devices of the
  `data` axis when one device holds several and their step counts
  differ. The default, `False`, keeps the initial assignment.

## The topology report

`ds.topology_report(problem)` summarizes the declared structure of
the iteration matrix: the unknown count and declared nonzeros, the
row degrees, the color count a sparse-Jacobian recovery needs, and
the widest rows with the variable owning each. Wide rows are what
make a sparse factorization expensive, so the report names the
variable whose equations create them.

```python
rep = ds.topology_report(problem)
rep.n, rep.nnz         # 64, 190
rep.max_row_degree     # 3
rep.color_count        # 3
```
