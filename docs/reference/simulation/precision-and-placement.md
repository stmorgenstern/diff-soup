# Precision and placement

`Precision` is the dtype policy of a solve, and `Layout` places
experiments and factorizations on the devices of a mesh. See
[precision](../../overview/simulation/precision.md) and [multi-GPU
parallelism](../../overview/inference/multi-gpu.md).

::: diff_soup.Precision
    options:
      members:
        - fp64
        - fp32
        - fp32_factor
        - fp32_ir

::: diff_soup.Layout
    options:
      members: false

::: diff_soup.global_mesh

::: diff_soup.mesh_groups

---

### Multi-process execution

`ds.distributed.run` splits a study's experiments over processes, and
`ds.distributed.initialize` starts the runtime for a `model` axis across them.

::: diff_soup.distributed.run

!!! warning "Launch through MPI"
    `ds.distributed.run` splits a study only under an MPI launch, such as
    `mpirun -n K python -m mpi4py script.py`. In a run started for
    `ds.distributed.initialize()` the study is not split and every process
    runs all of it.

::: diff_soup.distributed.initialize

::: diff_soup.distributed.configure

::: diff_soup.distributed.is_primary

::: diff_soup.distributed.owns

::: diff_soup.distributed.local_value

::: diff_soup.distributed.from_local_data

::: diff_soup.distributed.key

::: diff_soup.distributed.any_process
