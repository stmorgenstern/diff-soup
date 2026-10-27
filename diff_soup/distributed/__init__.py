"""Multi-process execution: initialization, global meshes, replicas, and the launcher.

Two routes run a script over several processes. The lane route lowers a study
over an MPI world (``mpirun -n K python -m mpi4py script.py``): write it as
``with ds.distributed.run(study) as lowered:``, whose block runs on process 0
while the other processes serve the study's evaluations and exit when the
block ends. The global-mesh route calls ``ds.distributed.initialize()`` first
in every process, each of which runs the whole script over a mesh of every
process's devices; start it with ``python -m diff_soup.distributed.launch`` or
any launcher ``initialize`` detects.
"""

from diff_soup._distributed import (
    any_process,
    assert_decision,
    assert_decision_value,
    configure,
    from_local_data,
    global_mesh,
    initialize,
    is_primary,
    key,
    local_value,
    mesh_groups,
    owns,
    run,
)

__all__ = [
    "any_process",
    "assert_decision",
    "assert_decision_value",
    "configure",
    "from_local_data",
    "global_mesh",
    "initialize",
    "is_primary",
    "key",
    "local_value",
    "mesh_groups",
    "owns",
    "run",
]
