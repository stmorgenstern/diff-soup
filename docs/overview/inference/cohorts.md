# Fitting a cohort across an allocation

This page provides an overview of fitting a large hierarchical study on
several GPUs, with one process per GPU, and of checkpointing the fit so it
survives the end of an allocation. Throughout, we reuse the study of
[hierarchical fits](hierarchical.md#pose-the-joint-inference-problem).

## The whole script

```python
import diff_soup as ds

study = ds.inference.Study(experiments, parameters=hierarchy)

with ds.distributed.run(study) as lowered:
    report = lowered.prepare()
    result = ds.inference.optsolve(lowered, ds.inference.TrustRegion(), checkpoint="checkpoints")
```

## Launch on every process

The script runs as one MPI world with one process per GPU:

```bash
srun --mpi=pmi2 --kill-on-bad-exit=1 -n K --gpus-per-task=1 python -m mpi4py cohort_fit.py
```

- `ds.distributed.run(study)` lowers the study and splits its experiments
  over the processes. Process 0 runs the block, and the other processes
  evaluate their experiments for it.
- The other processes exit when the block ends, so results are written
  inside the block.
- `mpirun -n K python -m mpi4py cohort_fit.py` starts the same run outside
  Slurm. In one process the block runs as written.
- The checkpoint directory must be on a filesystem every process sees.
- `ds.inference.sample` is refused in a run of several processes, and
  `ds.inference.bootstrap` runs over every process.
- `ds.distributed.run(study, layout=ds.Layout(processes=K))` states how
  many processes the run expects, and a run launched with another count
  is refused.

## Place the experiments

Lowering places every experiment on one process before anything is
compiled, balancing the processes by a modelled cost of each experiment. A
later run can balance by the seconds each experiment took in an earlier
run, read from that run's checkpoint:

```python
with ds.distributed.run(study, measured="checkpoints") as lowered:
    report = lowered.prepare()
```

- `report.placement` lists the experiments each process holds and whether
  the costs were modelled or measured.
- A run resumed on another number of processes places its experiments
  afresh.

## Prepare and take inventory

`lowered.prepare()` compiles every program on every process before the fit
and reports what it compiled and how long each stage took.
`lowered.inventory()` describes the layout of the parameters and the data.

```python
report = lowered.prepare()
inventory = lowered.inventory()
```

- `report.records` holds one entry per compiled program, `report.stages`
  the seconds of each stage of the setup, and `report.seconds` the total.
- `lowered.inventory()` returns one JSON-serializable dict with the
  parameter columns by group, each cluster's width and experiments, and the
  residual rows by kind.

## Checkpoint, stop, and resume

`checkpoint=` on `ds.inference.optsolve` saves the fit as it runs:
`TrustRegion` after every attempt, `LevenbergMarquardt` and
`NewtonTrustRegion` after every accepted step. When the directory already
holds a checkpoint, the fit resumes from it. Under `TrustRegion` the next
attempt is the one the uninterrupted fit would have made.

```python
class StopBeforeTheLimit(ds.inference.Callback):
    def on_step(self, info):
        if info.accepted and time_is_short():
            raise ds.inference.StopFit()


result = ds.inference.optsolve(
    lowered, ds.inference.TrustRegion(),
    callbacks=(StopBeforeTheLimit(),), checkpoint="checkpoints")
```

- A callback that raises `ds.inference.StopFit` ends the fit after the
  current step with the status `STOPPED`.
- Running the same script again resumes from the newest checkpoint. Under
  `TrustRegion`, `max_steps` counts the steps of the whole fit, across
  resumes. The other optimizers restart from the saved point with a fresh
  budget.
- A checkpoint of other parameters, or of a different study, is refused.
