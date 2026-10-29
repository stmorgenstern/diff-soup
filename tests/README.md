# Test suite contract

This suite proves the library's public behaviour and a small set of load-bearing
internal contracts. It is written against this contract, not against the history of
the code: a test states one claim about one documented feature or one reusable
contract, and nothing else. Budgets are fixed up front and pinned in
`suite_manifest.json`; growth is a review decision, never a side effect.

## Tiers

One directory per tier under `tests/`. A test declares no tier: `conftest.py`
reads it from the directory the module sits in, adds the tier marker and the tier's
per-test timeout, and raises a usage error for a module collected from anywhere else.

| tier | question it answers | marker | timeout | selected |
|---|---|---|---|---|
| `public_api/` | does a documented feature work through the public imports? | `unit` | 120 s | always |
| `contracts/` | does a reusable contract hold under jit, AD, vmap, scan, dtype changes and hashing? The linear-solver contract lives here, one file per backend family. | `unit` | 120 s | always |
| `performance_contracts/` | do the structural performance contracts hold: no retrace for a value change, program text independent of dynamic values, one factorization per setup, bounded carries? | `unit` | 120 s | always |
| `multi_device/` | does a sharded or multi-process solve behave as the single-device one, and is the unsupported route refused? | `unit` | 300 s | always |
| `integration/` | does an optional native backend satisfy the same public contract? | `integration` | 180 s | `--run-integration` |
| `validation/` | does a representative physical model satisfy an independent analytical invariant? | `validation` | 300 s | `--run-validation` |

There is no `linear/` tier. Timeouts apply when `pytest-timeout` is installed.

## Budgets (ceilings, pinned in the manifest)

| scope | files | test functions | collected items | source lines |
|---|---|---|---|---|
| whole suite | 43 | 390 | 565 | 14,100 |
| `public_api` | 19 | 175 | 277 | 5,592 |
| `contracts` | 17 | 160 | 220 | 5,400 |
| `performance_contracts` | 2 | 11 | 28 | 480 |
| `multi_device` | 2 | 29 | 34 | 960 |
| `integration` | 2 | 11 | 32 | 680 |
| `validation` | 1 | 4 | 20 | 420 |

Source lines count every Python file under `tests/`, the infrastructure and
`support/` included; the tier lines count the files in the tier's directory.
`suite_policy.py` enforces files, functions and lines; `conftest.py` enforces the
collected ceilings whenever a run collects exactly the manifest's modules. The whole
suite runs on one CPU stream in under 25 minutes, and no file takes longer than
two minutes; per-file durations are recorded in the session report.

## Rules every test follows

1. **One file per public verb group.** The manifest's `feature_contracts` names the
   group; a new group is a manifest decision made in its own commit.
2. **Public behaviour only.** Tests import `diff_soup` and the namespaces the
   manifest lists under `public_namespaces` (`diff_soup.inference`,
   `diff_soup.linear`, `diff_soup.domains`, ...). Any other `diff_soup` module is
   private. A `contracts/` file whose contract is about one private module may
   import that one module, named in the manifest under `private_imports`; no other
   tier may import a private module, and `support/` may not either.
3. **One assertion per claim.** A test asserts the claim its name states. A
   module-scoped fixture may do the expensive work (one solve, one fit) that several
   small tests then read.
4. **Every numeric bound cites its measurement.** A tolerance (`atol=`, `rtol=`,
   `tol=`) or a float comparison in an `assert` carries, on its line or the lines
   just above it (or at the module constant it names), a comment with the word
   `measured`, the measured value and the run that produced it, for example
   `# measured 3.1e-9, CPU VM 2026-10-22, pytest tests/contracts/test_x.py`.
   Exact comparisons (`==`, `assert_array_equal`) and a zero bound need no citation.
5. **No private seams.** No mocks (`unittest.mock` in any import form, `mocker`),
   no `monkeypatch` or `pytest.MonkeyPatch`, no writes to private attributes or
   module globals, no wall clock (`time` in any form; a duration is measured by a
   subprocess the test spawns, never by the test), no stored baselines (`.npy`,
   `.npz`, `.csv`, or `.json` under `support/` unless listed in `support_data`), no
   source-text greps of the library, no removed-API pins, no duplicate parameter
   rows across files.
6. **Small problems, measured.** A test uses the smallest problem that exercises
   the claim; a file runs in under two minutes on one CPU stream.
7. **Hardware is gated by marker, never by a skip in the body.** A row that needs
   a device, a native backend or a process count carries `requires_cuda`,
   `requires_cudss`, `requires_klu`, `requires_superlu`, `requires_two_devices`,
   `requires_three_devices`, `requires_four_devices` or `requires_two_processes` and
   is never reported as passing on a host that lacks it.
   Such rows are counted as owed in the session report.
8. **Shipped text is written for public readers.** No internal codenames, no
   migration narrative; a test's name and docstring state the claim.

## Capability markers and strict mode

The capability markers `requires_cuda`, `requires_cudss`, `requires_klu` and
`requires_superlu` (the SuperLU_MT extension, which a default build compiles only
where a SuperLU_MT install is found) are declared in `pyproject.toml`; `conftest.py`
declares `requires_two_devices` (`jax.device_count() >= 2`),
`requires_three_devices` (`jax.device_count() >= 3`),
`requires_four_devices` (`jax.device_count() >= 4`) and
`requires_two_processes` (`jax.process_count() >= 2`).
A marked test whose capability the host lacks is skipped with the marker named in the
reason (`owed: requires_cudss ...`), so the `-rs` summary lists the owed rows.

`--strict-capabilities` is the gate's mode. Under it, a marker skip stays a skip
(the gate counts them as owed), and every other skip in a selected tier is a failure:
a `skipif` marker, a `pytest.skip()` call in a test or fixture, and a module-level
skip (`importorskip`, `pytest.skip(allow_module_level=True)`), which is a collection
error. Expected failures (`xfail`) are reported as usual. A tier is selected unless
it is an opt-in tier whose flag was not passed; those skips stay skips.

`--require-capabilities` is for a host that has every backend built. A missing build
capability (`requires_cuda`, `requires_cudss`, `requires_klu`, `requires_superlu`)
becomes a setup failure naming the marker, so a run on a host that left a backend
unbuilt cannot pass. A missing `requires_two_devices`, `requires_three_devices`,
`requires_four_devices` or `requires_two_processes` stays an owed skip under the flag: a device count or a process world belongs to the
launch, not to the build, so a one-process run lists those rows as owed and a launch
that provides them runs them.

## Running

```bash
# the gate on a CPU host, one stream
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python tests/suite_policy.py --report
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q tests \
    --run-integration --run-validation --strict-capabilities -p no:cacheprovider -rfEs --durations=0

# one file while working (the policy still runs at session start)
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python -m pytest -q tests/contracts/test_linear_dense.py -p no:cacheprovider
```

`python tests/suite_policy.py --report` prints one line per tier,
`tier: files a/c  functions a/c  lines a/c` (actual over ceiling), and a total line;
it exits non-zero exactly when the plain check does. `pytest_sessionstart` runs the
same check with one relaxation: a module the manifest lists but the tree does not yet
hold is reported, not refused, so a file can be run before the tree is complete.

`conftest.py` clears the JAX and Equinox compile caches after every module, not
after every test: a module of at most 20 small items retains well under the memory
that calls for a clear per test (a module of fifty large solves retains about eleven
gigabytes), and sharing compiled programs across a module's tests is what keeps the
suite inside its time budget. `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1` is set before
JAX is imported and x64 is enabled for the whole run.

## Where the suite runs

Hosted CI never executes a test body: it checks formatting, lint, this policy, that
the tree imports and collects, and the benchmark harness. The suite runs before a pull
request is opened or updated, on a host with the native backends built, in strict
mode with the opt-in tiers on and every build capability required:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests --run-integration --run-validation \
    --strict-capabilities --require-capabilities -p no:cacheprovider -rfEs --durations=0
```

On such a host a row fails when its claim fails or when its `requires_cuda`,
`requires_cudss`, `requires_klu` or `requires_superlu` marker is unsatisfied. On a GPU
host the CPU platform must stay enabled (`JAX_PLATFORMS=cuda,cpu`, or `JAX_PLATFORMS`
unset) so that the KLU and SuperLU rows run on the CPU device; `conftest.py` fails those
rows in setup, naming `JAX_PLATFORMS`, when the run has no CPU device. The
two-process rows, and the two-device rows on a host with one device, are owed skips
that the `-rs` summary lists: when every other row passes, the pull-request gate
reports `result: PASS` and lists those owed rows. The two-process rows run in a world of two processes whose
ranks call `ds.distributed.initialize()` before `pytest.main(...)`: under `mpirun -n 2`
with `DIFFSOUP_COORDINATOR=localhost:<port>`, and under `srun --mpi=pmi2 -n 2` with
`DIFFSOUP_MGMN=1` for the cuDSS row whose factorization spans the two processes.
