# FX1 hand-off: the native slot registry, SuperLU_MT and the native findings

Session FX1 (Opus 5.5, xhigh), 2026-11-08 from 22:40 EST. Brief:
`/drive2/research2/RemainingThesisWork/batjax/docs/briefs/nested/fx1-ds-native.md` (no wait; base
`opt/integrate` df55d11, with batch A and `opt/rf-tol` in). Clone
`/drive2/research2/diff_soup_wave2/fx-native`, branch `opt/fx-native`. Throwaway scripts and logs in
`.ports/` (git-excluded); the review's repros are copied into `.ports/review/`.

Commands below use `$SLOT=/drive2/research2/RemainingThesisWork/batjax-wt/slot.sh`,
`$PY=/drive2/research2/diff_soup_umbrella/envs/ds111/bin/python` and `.ports/gpu.sh <args>` (the
brief's `flock gpu.lock systemd-run ... MemoryMax=10G` line with `PYTHONPATH=<clone>`,
`JAX_ENABLE_X64=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`). CPU rows ran
`SLOT_MEM=4G $SLOT env PYTHONPATH=<clone> JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY <script>`.

## Native modules

The three extensions are built from this tree with RP6's Docker scripts (copies in `.ports/build/`,
from `rf-linear/.ports/build/`): `bash .ports/build/build_<ext>_ffi.sh .ports/build/<out>
diff_soup/_internal/<ext>_ffi`, 40 s each, and installed by rename into `diff_soup/_internal/`. The
shipped modules (`rf-store/rf-linear/final/`) are kept in `.ports/build/shipped/` and in
`.ports/baseline/` (a copy of the tree's Python with the shipped modules, for before/after rows; run
it with that directory as the working directory or the script's, never the clone root).

| module | milestone | sha256 |
|---|---|---|
| `cudss_ffi/cu12/_native_solver*.so` | 1 | `fe5009d49c54124616879ca3014af43156084f7f23d442d1f6c5429ca0a657f4` |
| `klu_ffi/_native_klu*.so` | 1 | `4473e5ae50d3ea2db3fb5a4fff4a883e59b503bc44e2e30e08028038908a70bf` |
| `superlu_ffi/_native_superlu*.so` | 1 | `2d62a8158a87b56f17102af3ec303872e808ed2a366f984377f4ad6330dfd197` |
| `superlu_ffi/_native_superlu*.so` | 2 (installed) | `f3f08ca5e77b386e9127558277bde6983044738f9564f04a408b3b901e05f4bd` |

## Milestone 1: the registry refactor (N7)

What changed:

- **Sites belong to their analysis** (`slot_registry.h`, `ForgetAnalysis`): releasing an analysis
  erases its sites; its slots hold the analysis themselves until their context is released. cuDSS's
  `contract_analysis_release` also drops the analysis's `storage_contexts` entries.
- **A site miss never trusts the latest generation** (`contract_handlers.inc`): a solve, refactor or
  invalidate takes the producing call's run-local words only when they are this very handle's,
  written in this execution (run id) into the buffer the call reads its generation from
  (`Site::result_buffer`); otherwise it reads its handle operands and the generation rule rebuilds.
  `ResolveStaticPath`, `Site::forward_of_key`, `Slot::lane_generations` and
  `CallAttrs::StaticPath()` are deleted: with the miss reading the handle they had no reader.
- **New finding, same mechanism, fixed by the buffer check:** a jitted factor-and-solve called twice
  inside one program is lowered once (JAX caches the nested function), so after XLA inlines it the two
  factor calls share one site key and one slot; the solves took the latest call's words and one of them
  solved with the other's factor, status SUCCESS, single thread (`.ports/m1/dup_sites.py cudss`:
  error 5.28 on the shipped module, 7e-16 now). KLU and SuperLU are not affected (they compare values).
- **One thread per `SolveContext`**, checked at the entry: `SolveContext.register` (which `factor` and
  `empty_factor` call) records the first thread that factors in the context and raises
  `FactorContractError` in any other; documented on `SolveContext`. `DEFAULT_CONTEXT` has no owner:
  the solver engines factor in it from every thread, and `checkpointed_map(execution="processes")`
  runs tasks in one thread per device group. A compiled program still runs from any thread (that is
  what r6 does), and its answers are right by the generation rule.
- Tests: `test_linear_dense.py::test_a_solve_context_belongs_to_the_thread_that_first_factors_in_it`
  and `test_linear_cudss.py::test_a_function_called_twice_in_one_program_solves_each_call_with_its_own_factor`
  (fails on the shipped cuDSS module, passes now); manifest and README raised by exactly +2 functions,
  +2 collected, +34 lines; `tests/suite_policy.py` passes.
- Changelog: one breaking line (the thread rule), two bugfix lines (cuDSS's shared slot, the sites).

| check | exit | reading |
|---|---:|---|
| `.ports/gpu.sh .ports/review/r6_gpu_shared_big.py`, three runs (`.ports/m1/r6*.log`) | 0, 0, 0 | fused program, two threads x 40 solves on one slot: **0 bad in each thread, worst 2.4e-16, three runs**; misses caught (`host_reads` in the fused row) 1, 1, 3. Review: 3 to 5 wrong per thread, 0.43 |
| `.ports/gpu.sh .ports/review/r5_gpu_shared.py` | 0 | fused 0 bad (2.9e-16), split 0 bad |
| `.ports/gpu.sh .ports/review/r4_gpu.py` | 0 | A1 own contexts 3.4e-16; **A2 shared slot 3.3e-16** (review 0.125); B as before (A^T 2e-16, A^H 0.66/0.76); C sites 4 to 16 with analyses 3 to 15 (the analyses stay alive, below) |
| `r1_flat.py 30 clear` (the review's r1 plus `jax.clear_caches()` and `_clear_analyses()` each round), this tree / shipped modules | 0 / 0 | KLU and SuperLU **live_sites 0, 0, 0 after 1, 10, 30 rounds** / 1, 10, 30 (analyses 0 in both) |
| the review's r1 unchanged, 30 and 100 rounds | 0 | sites = analyses = rounds, as before: the Analysis objects are never collected in that loop (below) |
| `.ports/gpu.sh .ports/m1/dup_sites.py cudss` (new finding) | 0 | 1.2e-15 and 7.0e-16; shipped module 1.2e-15 and **5.28** |
| `.ports/gpu.sh -m pytest -q -p no:cacheprovider --run-integration tests/contracts/test_linear_cudss.py` | 0 | 39 passed, 2 skipped (two processes, two devices); the static-dispatch row (`host_reads == 0`) holds |
| new cuDSS row against the shipped module (`.ports/baseline` as working directory) | 1 | fails: the second solve returns the first factor's solution |
| `$SLOT ... pytest -q tests/contracts/test_linear_dense.py -k "thread or context"` | 0 | 2 passed |

**r1 and the gate as written.** The gate says r1 with live sites flat over 30 rounds. With the review's
script unchanged the analyses themselves stay alive, so their sites do too: JAX's abstract-evaluation
cache (`jax._src.interpreters.partial_eval._cached_abstract_eval`, a weakref LRU cache that holds the
primitive's params, the `Analysis` among them) and `analyze`'s memo of the 64 most recent analyses
(`_RECENT`) keep every round's `Analysis` (`.ports/m1/who_holds.py`: alive after `del` and `gc`, held
by that cache). The review read `live_analyses` 30 as "the Analysis deleted"; it was not. With both
caches dropped each round the analyses are collected, and the sites now go with them (0 flat) where
the shipped modules keep one per round (30). Whether diff-soup should keep the `Analysis` out of
cached primitive params is a separate question, not taken up here.

## Next

Milestone 2 (SuperLU's kind check), then 3 (SuperLU_MT measurements and the fenced locks), 4 (rows
14, 15), 5 (Jetson), 6 (GPU files, suite rows, ruff).
