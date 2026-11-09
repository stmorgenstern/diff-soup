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

## Milestone 2: SuperLU's kind check (review row 2)

`superlu_solver.cpp`, the solve: `trusted = storage != 0 && slot->trusted_run == run.run_id &&
slot->complex == complex`, KLU's check (`klu_ffi.cpp:504`). A slot of the other kind is then compared
by value (`SameValues` refuses another kind) and factored afresh from the solve's values. Changelog:
one bugfix line.

| check | exit | reading |
|---|---:|---|
| `.ports/m2/r5c_superlu_status.py` (the review's r5c), milestone 1 module (`r5c_before.log`) | 0 | SuperLU status 0, residual nan, not finite (as the review) |
| the same, milestone 2 module `f3f08ca5` (`r5c_after.log`) | 0 | **SuperLU status 0, reported residual 2.2e-16, finite, true relative residual 1.2e-16**; KLU 2.1e-16 unchanged |

## Regression rows at a2ce8a0 (milestones 1 and 2, the modules of `rf-store/fx-native/m2/`)

| command | exit | reading |
|---|---:|---|
| `SLOT_MEM=8G $SLOT env PYTHONPATH=$PWD JAX_PLATFORMS=cpu $PY -m pytest -q -p no:cacheprovider --run-integration tests/contracts/test_linear_{host,complex,dense,primitive,ad}.py` (slot's 5 threads: SuperLU on 5) | 0 | 95 passed, 21 skipped (cuDSS rows on the CPU), 41 s |
| `.ports/gpu.sh -m pytest -q -p no:cacheprovider --run-integration tests/contracts/test_linear_cudss.py` | 0 | 39 passed, 2 skipped (two processes, two devices) |
| the same, `tests/contracts/test_linear_complex.py` | 0 | 57 passed |
| the same, `tests/integration/test_native_backends.py` | 0 | 14 passed, 182 s |
| the same, `tests/contracts/test_linear_ad.py` | 0 | 13 passed |
| `tests/suite_policy.py`; `ruff check` and `ruff format --check` on `diff_soup/linear` and the two test files | 0 | passed; clean |

**Ready to merge at `a2ce8a0` (milestones 1-2)**, with the three modules of
`/drive2/research2/diff_soup_wave2/rf-store/fx-native/m2/` (`SHA256SUMS` there) installed by rename into
`diff_soup/_internal/` outside a running suite. The module sources changed in `slot_registry.h`, so all
three must travel: KLU `4473e5ae`, SuperLU `f3f08ca5`, cuDSS cu12 `fe5009d4`. The cu13 module and the
aarch64 modules are not rebuilt yet (milestone 5 rebuilds the Jetson ones).

## Milestone 3: SuperLU_MT (N11), measured; the shipped patch kept

**Decision: the shipped patch stays** (no native source change). The lighter fix (fixupL's fix plus
acquire-release spin locks, pipelining kept) gives no wrong solve in 8,000 at 5 and at 32 threads
where fixupL's fix alone gives 7 in 12,000, so pipelining's wrong answers were a memory-ordering
fault, but its factors are not bit-identical to one thread's (1,629 and 2,000 rounds of 2,000 differ
at 5 threads): pipelining applies a parent's updates in a schedule-dependent order. R17's gate asks
for both, so it fails, and the six-run row was not needed. The patch costs time on some patterns
(the random one 3.3x at 5 threads) and saves it on others (the LFP cell); recorded for the owner.

Setup (`.ports/m3/`): SuperLU_MT 1e4bc160 built four ways at the release flags with the host gcc 13.3
(`slu/variant.sh <name> <kind>`): `up` (upstream), `ship` (`patches/0001` as shipped), `fixl` (its
`util.c` hunk only, RP6's fault 1 fixed, pipelining kept), `fence` (`fixl` plus `slu/fence.py`: an
acquire load in `await` and before it in `p?gstrf_panel_bmod.c`, a release store for every column
release in `p?gstrf_thread.c`; on x86 these compile to the same moves, so what they change is the
compiler's ordering). Driver `slu_time.c` (RP6's `slu_driver.c`, the extension's exact calls, plus the
wall time of each `pdgstrf`). Patterns: `chain` (banded, n 50,000, half-bandwidth 20: a one-path
elimination tree, `dump_matrix.py chain`), `lfp50x10` (the LFP half cell of `batjax-wt/sparsity.py` at
50 volumes x 10 particles, n 51,748, 458,101 entries, `J + M/h` at its start with h = 1e-3,
`dump_lfp.py 50 10` run from the batjax checkout), `random` (n 1,500, density 0.2 %; a sparser
4,000 x 4,000 one overflows the driver's fixed L storage). Workstation (32 logical CPUs), slot queue,
machine otherwise idle at 00:30 EST; 32 threads is every logical CPU, so those rows are noisy.

**Factor time, ms, median (minimum) of 20 rounds** (`timing.sh 20 up ship fixl fence`, exit 0, `timing.log`):

| pattern | threads | up | ship | fixl | fence |
|---|---:|---:|---:|---:|---:|
| chain | 1 | 44.4 (44.1) | 44.7 (44.3) | 44.7 (44.5) | 45.1 (44.7) |
| chain | 5 | 52.7 (51.6) | 51.1 (50.4) | 52.3 (51.0) | 52.8 (51.0) |
| chain | 32 | 648 (407) | 339 (142) | 673 (603) | 660 (554) |
| lfp50x10 | 1 | 30.1 (29.8) | 31.4 (30.3) | 29.9 (29.6) | 29.9 (29.3) |
| lfp50x10 | 5 | 22.5 (21.3) | **20.0 (18.8)** | 23.7 (21.8) | 22.6 (22.3) |
| lfp50x10 | 32 | 136 (109) | **91 (74)** | 134 (111) | 144 (116) |
| random | 1 | 20.2 (20.0) | 19.9 (19.8) | 20.0 (19.9) | 20.0 (19.6) |
| random | 5 | 6.0 (6.0) | **19.9 (19.9)** | 6.1 (6.0) | 6.0 (6.0) |
| random | 32 | 12.6 (10.0) | **33.0 (29.6)** | 13.8 (10.8) | 14.1 (10.5) |

**R17's gate rounds** (`gate.sh <variant>`: 2,000 factor-and-solve rounds of the 24 x 24 system and of
a random 400 x 400 at 1, 5 and 32 threads, bits compared with one thread's; exit 0 each):

| variant | wrong at 5 / 32 (24 x 24; 400) | rounds not bit-identical to 1 thread at 5 / 32 (24 x 24; 400) |
|---|---|---|
| up | 78 / 35; 98 / 108 | 1,645 / 1,593; 2,000 / 2,000 |
| ship | 0 / 0; 0 / 0 | **0 / 0; 0 / 0** |
| fixl | 0 / 0; 0 / 3 | 1,762 / 1,597; 2,000 / 2,000 |
| fence | 0 / 0; 0 / 0 | **1,629 / 1,569; 2,000 / 2,000** |

Three more runs of the 400 x 400 (`extra.sh`, exit 0): fixl wrong 1, 0, 0 at 5 and 2, 3, 1 at 32;
fence 0 everywhere.

**Row 10, the per-thread OpenMP pools: recorded, not routed.** `drv_pool` (`slu_pool.c`, ship library,
5 threads; `pool.sh`, exit 0, `pool.log`), factor ms median (minimum): every round on one host thread /
rotating over 14 long-lived host threads with warm pools / a new host thread (a new pool) each round:
24 x 24 0.009 / 0.014 / 0.067; 400 x 400 1.15 / 1.30 / 1.38; LFP 20.3 (19.6) / 23.7 (18.4) / 33.6 (23.6).
A pool is built once per XLA thread (the review saw 14 OS threads added over 400 calls, three or four
pools of five); after that, rotating over warm pools costs 0 to 17 % at the median and nothing at the
minimum, while one worker thread would add a hand-off to every call, more than a 24 x 24 factor
takes. Routing was not built.

Docs: the stale warnings "SuperLU_MT on more than one thread ... an upstream defect ... set
`OMP_NUM_THREADS=1`" (`docs/reference/simulation/linear.md`, `docs/overview/simulation/linear.md`),
false since R17's patch, are now notes: bit-identical to one thread, and more threads are not always
faster.

## Milestone 4: rows 14 and 15

- Row 14: `perturbed_pivot_factorizations` removed from cuDSS's `Stats` and `StatsDict` (it was never
  incremented; no Python docstring or doc page named it). SuperLU's `Stats` gains `singular` (factor
  requests whose `pdgstrf`/`pzgstrf` info is 1 to n), in `stats()` and `SuperLU.native_stats`'s
  docstring, as KLU counts it.
- Row 15: `87-real` (Jetson Orin) in the default `DIFF_SOUP_CUDA_ARCHITECTURES` (and the comment);
  the installation page's source build says that the vendored SuiteSparse is x86-64 and names
  `-C cmake.define.KLU_VENDOR_DIR=<tree>` for ARM.
- Changelog: one line under Features (SuperLU's `singular`), two under Other (the removed key; the
  default architectures and the ARM line).
- Modules rebuilt from this tree: SuperLU `b6ce5203`, cuDSS cu12 `d22bfda9` (the default list now
  gives sm_80, sm_87, sm_89, sm_90 code, `cuobjdump --list-elf`); KLU unchanged (`4473e5ae`). All three
  in `/drive2/research2/diff_soup_wave2/rf-store/fx-native/m4/` with `SHA256SUMS`.

| check | exit | reading |
|---|---:|---|
| `$SLOT ... .ports/m4_stats.py` (a 3 x 3 with a zero pivot on SuperLU) | 0 | status 1, `singular` +1 |
| `SLOT_MEM=8G $SLOT ... pytest -q -p no:cacheprovider --run-integration tests/contracts/test_linear_{host,complex,dense,primitive,ad,cudss}.py tests/integration/test_native_backends.py` (CPU) | 0 | 114 passed, 57 skipped (cuDSS rows), 63 s |
| `ruff check diff_soup tests`, `ruff format --check diff_soup tests` | 0, 0 | clean; 549 files formatted |

## Milestone 5: aarch64 (R18) on the Jetson

The tree at 73676b1 (milestones 1, 2, 4) synced as REFACTOR-COMMON says (no `.git`, `.ports`, `.so`),
SuperLU_MT 1e4bc160 archived there, and the three extensions built natively by
`.ports/jetson/jetson_build.sh` (RP6's script with this tree's names, plus `cmake --install` and the
release's `patchelf` rpaths), under `flock ~/jetson.lock` and a 3.9 GB scope; then
`.ports/jetson/jetson_tests.sh`, one process per file, GPU default, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
the integration file with `-p long_timeout` (1800 s per test, as RP6 ran it). Logs copied to
`.ports/jetson/`; the synced tree and build directory on the Jetson are deleted.

| step | exit | reading |
|---|---:|---|
| SuperLU_MT; configure | 0, 0 | 28 s |
| `_native_klu`, `_native_superlu`, `_native_solver_cu13` | 0, 0, 0 | 37 s, 20 s, 82 s |
| install, patchelf, placed | 0 | kept in `/drive2/research2/diff_soup_wave2/rf-store/fx-native/jetson/` (16 MB, `SHA256SUMS`, `BUILD_INFO`): KLU `5bd75f96` (+ `klu_ffi/lib`), SuperLU `7d7ff6c3`, cuDSS cu13 `231949e7` |
| `jetson_check.py` (each module loads; a 40-unknown system, real and complex, plain and transposed) | 0 | KLU 1.4e-15, SuperLU 7.5e-16, **cuDSS 6.1e-16 on the Orin's GPU** |
| `tests/contracts/test_linear_complex.py` | 0 | 57 passed, 94 s, cgroup peak 947 MB |
| `tests/contracts/test_linear_ad.py --run-integration` | 0 | 13 passed, 124 s |
| `tests/contracts/test_linear_host.py` | 0 | 27 passed (the five-thread SuperLU row on 5 of 6 cores) |
| `tests/contracts/test_linear_dense.py` (the thread-owner row) | 0 | 13 passed |
| `tests/contracts/test_linear_cudss.py --run-integration` | 0 | 39 passed, 2 skipped (two processes, two devices); the new duplicate-site row passes on the Orin |
| `tests/integration/test_native_backends.py --run-integration -p long_timeout` | 1 | 11 passed, 3 failed: the three `diffrax` rows (`ModuleNotFoundError: No module named 'diffrax'`, not in the Jetson venv, as in RP6's R18 run); every solver row passes. 949 s, cgroup peak 2.2 GB |

## Next

Milestone 6 (GPU files on the workstation, suite rows, ruff).
