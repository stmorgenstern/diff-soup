# diff_soup performance qualification

This directory is release infrastructure, not an ad hoc timing notebook. Every
result validates against `schema.json`, records its source or wheel provenance,
captures a compatibility-sensitive environment key, synchronizes every timed
JAX call, and retains raw samples plus StableHLO fingerprints. Wall-clock
measurements never gate hosted pull requests.

## Stable commands

```bash
JAX_PLATFORMS=cpu python -m benchmarks.run \
  --suite pr-cpu \
  --output benchmark-results/pr-cpu.json

python -m benchmarks.run \
  --suite nightly-gpu \
  --output benchmark-results/nightly-gpu.json

python -m benchmarks.run \
  --suite release \
  --installed-wheel dist_release/diff_soup-*.whl \
  --output benchmark-results/release.json

python -m benchmarks.compare \
  --baseline accepted-baselines/l40s-cuda12-cudss080.json \
  --candidate benchmark-results/nightly-gpu.json
```

The installed-wheel mode creates a fresh virtual environment, installs the
wheel, copies only the benchmark driver, changes out of the checkout, and
requires `source_tree_import_check="wheel"`. It cannot accidentally import
`diff_soup` from the repository.

Reference and release runs use three new processes. Aggregate them before
comparison:

```bash
python -m benchmarks.aggregate \
  --inputs benchmark-results/run-{1,2,3}.json \
  --output benchmark-results/aggregate.json
```

## Tiers

- Pull requests run the `pr-cpu` harness smoke in the hosted `health` job of
  `.github/workflows/ci.yml`. It checks schema validity and comparator failure
  injection on a fresh install. `tests/performance_contracts` (true batch
  lowering, allowed custom-call targets, forbidden serialized targets,
  value-stable compile caching) runs with the rest of the suite on the
  cluster, never on a hosted runner.
- The `nightly-cpu` and `nightly-gpu` reference suites run on the cluster by
  hand; no self-hosted runner is registered with the repository. Compare only
  when the computed environment key exactly equals the accepted baseline key,
  and keep the JSON, comparison reports, logs, and StableHLO evidence.
- The Slurm submission scripts that drive the reference qualifications are
  maintainers' cluster infrastructure and are not part of this repository.

## Result and comparison policy

Correctness errors, missing/changed cases, schema violations, failed lowering
assertions, a forbidden custom call, retracing contracts, or any serialized
native batch call are hard failures. Timing, compile, and memory regressions
must exceed both the relative threshold and a calibrated absolute noise floor.
Excessive relative MAD is inconclusive and must be rerun; inconclusive does not
pass.

Default maximum regressions are 15% warm runtime, 20% compile time, and 15%
peak memory. The L40S policy also requires the eight-direction cuDSS `jacfwd`
and width-16 independent `vmap` paths to remain at least 4x faster than their
explicit scalar-dispatch comparators.

KLU’s supported batch is a shared sparse matrix with multiple right-hand
sides. Independent-matrix KLU batches are rejected, not serialized. A `vmap` over
independent cuDSS matrices runs as one batched contract factorization and
solve (`diff_soup.linear.factor` / `solve` on `CuDSS()`), as the
`cudss-independent-matrix-vmap-16` case measures.

## Baseline governance

Recorded results live outside this repository. `benchmarks/` holds the
harness — the runner, the comparator, the schema, the calibration policy
templates, and `example_result.json`, a placeholder-valued result the schema
contract test reads. Accepted baselines, comparison reports, and the study
drivers are archived by the maintainers and fetched into the working
directory for a comparison; no path under `benchmarks/` is one of them.

Ordinary jobs only read baselines. They never overwrite or promote them.
Calibrate floors from at least three fresh-process results:

```bash
python -m benchmarks.calibrate \
  --inputs benchmark-results/calibration-{1,2,3}.json \
  --template benchmarks/policies/2.0-default.json \
  --output benchmark-results/calibrated-policy.json
```

Promotion requires a passing candidate, a passing before/after report, the
exact environment key, a reason, and a named reviewer:

```bash
python -m benchmarks.promote \
  --candidate benchmark-results/aggregate.json \
  --comparison-report benchmark-results/comparison.json \
  --policy benchmark-results/calibrated-policy.json \
  --environment-key cpu-0123456789abcdef \
  --reason "three-process 2.0.0 calibration" \
  --reviewer USER \
  --output accepted-baselines/cpu-reference-2026-07-23.json
```

Promotion refuses dirty checkouts and existing output paths. Review the new
file and record it in the maintainers' baseline archive; the archive retains
the prior accepted record.
