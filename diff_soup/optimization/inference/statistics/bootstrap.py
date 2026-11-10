"""The bootstrap of a fit over the members of the top level of its study.

``bootstrap`` reweighs the members of a lowered study's top level (its
particles, say). From one evaluation at the fit it computes every
replicate's one-step estimate and the fit's Laplace and cluster-sandwich
covariances; with ``steps`` it then runs every replicate's own trust region
from the fit, all replicates together, and reports how each one ended. Its
result gives percentile and BCa intervals of a parameter, or of any
statistic of the decoded parameters, and pointwise bands of a function on a
grid, each with a status that says whether it can be trusted; it saves to
one ``.npz`` file and loads from it. A run keeps a checkpoint of its
iterations in a directory and resumes from it; it runs in shards, one
process per shard, whose parts merge into the full result.
"""

from __future__ import annotations

import functools
from types import MappingProxyType
from typing import Any

import jax
import numpy as np

from diff_soup.optimization.inference.fitting.result import OptResult
from diff_soup.optimization.inference.statistics._bootstrap_saved import (
    _at_fit,
    _at_fit_record,
    _declaration,
    _directory,
    _positive_integer,
    _resume,
    _saved_at_fit,
    _shard_of,
)
from diff_soup.optimization.inference.statistics.bootstrap_intervals import (
    BootstrapBand,
    BootstrapInterval,
    BootstrapOneStep,
    _one_of,
)
from diff_soup.optimization.inference.statistics.bootstrap_outcomes import (
    BootstrapOutcome,
    BootstrapOutcomes,
)
from diff_soup.optimization.inference.statistics.bootstrap_result import BootstrapResult

# Smallest number of resampling clusters a cluster bootstrap can report from.
_MIN_CLUSTERS = 3


# ---------------------------------------------------------------------------
# The bootstrap of a lowered study over the members of its top level
# ---------------------------------------------------------------------------

_SCHEMES = ("bayesian", "multinomial")


def _draw_weights(
    key: Any, replicates: int, members: int, scheme: str, *, numbers: np.ndarray | None = None
) -> np.ndarray:
    """``(R, M)`` member weights: replicate ``r`` draws from ``jax.random.fold_in(key, r)``.

    ``"bayesian"``: ``M e / sum(e)`` with ``e`` independent exponentials
    (every weight positive, the weights summing to ``M``). ``"multinomial"``:
    the counts of ``M`` draws of a member with replacement. Replicate ``r``'s
    weights do not depend on the number of replicates. ``numbers`` draws the
    rows of those replicates only, in its order (a shard's).
    """
    rows = []
    chosen = range(replicates) if numbers is None else (int(number) for number in numbers)
    for replicate in chosen:
        try:
            sub = jax.random.fold_in(key, replicate)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"key= is a JAX random key, such as jax.random.key(seed); got {type(key).__name__}."
            ) from error
        if scheme == "multinomial":
            draws = np.asarray(jax.random.randint(sub, (members,), 0, members))
            rows.append(np.bincount(draws, minlength=members).astype(np.float64))
        else:
            raw = np.asarray(jax.random.exponential(sub, (members,)), dtype=np.float64)
            rows.append(members * raw / np.sum(raw))
    return np.stack(rows)


def bootstrap(
    problem: Any,
    fit: Any,
    *,
    replicates: int,
    key: Any,
    over: Any = None,
    weights: str = "bayesian",
    steps: Any = 3,
    weight_priors: bool = True,
    solver: Any = None,
    max_attempts: Any = None,
    require_converged: bool = True,
    checkpoint: Any = None,
    keep: str = "last",
    every: int = 1,
    on_conflict: str = "error",
    callbacks: Any = (),
    chunk: Any = "auto",
    shard: Any = None,
) -> BootstrapResult:
    """Bootstrap a hierarchical fit by reweighting the members of the study's top level.

    Each replicate draws one weight per member and steps from the fit's point under those
    weights. The top level needs at least three members.

    Args:
        problem: The [`LoweredStudy`][diff_soup.optimization.inference.LoweredStudy] the fit was made on.
        fit: The [`OptResult`][diff_soup.optimization.OptResult] of
            [`TrustRegion`][diff_soup.optimization.inference.TrustRegion] on `problem`.
        replicates: The number of replicates, a positive integer.
        key: The JAX random key that fixes the weights, such as `jax.random.key(0)`.
        over: The level to reweight, by name or as `1`, which must be the top level, or `None`
            for the top level.
        weights: `"bayesian"` for continuous weights that sum to the member count, or
            `"multinomial"` to resample the members with replacement.
        steps: The accepted trust-region steps each replicate takes from the fit, `0` for a
            one-step estimate, or `"refit"` to refit each replicate to convergence.
        weight_priors: Whether each member's prior and regularization terms take the member's
            weight.
        solver: The `TrustRegion` the replicates step with, without `weights=`, `nested_at=`
            or `inequalities=`, or `None` for `TrustRegion(stationarity_tol=1e-8)`.
        max_attempts: The step attempts allowed per replicate, or `None` for the solver's
            `max_steps` under `"refit"` and `4 * steps` otherwise.
        require_converged: Whether to refuse a fit that did not converge.
        checkpoint: A directory to save the run in and resume it from, made when missing, used
            when `steps` is not `0`.
        keep: `"last"` to keep only the newest checkpoint file, or `"all"` to keep every one.
        every: The number of iterations between two checkpoint files.
        on_conflict: `"error"` to refuse a checkpoint saved with other settings, or `"restart"`
            to discard it.
        callbacks: Functions, or objects with `on_step`, called with a record of each iteration.
        chunk: The number of replicates evaluated together, `"auto"`, a positive integer, or
            `None` for the lowered study as it is.
        shard: A pair `(i, n)` that runs only the replicates whose number modulo `n` is `i`, or
            `None` for every replicate.

    Returns:
        A [`BootstrapResult`][diff_soup.optimization.inference.BootstrapResult].

    Raises:
        TypeError: If `problem` is not a lowered study or `fit` is not an `OptResult`.
        ValueError: If the fit is of another study, did not converge under
            `require_converged=True`, or the top level has fewer than three members.
        ConfigurationError: If `solver` is `None` and the fit was made under `inequalities=`.
    """
    from diff_soup._errors import ConfigurationError
    from diff_soup.optimization.inference import LoweredStudy
    from diff_soup.optimization.inference.hierarchical._population import (
        PopulationCheckpoint,
        PopulationRoute,
        refuse_inequality_fit,
    )
    from diff_soup.optimization.inference.hierarchical._replicates import (
        FitProducts,
        ReplicateStepper,
        inherited_noise,
    )
    from diff_soup.optimization.inference.hierarchical.level_solvers import TrustRegion

    if not isinstance(problem, LoweredStudy):
        raise TypeError(
            "bootstrap takes the lowered study the fit was made on, ds.lower(study); got "
            f"{type(problem).__name__}. Lower the study with ds.lower(study), fit it with "
            "inf.optsolve(lowered, inf.TrustRegion()), and bootstrap that fit."
        )
    if not isinstance(fit, OptResult):
        raise TypeError(
            f"bootstrap takes the result of inf.TrustRegion on the lowered study as fit=; got "
            f"{type(fit).__name__}."
        )
    if not _positive_integer(replicates):
        raise ConfigurationError(
            f"bootstrap replicates= is a positive integer; got {replicates!r}."
        )
    if not _one_of(weights, _SCHEMES):
        raise ConfigurationError(
            f"bootstrap weights= is 'bayesian' or 'multinomial'; got {weights!r}."
        )
    refit = _one_of(steps, ("refit",))
    if not (
        refit
        or (
            isinstance(steps, (int, np.integer)) and not isinstance(steps, bool) and int(steps) >= 0
        )
    ):
        raise ConfigurationError(
            f"bootstrap steps= is a number of accepted steps, an integer of at least 0, or "
            f"'refit'; got {steps!r}."
        )
    if weight_priors is not True and weight_priors is not False:
        raise ConfigurationError(
            f"bootstrap weight_priors= is True or False; got {weight_priors!r}."
        )
    if require_converged is not True and require_converged is not False:
        raise ConfigurationError(
            f"bootstrap require_converged= is True or False; got {require_converged!r}."
        )
    if max_attempts is not None and not _positive_integer(max_attempts):
        raise ConfigurationError(
            f"bootstrap max_attempts= is None or a positive integer; got {max_attempts!r}."
        )
    if not _one_of(keep, ("last", "all")):
        raise ConfigurationError(f"bootstrap keep= is 'last' or 'all'; got {keep!r}.")
    if not _positive_integer(every):
        raise ConfigurationError(f"bootstrap every= is a positive integer; got {every!r}.")
    if not _one_of(on_conflict, ("error", "restart")):
        raise ConfigurationError(
            f"bootstrap on_conflict= is 'error' or 'restart'; got {on_conflict!r}."
        )
    directory = _directory(checkpoint)
    part = _shard_of(shard)
    if part is not None and part[1] > int(replicates):
        raise ConfigurationError(
            f"bootstrap shard={shard!r} splits {int(replicates)} replicates into {part[1]} "
            "shards; every shard holds at least one replicate, so n is at most replicates=."
        )
    if not (chunk is None or _one_of(chunk, ("auto",)) or _positive_integer(chunk)):
        raise ConfigurationError(
            f"bootstrap chunk= is 'auto', a positive integer, or None; got {chunk!r}."
        )
    try:
        iter(callbacks)
    except TypeError as error:
        raise ConfigurationError(
            f"bootstrap callbacks= is a sequence of callbacks; got {type(callbacks).__name__}."
        ) from error

    state = getattr(fit, "state", None)
    saved = (
        None
        if state is None or "fingerprint" not in state
        else str(np.asarray(state["fingerprint"]))
    )
    if saved is not None and saved != str(problem.parameter_schema.fingerprint):
        raise ValueError(
            "fit= was made on another study than problem=; bootstrap the fit with the lowered "
            "study it was made on."
        )
    if require_converged and not fit.converged:
        raise ValueError(
            f"the fit ended with status {fit.status.value!r} and did not converge, so its point "
            "is not a stationary point of its objective and every replicate would carry the "
            "fit's own remaining step. Continue the fit until it converges, or pass "
            "require_converged=False to bootstrap from it as it is."
        )
    if solver is None:
        refuse_inequality_fit(fit)
        solver = TrustRegion(stationarity_tol=1e-8)
    looped = refit or int(steps) >= 1
    if looped and max_attempts is None:
        max_attempts = int(solver.max_steps) if refit else 4 * int(steps)
    population_route = functools.partial(
        PopulationRoute, problem, fit, over=over, weight_priors=weight_priors, solver=solver
    )
    # The evaluation at the fit prepares the study's own blocks programs when it is made; a run
    # resumed from a checkpoint makes none.
    home = population_route(population=1, chunk=None, modes=())
    members = tuple(home.members[0])
    count = len(members)
    if count < _MIN_CLUSTERS:
        raise ValueError(
            f"bootstrap resamples the members of the level {home.over!r}, and it has "
            f"{count}; resampling needs at least {_MIN_CLUSTERS} members."
        )
    # The replicates and leave-one-member-out members this run holds: every one, or a shard's.
    if part is None:
        numbers = np.arange(int(replicates), dtype=np.int64)
        jackknife = np.arange(count, dtype=np.int64)
    else:
        numbers = np.arange(part[0], int(replicates), part[1], dtype=np.int64)
        jackknife = np.arange(part[0], count, part[1], dtype=np.int64)
    drawn = _draw_weights(key, int(replicates), count, weights, numbers=numbers)
    lane_members = np.asarray(home.nodes, dtype=np.int64)[:, 0]
    declaration = _declaration(
        problem,
        fit,
        key,
        scheme=str(weights),
        replicates=int(replicates),
        steps="refit" if refit else int(steps),
        max_attempts=None if not looped else int(max_attempts),
        over=str(home.over),
        weight_priors=bool(weight_priors),
        solver=solver,
    )
    if not looped:
        home.prepare(("blocks",))
        products = FitProducts(home, fit, drawn, x_scale=solver.x_scale, numbers=numbers)
        at = _at_fit(products, jackknife, lane_members, count)
        requests = tuple(home.requests)
        work = {
            "prepared": home.prepared,
            "solves": products.solves,
            "seconds": products.seconds,
        }
        outcomes = counts = None
    else:
        # The replicates, then one member per member of the top level that leaves it out.
        population = int(numbers.size + jackknife.size)
        home.resize(population)
        writer = saved = None
        if directory is not None:
            writer = PopulationCheckpoint(
                directory,
                keep=keep,
                every=int(every),
                fingerprint={
                    **declaration,
                    "shard": "none" if part is None else f"({part[0]}, {part[1]})",
                },
            )
            saved = _resume(writer, on_conflict)
        products = None
        if saved is None:
            home.prepare(("blocks",))
            products = FitProducts(
                home, fit, drawn, x_scale=solver.x_scale, keep=True, numbers=numbers
            )
            at = _at_fit(products, jackknife, lane_members, count)
        else:
            at = _saved_at_fit(saved["products"])
        route = population_route(population=population, chunk=chunk, modes=("value", "blocks"))
        if products is not None:
            # Every member starts at the fit's point, so every member starts with the rungs and
            # first steps that the evaluation there left with the member that took part in it.
            evaluated = int(products.requested[0])
            route.rungs[...] = home.rungs[evaluated]
            route.first_steps[...] = home.first_steps[evaluated]
        stepper = ReplicateStepper(
            route,
            home,
            products,
            drawn,
            solver=solver,
            steps=steps,
            max_attempts=int(max_attempts),
            noise=inherited_noise(fit),
            numbers=numbers,
            jackknife=jackknife,
            state=saved,
            fit_converged=bool(getattr(fit, "converged", False)),
        )
        if writer is not None:
            writer.begin(stepper.iteration, _at_fit_record(at))
        if saved is not None:
            stepper.rebuild()
        stepper.run(tuple(callbacks), checkpoint=writer)
        columns = stepper.columns()
        host = columns.pop("host_seconds")
        outcomes = BootstrapOutcomes(replicates=int(numbers.size), **columns)
        counts = stepper.counts()
        requests = tuple(home.requests) + tuple(route.requests)
        attempts = int(np.sum(outcomes.attempts))
        solves = {} if products is None else dict(products.solves)
        seconds = {} if products is None else dict(products.seconds)
        work = {
            "prepared": route.prepared,
            "prepared_fit": home.prepared,
            "iterations": tuple(stepper.events),
            "solves": MappingProxyType(
                {**solves, "loop": int(sum(member.solves for member in stepper.members))}
            ),
            "seconds": MappingProxyType(
                {**seconds, "loop": stepper.seconds, "loop_host": float(np.sum(host))}
            ),
            "attempts": attempts,
            "host_seconds": host,
            "host_seconds_per_attempt": float(np.sum(host)) / attempts if attempts else 0.0,
        }
        if saved is not None:
            work["resumed_from"] = int(saved["iteration"])
    work = MappingProxyType(
        {
            "requests": requests,
            "blocks": sum(1 for request in requests if request["mode"] == "blocks"),
            "value": sum(1 for request in requests if request["mode"] == "value"),
            **work,
        }
    )
    stationarity = dict(fit.info or {}).get("stationarity")
    return BootstrapResult._from_at_fit(
        at,
        weights=drawn,
        members=members,
        over=str(home.over),
        weight_scheme=str(weights),
        weight_priors=bool(weight_priors),
        steps="refit" if refit else int(steps),
        max_attempts=None if not looped else int(max_attempts),
        outcomes=outcomes,
        fit_status=str(fit.status.value),
        fit_converged=bool(fit.converged),
        fit_stationarity=None if stationarity is None else float(stationarity),
        work=work,
        parameter_schema=problem.parameter_schema,
        shard=part,
        replicate_indices=numbers,
        jackknife_indices=jackknife,
        _counts=counts,
        _declaration=declaration,
    )


# Re-exported for the import path they had before the split: `BootstrapBand`,
# `BootstrapInterval` and `BootstrapOneStep` live in `bootstrap_intervals`, `BootstrapOutcome`
# and `BootstrapOutcomes` in `bootstrap_outcomes`, and `BootstrapResult` in `bootstrap_result`.
__all__ = [
    "BootstrapBand",
    "BootstrapInterval",
    "BootstrapOneStep",
    "BootstrapOutcome",
    "BootstrapOutcomes",
    "BootstrapResult",
    "bootstrap",
]
