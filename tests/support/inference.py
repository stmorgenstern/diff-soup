"""Small inference problems, data and fits shared by the suite (public imports only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.optimization import inference as inf

#: The true rate of every decay data set unless a builder is told otherwise.
TRUE_RATE = 0.6
#: Sample times of a decay measurement: five frames on the unit span.
TIMES = np.linspace(0.0, 1.0, 5)


@dataclass(frozen=True)
class Decay:
    """A built decay recipe: ``u' = -rate u + source`` on a small cell grid.

    ``rate`` and ``source`` are the recipe's parameter handles (``source`` is
    ``None`` for the one-parameter model); ``spec`` is the recipe, ``model``
    its build.
    """

    spec: Any
    model: Any
    rate: Any
    source: Any = None

    @property
    def initial_state(self):
        """The recipe's assembled initial state (ones)."""
        return self.spec.assemble_state({})


def decay_recipe(
    name: str = "suite_decay",
    *,
    nodes: int = 1,
    source: bool | str = False,
    rate: float = 0.45,
    rate_bounds=(0.1, 1.2),
    rate_prior=None,
    source_prior=None,
    precision=None,
    layout=None,
) -> Decay:
    """A decay recipe with a learnable ``rate`` and, optionally, a learnable ``source``.

    ``source="inert"`` declares ``source`` and leaves it out of the equation.
    """
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(nodes, 0.0, 1.0, method="fvm"))
    rate_handle = spec.param(
        "rate", value=rate, learnable=True, bounds=rate_bounds, prior=rate_prior
    )
    source_handle = None
    if source:
        source_handle = spec.param(
            "source", value=0.0, learnable=True, bounds=(-0.25, 0.25), prior=source_prior
        )
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    if source and source != "inert":

        @spec.equation(writes="u", reads=("u",))
        def affine_decay(q):
            return -q.p.rate * q.u + q.p.source

    else:

        @spec.equation(writes="u", reads=("u",))
        def decay(q):
            return -q.p.rate * q.u

    build = {"precision": precision} if precision is not None else {}
    if layout is not None:
        build["layout"] = layout
    return Decay(spec, spec.build(**build), rate_handle, source_handle)


def decay_values(times=TIMES, rate: float = TRUE_RATE, source: float = 0.0):
    """The exact mean of ``u' = -rate u + source`` from ``u(0) = 1``."""
    times = np.asarray(times, dtype=np.float64)
    return source / rate + (1.0 - source / rate) * np.exp(-rate * times)


def solver(rtol: float = 1e-6, atol: float = 1e-8, linear: str = "dense"):
    """The stock JNDF solver of the suite's fits."""
    return ds.JNDF.stock(linear=linear, rtol=rtol, atol=atol)


def mean_measurement(
    values, times=TIMES, *, likelihood=None, name: str = "mean", weight=None, observe=None
) -> Any:
    """A measurement of the spatial mean of the state (or of ``observe(state)`` when given)."""
    keywords = {"weight": weight} if weight is not None else {}
    if weight is None:
        keywords["likelihood"] = inf.Gaussian(scale=0.02) if likelihood is None else likelihood
    return inf.Measurement(
        name,
        times=jnp.asarray(times),
        values=values if callable(values) else jnp.asarray(values),
        observe=(lambda state: jnp.mean(state)) if observe is None else observe,
        **keywords,
    )


def decay_experiment(
    decay: Decay,
    id: str = "decay",
    *,
    rate: float = TRUE_RATE,
    source: float = 0.0,
    times=TIMES,
    likelihood=None,
    weight=None,
    noise=None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    linear: str = "dense",
    **experiment: Any,
) -> Any:
    """One experiment observing the mean of ``decay`` at ``times``.

    ``noise`` (an array the size of ``times``) is added to the exact values.
    """
    values = decay_values(times, rate, source)
    if noise is not None:
        values = values + np.asarray(noise)
    return inf.Experiment(
        model=decay.model,
        solver=solver(rtol, atol, linear),
        t_span=(0.0, float(np.asarray(times)[-1])),
        initial_state=decay.initial_state,
        measurements=(mean_measurement(values, times, likelihood=likelihood, weight=weight),),
        id=id,
        **experiment,
    )


#: A fixed small perturbation pattern: data that no parameter fits exactly.
NOISE = np.asarray([0.0, 0.004, -0.004, 0.002, -0.002])
#: The owners and sources of the three-experiment decay study.
CELLS = (("cell-a", -0.05), ("cell-b", 0.0), ("cell-c", 0.05))


def decay_study(decay: Decay, *, noisy: bool = True, **experiment: Any) -> Any:
    """Three experiments sharing ``rate``, each owner with its own ``source``."""
    experiments = tuple(
        decay_experiment(
            decay,
            owner,
            source=source,
            noise=(index + 1) * NOISE if noisy else None,
            **experiment,
        )
        for index, (owner, source) in enumerate(CELLS)
    )
    return inf.Study(
        experiments,
        parameters=ds.ParameterSet(shared=(decay.rate,), varying={"cell": (decay.source,)}),
        clusters={"cell": tuple(owner for owner, _ in CELLS)},
        independent=True,
    )


#: Per experiment of the study of partial reads: its id, its cell and its true source.
READS_EXPERIMENTS = (("e1", "c1", -0.05), ("e2", "c2", 0.05), ("e3", "c2", 0.0))
#: The solver tolerances of that study: its fits then stop on the gradient, below the
#: noise of the cost (measured: twin fits agree to 6.8e-13 relative, CPU workstation
#: 2026-11-20, pytest tests/contracts/test_study_reads.py).
READS_RTOL = 1e-10
READS_ATOL = 1e-12


def reads_study(reads: str, placement: str) -> Any:
    """Three experiments: ``e1`` and ``e2`` read ``rate`` and ``source``, ``e3`` reads ``rate`` alone.

    Under ``reads="declared"`` the model of ``e3`` does not declare ``source``;
    under ``reads="all"`` it declares it and never reads it, an inert
    placeholder. ``placement="shared"`` shares ``source``; ``"level"`` gives
    each cell its own (``e1`` in ``c1``, ``e2`` and ``e3`` in ``c2``).
    """
    priors = {
        "rate_prior": inf.NormalPrior(mean=0.5, scale=0.5),
        "source_prior": inf.NormalPrior(mean=0.0, scale=0.2),
    }
    used = decay_recipe("reads_used", source=True, **priors)
    third = decay_recipe(
        "reads_inert" if reads == "all" else "reads_bare",
        source="inert" if reads == "all" else False,
        **priors,
    )
    experiments = tuple(
        decay_experiment(
            third if index == 2 else used,
            name,
            source=source,
            noise=(index + 1) * NOISE,
            rtol=READS_RTOL,
            atol=READS_ATOL,
            parameters={"cell": cell},
        )
        for index, (name, cell, source) in enumerate(READS_EXPERIMENTS)
    )
    shared = (used.rate, used.source) if placement == "shared" else (used.rate,)
    per = {} if placement == "shared" else {"cell": (used.source,)}
    hierarchy = inf.Hierarchy(levels=("cell",), shared=shared, per=per)
    return inf.Study(experiments, parameters=hierarchy, reads=reads)


def small_fit(problem, optimizer=None, **options: Any) -> Any:
    """``optsolve`` without the end-of-fit diagnostics (they are not under test here)."""
    options.setdefault("diagnostics", False)
    return inf.optsolve(problem, optimizer, **options)
