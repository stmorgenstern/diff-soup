"""A source experiment and the experiments it drives through a link (public imports only).

The source solves ``u' = -a u`` from ``u(0) = 1`` and produces its mean ``y`` at
``LINK_TIMES``. Each driven experiment solves ``z' = -b z + y(t)`` from ``z(0) = 0``,
``y`` the piecewise-linear signal of the source's output. Both recipes declare
``a`` and ``b``; each reads one of them. The study shares ``a`` and gives every
member of the level ``cell`` its own ``b``; the source sits in a cell of its own,
where ``b`` is inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.optimization import inference as inf
from diff_soup.optimization.inference.approx import PiecewiseLinear

#: The times of the source's output, the knots of every driven signal.
LINK_TIMES = np.linspace(0.0, 2.0, 21)
#: The measurement times of every experiment.
TIMES = np.linspace(0.25, 2.0, 8)
#: The true rates: ``a`` of the source, ``b`` of each driven cell.
TRUE = {"a": 0.8, "c1": 1.3, "c2": 0.9}
#: The cells of the driven experiments.
CELLS = ("c1", "c2")


@dataclass(frozen=True)
class Linked:
    """The two built recipes and the handles ``a`` and ``b`` (the source's)."""

    source_spec: Any
    source: Any
    driven_spec: Any
    driven: Any
    a: Any
    b: Any


def _recipe(name: str, driven: bool, values: Any = None) -> tuple:
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    a = spec.param("a", value=0.5, learnable=True, bounds=(0.05, 3.0))
    b = spec.param("b", value=0.5, learnable=True, bounds=(0.05, 3.0))
    if driven:
        given = np.zeros(LINK_TIMES.shape) if values is None else values
        signal = PiecewiseLinear(jnp.asarray(LINK_TIMES), jnp.asarray(given))
        spec.function("y", signal, learnable=False)
        spec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

        @spec.equation(writes="z", reads=("z",))
        def relax(q):
            return -q.p.b * q.z + q.fn.y(q.t)

    else:
        spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

        @spec.equation(writes="u", reads=("u",))
        def decay(q):
            return -q.p.a * q.u

    return spec, spec.build(), a, b


def linked_recipes(name: str = "linked") -> Linked:
    """The source and driven recipes, built."""
    source_spec, source, a, b = _recipe(f"{name}_source", False)
    driven_spec, driven, _, _ = _recipe(f"{name}_driven", True)
    return Linked(source_spec, source, driven_spec, driven, a, b)


def driven_alone(recipes: Linked, values: Any, name: str = "linked_alone") -> Any:
    """The driven recipe built with the signal's values given, without a link."""
    return _recipe(name, True, values)[1]


def source_mean(a: float, times: Any = TIMES) -> np.ndarray:
    """The source's exact mean ``exp(-a t)``."""
    return np.exp(-a * np.asarray(times, dtype=np.float64))


def driven_mean(a: float, b: float, times: Any = TIMES) -> np.ndarray:
    """The driven experiment's exact mean on the piecewise-linear signal of ``exp(-a t)``.

    On each piece the signal is ``alpha + beta s`` and ``z(s) = c0 + c1 s + (z0 - c0) e^{-b s}``
    with ``c1 = beta / b`` and ``c0 = (alpha - c1) / b``.
    """
    knots = LINK_TIMES
    values = source_mean(a, knots)
    out = []
    for t in np.asarray(times, dtype=np.float64):
        z = 0.0
        for left, right, start, end in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
            if left >= t:
                break
            beta = (end - start) / (right - left)
            span = min(right, t) - left
            c1 = beta / b
            c0 = (start - c1) / b
            z = c0 + c1 * span + (z - c0) * np.exp(-b * span)
        out.append(z)
    return np.asarray(out)


def solver(rtol: float = 1e-10, atol: float = 1e-12) -> Any:
    """The tight JNDF solver every experiment of the linked study uses."""
    return ds.JNDF.stock(linear="dense", rtol=rtol, atol=atol)


def _measured(name: str, values: Any) -> tuple:
    return (
        inf.Measurement(
            name,
            times=jnp.asarray(TIMES),
            values=jnp.asarray(values),
            observe=lambda state: jnp.mean(state),
            likelihood=inf.Gaussian(sigma=0.01),
        ),
    )


def source_experiment(recipes: Linked, *, observe: Any = None, **experiment: Any) -> Any:
    """The source experiment, its output ``y`` its mean (or ``observe``) at ``LINK_TIMES``."""
    output = inf.Output(
        "y",
        times=LINK_TIMES,
        observe=(lambda state: jnp.mean(state)) if observe is None else observe,
    )
    experiment.setdefault("outputs", (output,))
    experiment.setdefault("parameters", {"cell": "source"})
    return inf.Experiment(
        model=recipes.source,
        solver=solver(),
        t_span=(0.0, 2.0),
        initial_state=recipes.source_spec.assemble_state({}),
        measurements=_measured("x", source_mean(TRUE["a"])),
        id="source",
        **experiment,
    )


def driven_experiment(recipes: Linked, cell: str, *, model: Any = None, **experiment: Any) -> Any:
    """The driven experiment of ``cell``, linked to the source's output unless told otherwise."""
    experiment.setdefault("links", (inf.Link(source="source", output="y", into="y"),))
    experiment.setdefault("parameters", {"cell": cell})
    return inf.Experiment(
        model=recipes.driven if model is None else model,
        solver=solver(),
        t_span=(0.0, 2.0),
        initial_state=recipes.driven_spec.assemble_state({}),
        measurements=_measured("z", driven_mean(TRUE["a"], TRUE[cell])),
        id=cell,
        **experiment,
    )


def linked_study(recipes: Linked, **source: Any) -> Any:
    """The source and the two driven cells, ``a`` shared and ``b`` one per cell."""
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(recipes.a,), per={"cell": (recipes.b,)})
    experiments = (source_experiment(recipes, **source),) + tuple(
        driven_experiment(recipes, cell) for cell in CELLS
    )
    return inf.Study(experiments, parameters=hierarchy)


#: The true point in the study's columns: ``a``, the source cell's inert ``b``, then each cell's.
TRUE_POINT = np.asarray([TRUE["a"], 0.5, TRUE["c1"], TRUE["c2"]])
#: A wrong start for the fits.
START = np.asarray([0.4, 0.5, 0.6, 0.6])
