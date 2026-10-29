"""``ds.compile``: compile a ``pdesolve`` call ahead of time without running it.

``compile`` closes over the static arguments of one solve (the problem, the
solver, the output plan, the differentiation strategy, events, the control
state, solver keywords), lowers ``pdesolve`` on the shapes and dtypes of the
traced ones (``t_span``, ``y0``, ``theta``, ``lambda0`` and ``SaveAt.ts``) and
compiles it. The returned ``CompiledSolve`` runs the executable on new values of
the same shapes and dtypes. Lowering traces the solve and compiling builds the
executable; neither evaluates a residual.

A ``Solution`` holds non-array members (the solver view, layout records), which
``jax.jit`` cannot return, so the traced function returns the array half and the
other half is captured once at trace time and recombined after every call.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup.pdesolve import pdesolve

__all__ = ["CompiledSolve", "compile"]

#: The traced arguments as the caller names them, in the order the executable takes them.
_NAMES = ("t_span[0]", "t_span[1]", "y0", "theta", "lambda0", "saveat.ts")


def _as_array(value: Any) -> Any:
    """``value`` as a JAX array, or ``None`` unchanged."""
    return None if value is None else jnp.asarray(value)


def _describe(value: Any) -> str:
    """``shape (8,) and dtype float64`` for an array, ``None`` for ``None``."""
    if value is None:
        return "None"
    return f"shape {tuple(value.shape)} and dtype {jnp.dtype(value.dtype).name}"


def _span(t_span: Any) -> tuple:
    """``(t0, t1)`` from ``t_span``, refused unless it is a pair."""
    try:
        t0, t1 = t_span
    except (TypeError, ValueError):
        raise ValueError(f"ds.compile: t_span must be a (t0, t1) pair, got {t_span!r}.") from None
    return t0, t1


def _save_times(saveat: Any) -> Optional[jax.Array]:
    """The save times ``compile`` traces, or ``None`` when ``saveat`` names none.

    Unordered times are sorted and deduplicated before the solve, which fixes the
    output shape from their values, so only ``ordered=True`` times can be traced.
    """
    ts = None if saveat is None else getattr(saveat, "ts", None)
    if ts is None:
        return None
    if not getattr(saveat, "ordered", False):
        raise ValueError(
            "ds.compile traces SaveAt.ts, and sorting and deduplication determine the "
            "fixed output shape from its values; pass ordered=True for times that are "
            "already strictly increasing."
        )
    return jnp.asarray(ts)


class CompiledSolve:
    """A solve compiled by [`compile`][diff_soup.compile], ready to run.

    Call it with new values of the shapes and dtypes it was compiled for. Every
    other argument of the solve is fixed at compile time.

    Attributes:
        compile_seconds: The wall time of tracing and compiling the solve, in seconds.
    """

    def __init__(self, executable, treedefs, statics, save_times, out_static, seconds):
        """Hold the compiled executable and what its inputs must look like."""
        self._executable = executable
        self._treedefs = treedefs
        self._statics = statics
        self._save_times = save_times
        self._out_static = out_static
        self.compile_seconds = seconds

    def __call__(self, t_span: tuple, y0: Any, *, theta: Any = None, lambda0: Any = None):
        """Run the compiled solve from new values.

        Args:
            t_span: Start and end times `(t0, t1)`.
            y0: Initial state.
            theta: Parameter values, required when `compile` was given them and `None` otherwise.
            lambda0: Initial values of the global unknowns, required when `compile` was given them and `None` otherwise.

        Returns:
            The [`Solution`][diff_soup.Solution] of the solve.

        Raises:
            ValueError: When an argument's shape or dtype differs from the compiled one.
        """
        t0, t1 = _span(t_span)
        values = (t0, t1, y0, theta, lambda0, self._save_times)
        args = tuple(
            self._check(name, value, treedef, static)
            for name, value, treedef, static in zip(_NAMES, values, self._treedefs, self._statics)
        )
        dynamic = self._executable(*args)
        return eqx.combine(dynamic, self._out_static)

    def _check(self, name, value, treedef, static):
        """The array half of one argument, refused unless it matches the compiled one."""
        if name != "theta":
            value = _as_array(value)
        dynamic, _ = eqx.partition(value, eqx.is_array)
        leaves, given = jax.tree_util.tree_flatten(dynamic)
        if given != treedef:
            if value is None:
                hint = "it was given at compile time, so pass a value of the compiled shape"
            elif treedef.num_leaves == 0:
                hint = "it was None at compile time, so pass None or compile again with it"
            else:
                hint = "pass a value built like the compiled one, or compile again"
            raise ValueError(f"ds.compile: {name} does not match the compiled solve: {hint}.")
        checked = []
        for leaf, want in zip(leaves, static):
            if getattr(leaf, "weak_type", False):
                # A Python number has no dtype of its own; it takes the compiled one.
                leaf = leaf.astype(want.dtype)
            if leaf.shape != want.shape or leaf.dtype != want.dtype:
                raise ValueError(
                    f"ds.compile: {name} was compiled for {_describe(want)}, "
                    f"got {_describe(leaf)}. Compile again for new shapes or dtypes."
                )
            checked.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, checked)

    def cost_analysis(self) -> Any:
        """The compiled executable's cost estimate, as `jax.stages.Compiled.cost_analysis` reports it."""
        return self._executable.cost_analysis()

    def memory_analysis(self) -> Any:
        """The compiled executable's memory use, as `jax.stages.Compiled.memory_analysis` reports it."""
        return self._executable.memory_analysis()

    def __repr__(self) -> str:
        """The compiled shapes and the compile time."""
        texts = []
        for name, treedef, static in zip(_NAMES, self._treedefs, self._statics):
            if treedef.num_leaves == 0:
                texts.append("None")
            elif name == "theta":
                texts.append(f"<{treedef.num_leaves} arrays>")
            else:
                (want,) = static
                texts.append(f"{jnp.dtype(want.dtype).name}{list(want.shape)}")
        t0, t1, y0, theta, lambda0, ts = texts
        shown = [f"t_span=({t0}, {t1})", f"y0={y0}", f"theta={theta}", f"lambda0={lambda0}"]
        if ts != "None":
            shown.append(f"ts={ts}")
        shown.append(f"compile_seconds={self.compile_seconds:.3g}")
        return f"CompiledSolve({', '.join(shown)})"


def compile(
    problem: Any,
    solver: Any,
    t_span: tuple,
    y0: Any,
    *,
    saveat: Any = None,
    theta: Any = None,
    lambda0: Optional[Any] = None,
    adjoint: Optional[Any] = None,
    event: Optional[Any] = None,
    control_state0: Optional[Any] = None,
    consistent_ic: Optional[bool] = None,
    **solver_kwargs: Any,
) -> CompiledSolve:
    """Compile a [`pdesolve`][diff_soup.pdesolve.pdesolve] call without running it.

    The arguments are those of `pdesolve`. `t_span`, `y0`, `theta` and `lambda0` fix only shapes and dtypes, and every other argument, the save times included, is fixed in the compiled solve. Nothing is solved here.

    ```python
    compiled = ds.compile(problem, solver, (0.0, 0.1), y0)
    sol = compiled((0.0, 0.2), 2 * y0)
    ```

    Args:
        problem: The problem, as for `pdesolve`.
        solver: The solver, as for `pdesolve`.
        t_span: Start and end times `(t0, t1)`, whose values may change at each call.
        y0: Initial state, whose values may change at each call.
        saveat: A [`SaveAt`][diff_soup.SaveAt], whose `ts` must be `ordered=True`.
        theta: Parameter values, then required at each call, or `None` for the problem's declared values.
        lambda0: Initial values of the global unknowns, then required at each call.
        adjoint: Differentiation strategy, as for `pdesolve`.
        event: An [`Event`][diff_soup.Event] monitored during the solve.
        control_state0: Initial operating point of a problem with declared controls.
        consistent_ic: Whether to make the initial state consistent before the first step.
        **solver_kwargs: Engine keywords for `solver.solve`, as for `pdesolve`.

    Returns:
        A `CompiledSolve` that runs the solve.

    Raises:
        ValueError: When `saveat.ts` is given with `ordered=False`.
    """
    t0, t1 = _span(t_span)
    save_times = _save_times(saveat)
    values = (_as_array(t0), _as_array(t1), _as_array(y0), theta, _as_array(lambda0), save_times)

    halves = [eqx.partition(value, eqx.is_array) for value in values]
    treedefs = tuple(jax.tree_util.tree_structure(dynamic) for dynamic, _ in halves)
    statics = tuple(
        tuple(
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
            for leaf in jax.tree_util.tree_leaves(dynamic)
        )
        for dynamic, _ in halves
    )
    rests = tuple(rest for _, rest in halves)
    captured: list = []

    def run(*dynamics):
        t0, t1, y0, theta, lambda0, ts = (
            eqx.combine(dynamic, rest) for dynamic, rest in zip(dynamics, rests)
        )
        plan = saveat if ts is None else dataclasses.replace(saveat, ts=ts)
        solution = pdesolve(
            problem,
            solver,
            (t0, t1),
            y0,
            saveat=plan,
            theta=theta,
            lambda0=lambda0,
            adjoint=adjoint,
            event=event,
            control_state0=control_state0,
            consistent_ic=consistent_ic,
            **solver_kwargs,
        )
        dynamic, static = eqx.partition(solution, eqx.is_array)
        captured.append(static)
        return dynamic

    abstract = tuple(
        jax.tree_util.tree_unflatten(treedef, list(static))
        for treedef, static in zip(treedefs, statics)
    )
    start = time.perf_counter()
    executable = jax.jit(run).lower(*abstract).compile()
    seconds = time.perf_counter() - start
    return CompiledSolve(executable, treedefs, statics, save_times, captured[-1], seconds)
