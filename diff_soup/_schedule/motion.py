"""A segment's solve under a moving endpoint: the field's units, its velocity, and their tangents."""

from __future__ import annotations

import functools
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.pdesolve import pdesolve
from diff_soup.problem.implicit import solver_view
from diff_soup.timestepping import SaveAt


def _field_units(view: Any) -> Optional[tuple]:
    """``(scale, offset)`` that map the solver's field unknowns back to the form of ``y0``.

    ``pdesolve`` hands ``y0`` to the solver view's ``to_internal_state``: a lowered
    problem stores its field unknowns as ``(value - offset) / scale``
    (``LoweredProblem.initial_u``) and takes the bordered globals of a full state as
    they stand; any other problem takes ``y0`` as it stands. The solver's states, the
    terminal one included, stay in the stored units, so the inverse is
    ``offset + scale * stored``. Returns ``None`` when the translation is the identity.

    Raises:
        ConfigurationError: If the problem translates its initial state with a
            conversion that has no inverse here.
    """
    source = getattr(view, "source", None)
    if source is None:
        return None
    if not view.direct_source:
        if callable(getattr(source, "to_internal_state", None)):
            raise ConfigurationError(
                f"a schedule hands each segment the state the previous one ended at, and "
                f"{type(source).__name__} translates the initial state with its own "
                "to_internal_state, which the schedule cannot invert; solve the segments one "
                "by one with pdesolve instead."
            )
        return None
    if not (
        callable(getattr(source, "initial_u", None))
        and callable(getattr(source, "initial_lambda", None))
    ):
        return None
    arrays = getattr(source, "_scale_arrays", None)
    if not callable(arrays):
        raise ConfigurationError(
            f"a schedule hands each segment the state the previous one ended at, and "
            f"{type(source).__name__} stores its field unknowns through initial_u without "
            "the field scales and offsets the schedule needs to map them back; solve the "
            "segments one by one with pdesolve instead."
        )
    # The arrays are constants of the model; evaluate them now even when the
    # schedule is being traced, so the identity can be recognized.
    with jax.ensure_compile_time_eval():
        scale, offset = arrays()
    if jnp.shape(scale) != (int(view.n_u),) or jnp.shape(offset) != (int(view.n_u),):
        raise ConfigurationError(
            f"the field scales of {type(source).__name__} cover {jnp.shape(scale)} and "
            f"{jnp.shape(offset)} entries, not the {int(view.n_u)} field unknowns; the "
            "schedule cannot map the carried state back."
        )
    traced = isinstance(scale, jax.core.Tracer) or isinstance(offset, jax.core.Tracer)
    if not traced and np.all(np.asarray(scale) == 1.0) and np.all(np.asarray(offset) == 0.0):
        return None
    return jnp.asarray(scale), jnp.asarray(offset)


def _slope(view, solver, t, state, theta, control):
    """The time derivative of the consistent state through ``(t, state)``, in the solver's units.

    One consistent initialization at ``(t, state)``; the slope is the tangent of its
    consistent state along ``(1, ydot)``, which ``jax.linearize`` reads from the
    implicit-function rule at the converged root, so the initialization is solved once.
    ``NaN`` where the initialization does not converge.
    """

    def consistent(time, value):
        result = view.consistent_initial_state(
            time,
            value,
            theta,
            control,
            tol=solver.atol,
            rtol=solver.rtol,
            backend=solver.linear,
        )
        return result.y, (result.ydot, result.converged)

    _, along, (ydot, converged) = jax.linearize(consistent, t, state, has_aux=True)
    slope = along(jnp.ones_like(t), ydot)
    return jnp.where(converged, slope, jnp.nan)


def _acceleration(view, solver, t, state, theta, control):
    """The second time derivative of the consistent state through ``(t, state)``."""
    slope, along = jax.linearize(
        lambda time, value: _slope(view, solver, time, value, theta, control), t, state
    )
    return along(jnp.ones_like(t), slope)


@eqx.filter_custom_jvp
def _vanishing(term, like, scale, args):
    """``scale * term(*args)`` at ``scale == 0``: zeros shaped as ``like``, differentiable.

    The value is zero and ``term`` is not evaluated. A derivative evaluates only the
    part it needs: a tangent of ``scale`` gives ``dscale * term(*args)``, and a tangent of
    ``args`` gives another vanishing product, ``scale`` times the tangent of ``term``, whose
    value is again zero. So ``term`` runs only where ``scale`` is differentiated, and its
    derivatives only where ``scale`` is differentiated again.
    """
    del term
    zero = jnp.zeros(like.shape, like.dtype)
    # The zeros select over every input, so ``jax.vmap`` batches them wherever it batches
    # an input, as it batches the tangent.
    for leaf in jax.tree_util.tree_leaves((scale, args)):
        if eqx.is_array(leaf):
            zero = jnp.where(False, jnp.sum(jnp.ravel(leaf)[:1]).astype(like.dtype), zero)
    return zero


def _term_tangent(term, args, dargs):
    """The tangent of ``term(*args)`` along ``dargs``."""
    return eqx.filter_jvp(term, args, dargs)[1]


@_vanishing.def_jvp
def _vanishing_jvp(primals, tangents):
    """``dscale * term(*args)``, plus ``scale`` times the tangent of ``term``, itself vanishing."""
    term, like, scale, args = primals
    _, _, dscale, dargs = tangents
    out = _vanishing(term, like, scale, args)
    tangent = None
    if dscale is not None:
        # A term that cannot be evaluated (a boundary whose initialization fails) has no
        # derivative: its tangent is NaN, and a reverse sweep gives its input a zero
        # cotangent rather than zero times NaN.
        value = term(*args)
        finite = jnp.isfinite(value)
        tangent = jnp.where(finite, dscale * jnp.where(finite, value, 0.0), jnp.nan)
    if jax.tree_util.tree_leaves(dargs):
        deeper = _vanishing(functools.partial(_term_tangent, term), like, scale, (args, dargs))
        tangent = deeper if tangent is None else tangent + deeper
    return out, tangent


def _like(state):
    """The shape and dtype of ``state``."""
    return jax.ShapeDtypeStruct(jnp.shape(state), jnp.result_type(state))


def _velocity(view, solver, t, step, state, theta, control):
    """The slope at ``(t, state)`` plus half the vanishing ``step`` times the acceleration."""
    half = _vanishing(_acceleration, _like(state), step, (view, solver, t, state, theta, control))
    return _slope(view, solver, t, state, theta, control) + 0.5 * half


def _moved(view, solver, t, step, state, theta, control):
    """How far the trajectory through ``(t, state)`` moves over the vanishing time ``step``.

    ``step * f + step**2 / 2 * f'``, with ``f`` the slope and ``f'`` the acceleration at
    ``(t, state)``: the Taylor shift of the flow, exact in every derivative that
    differentiates ``step`` at most twice. Zero in value.
    """
    args = (view, solver, t, step, state, theta, control)
    return _vanishing(_velocity, _like(state), step, args)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _motion(order, shift):
    """The motion of a segment end, zero in value, differentiable ``order`` more times."""
    del order
    return shift


@_motion.defjvp
def _motion_jvp(order, primals, tangents):
    """Pass the tangent on; refuse a derivative past the order the Taylor shifts carry."""
    if order == 0:
        raise NotImplementedError(
            "solve_schedule has no third derivative with respect to the start or end of a "
            "segment (a segment duration, t0, t_end or a sample time at a segment's end): "
            "a moving segment end enters as a second-order Taylor shift along the flow. "
            "Differentiate at most twice in these times."
        )
    (shift,), (dshift,) = primals, tangents
    return _motion(order - 1, shift), dshift


#: The derivatives a moving segment end carries: its Taylor shift is of second order.
_MOTION_ORDER = 2


def _endpoint_motion(value):
    """``value`` minus itself held out of differentiation: zero, with ``value``'s tangents."""
    return _motion(_MOTION_ORDER, value - jax.lax.stop_gradient(value))


def _segment_solve_impl(problem, solver, span, state, times, theta, control, options):
    """One unchanged primal solve, with moving-boundary flow derivatives."""
    solution = pdesolve(
        problem,
        solver,
        span,
        state,
        saveat=SaveAt(ts=times, ordered=True),
        theta=theta,
        control_state0=control,
        **options,
    )
    # The merged schedule owns no problem object; keep this AD boundary array-only.
    return eqx.tree_at(lambda s: s.prob, solution, None)


def _moving_solve(problem, solver, span, state, times, theta, control, options):
    """The segment's solve as a function of its ends, for differentiation.

    The solve runs between its ends held fixed, so its adaptive mesh stays frozen.
    Moving the start by ``s`` starts the solve from the state the flow through
    ``(start + s, state)`` has at ``start``; moving the end by ``e`` moves the terminal
    state along the flow by ``e``. Both are second-order Taylor shifts of the flow
    (``_moved``), zero in value. A sample at the start is the state at the start, and
    moves with it back along the start's shift; a sample at the end is the solution at
    that fixed time, and moves only with its own time.
    """
    view = solver_view(problem)
    n_u = int(view.n_u)
    units = _field_units(view)
    held = jax.lax.stop_gradient(span)
    start, end = _endpoint_motion(span[0]), _endpoint_motion(span[1])
    back = _moved(
        view, solver, held[0] + start, -start, view.to_internal_state(state), theta, control
    )
    shift = back if units is None else back.at[:n_u].multiply(units[0])
    out = _segment_solve_impl(problem, solver, held, state + shift, times, theta, control, options)
    ok = jnp.asarray(out.ok, dtype=bool)
    terminal = jnp.concatenate([out.terminal.y, out.terminal.lam])

    def rows(values):
        """The samples as full states, the field unknowns then the globals."""
        return values if out.lambdas is None else jnp.concatenate([values, out.lambdas], axis=1)

    # The flow carries the consistent state at the held start to the moved start
    # by minus the start's own shift, to the shift's order.
    at_start = (times == held[0])[:, None]
    samples = rows(out.ys) - jnp.where(at_start, back[None, :], 0.0)
    # SaveAt clips a query to the span, which halves a query-time tangent at
    # the end; the sample at the end is the terminal state, moved by its own time.
    at_end = ((times == held[1]) & ok)[:, None]
    query = _endpoint_motion(times)
    to_query = jax.vmap(lambda step: _moved(view, solver, held[1], step, terminal, theta, control))
    at_end_rows = (
        jax.lax.stop_gradient(samples)
        + (terminal - jax.lax.stop_gradient(terminal))[None, :]
        + to_query(query)
    )
    samples = jnp.where(at_end, at_end_rows, samples)
    forward = jnp.where(ok, _moved(view, solver, held[1], end, terminal, theta, control), 0.0)
    terminal = terminal + forward
    out = eqx.tree_at(
        lambda s: (s.ys, s.terminal.y, s.terminal.lam, s.terminal.t, s.stats.t_reached),
        out,
        (
            samples[:, :n_u],
            terminal[:n_u],
            terminal[n_u:],
            out.terminal.t + jnp.where(ok, end, 0.0),
            out.stats.t_reached + jnp.where(ok, end, 0.0),
        ),
    )
    if out.lambdas is not None:
        out = eqx.tree_at(lambda s: s.lambdas, out, samples[:, n_u:])
    return out


_segment_solve = eqx.filter_custom_jvp(_segment_solve_impl)


@_segment_solve.def_jvp
def _segment_solve_jvp(primals, tangents):
    """Differentiate the solve with its ends moving (``_moving_solve``)."""
    return eqx.filter_jvp(_moving_solve, primals, tangents)
