"""diff_soup.pdesolve — the public forward-simulation entry point.

``pdesolve`` is a thin polymorphic dispatcher: it validates ``t_span``,
normalizes the authored problem to the solver view, and forwards to
``solver.solve(problem, t_span, y0, **kwargs)``. The solver object owns its
own numerics (tolerances, linear backend, execution profile); this function
owns nothing but the per-run call — the span, the states, the output plan,
the differentiation strategy, and any events.

Usage::

    from diff_soup import pdesolve, JNDF, SaveAt

    sol = pdesolve(problem, JNDF.stock(rtol=1e-6, atol=1e-6), (0.0, 1.0), y0,
                    saveat=SaveAt(t1=True))
    print(sol.ts, sol.ys, sol.ok, sol.result)

``adjoint=None`` (the default) selects the forward-only differentiation
strategy: the cheapest primal solve plus forward-mode (``jax.jvp`` /
``jax.jacfwd``) support in O(1) memory. Reverse-mode differentiation
(``jax.grad`` / ``jax.vjp``) through the solve is an explicit opt-in — pass a
checkpointed adjoint object (e.g.
``adjoint=diff_soup.CheckpointAdjoint()``).

``pdesolve`` returns the result-coded
[Solution][diff_soup.timestepping.vsvo.solution.Solution]: a solve that does
not reach ``t_span[1]`` is never silently reported as success. Read
``sol.ok`` / ``sol.result`` / ``sol.stats`` to inspect the
outcome — ``pdesolve`` never raises on a non-convergent solve.

Every engine-specific knob (``dt0``, ``max_steps``, ``hmin``, ``hmax``,
``max_order``, ``linear``, ...) passes through ``**solver_kwargs`` unfiltered
to ``solver.solve``; the solver's own signature is the authority on what it
accepts, so an unrecognized keyword fails loudly there rather than being
silently dropped here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from diff_soup.timestepping.vsvo.solution import Solution

__all__ = ["pdesolve"]


def _coerce_problem(problem: Any) -> Any:
    """Normalize an authored problem to its identity-stable solver view.

    This top-level orchestration boundary accepts a
    ``MassProblem``/``ImplicitProblem``, an authored source exposing
    ``solver_problem()``, or an already-normalized view. The normalizer
    rejects every historical or incomplete runtime surface before a solver
    traces callbacks.

    Args:
        problem: The first positional argument to [pdesolve][].

    Returns:
        The cached solver view.
    """
    from diff_soup.problem.implicit import get_or_build_solver_view

    return get_or_build_solver_view(problem)


def _check_theta_structure(problem: Any, theta: Any) -> None:
    """Raise when ``theta`` is not shaped like the problem's own parameter pytree.

    A mapping ``{handle: value}``, a theta built for another problem, or a bare
    array would otherwise fail deep in the solver with JAX's pytree-structure
    message. Problems that declare no parameters are not checked.
    """
    import jax

    getter = getattr(problem, "default_theta", None)
    default = getter() if callable(getter) else None
    if default is None:
        return
    expected = jax.tree_util.tree_structure(default)
    given = jax.tree_util.tree_structure(theta)
    if given == expected:
        return
    hint = (
        "build it with ds.default_theta(problem, {handle: value})"
        if isinstance(theta, dict)
        else "pass the pytree ds.default_theta(problem, ...) returns for this problem"
    )
    raise ValueError(
        "pdesolve: theta does not have this problem's parameter structure; "
        f"{hint}. Got {type(theta).__name__} with {given.num_leaves} leaves, "
        f"expected {expected.num_leaves} leaves."
    )


def pdesolve(
    problem: Any,
    solver: Any,
    t_span: tuple,
    y0: Any,
    *,
    saveat: Any = None,
    lambda0: Optional[Any] = None,
    theta: Any = None,
    adjoint: Optional[Any] = None,
    event: Optional[Any] = None,
    control_state0: Optional[Any] = None,
    consistent_ic: Optional[bool] = None,
    trace: bool = False,
    **solver_kwargs: Any,
) -> "Solution":
    """Integrate a problem over a time span and return its [`Solution`][diff_soup.Solution].

    Every keyword reaches `solver.solve`, so engine keywords such as `dt0`,
    `max_steps` and `tolerances` pass through. A one-time warning fires when
    JAX's 64-bit mode is off.

    Args:
        problem: A [`LoweredProblem`][diff_soup.problem.LoweredProblem], [`MassProblem`][diff_soup.problem.MassProblem], or [`ImplicitProblem`][diff_soup.problem.ImplicitProblem].
        solver: A solver with a `solve` method, such as [`JNDF`][diff_soup.JNDF], [`JBDF`][diff_soup.JBDF] or [`JIDA`][diff_soup.JIDA].
        t_span: Start and end times `(t0, t1)`.
        y0: Initial state, the full state vector or, when `lambda0` is given, the field block.
        saveat: A [`SaveAt`][diff_soup.SaveAt] choosing the saved outputs, or `None` for the final state only.
        lambda0: Initial values of the global unknowns, packed after `y0`.
        theta: Parameter values, or `None` for the problem's declared values.
        adjoint: Differentiation strategy, such as [`CheckpointAdjoint`][diff_soup.CheckpointAdjoint] for reverse mode, or `None` for the solver's own.
        event: An [`Event`][diff_soup.Event] or [`EventSet`][diff_soup.timestepping.EventSet] monitored during the solve.
        control_state0: Initial operating point of a problem with declared controls.
        consistent_ic: Whether to make the initial state consistent before the first step, or `None` for the solver's default.
        trace: If `True`, attach a per-step diagnostic payload to the solution.
        **solver_kwargs: Engine keywords for `solver.solve`, such as `dt0`, `max_steps`, `tolerances` and `linear`.

    Returns:
        The [`Solution`][diff_soup.Solution] of the solve.
    """
    try:
        _t_len = len(t_span)
    except TypeError:
        raise TypeError(f"pdesolve: t_span must be a (t0, t1) tuple, got {type(t_span).__name__}.")
    if _t_len != 2:
        raise ValueError(f"pdesolve: t_span must have length 2 (t0, t1), got length {_t_len}.")

    if not hasattr(solver, "solve"):
        raise TypeError(
            f"pdesolve: solver of type {type(solver).__name__} does not implement the "
            "solver protocol. Pass an object exposing .solve(problem, t_span, y0, "
            "**kwargs) -- e.g. JNDF, JBDF, or JIDA."
        )

    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    # Freeze the one runtime contract before choosing an engine.
    problem = _coerce_problem(problem)
    if theta is not None:
        _check_theta_structure(problem, theta)

    solve_kwargs: dict = dict(
        saveat=saveat,
        theta=theta,
        lambda0=lambda0,
        adjoint=adjoint,
        trace=trace,
    )
    # These three are only forwarded when the caller actually supplies a
    # value, so a solver whose .solve does not (yet) declare the parameter is
    # unaffected by a default-path call -- mirroring the compile-once
    # contract: the default-path signature to every solver stays unchanged.
    if event is not None:
        solve_kwargs["event"] = event
    if control_state0 is not None:
        solve_kwargs["control_state0"] = control_state0
    if consistent_ic is not None:
        solve_kwargs["consistent_ic"] = consistent_ic
    solve_kwargs.update(solver_kwargs)

    return solver.solve(problem, t_span, y0, **solve_kwargs)
