"""Sensitivity runs: a solve wired for forward sensitivities and design gradients.

[sensitivity_solve][diff_soup.inference.design.solve.sensitivity_solve] splits a problem's parameters into the *analyzed*
set (``wrt`` — what the experiment tries to identify) and everything else
(including the *design* set — protocol coefficients an experiment-design loop
optimizes), and returns a
[SensitivityRun][diff_soup.inference.design.solve.SensitivityRun]:

* ``run(x)`` solves with the analyzed parameters set to the flat vector ``x``
  and returns the saved frames — using the solver's composable
  forward-sensitivity mode, so
* ``jax.jacfwd(run)(run.x0)`` is the sensitivity matrix
  ``S[f, k, j] = d y_f[k] / d x_j`` of shape
  ``(n_frames, n_state, n_directions)``, ``n_directions = run.x0.size``, and
* ``jax.grad`` (or ``jax.hessian``) of any scalar criterion built from ``S``
  — e.g. a Fisher-information design criterion — composes through the solve
  with respect to design parameters: ``jax.grad(lambda d: criterion(
  jax.jacfwd(run)(run.x0, run.design_theta(d))))(run.d0)``.

A direct ``jax.grad`` of the run itself is unsupported in this mode (the
solver raises with the same message for any reverse gradient of the solution);
use the default adjoints for plain reverse-mode training gradients.

The analyzed and design sets must be disjoint: a coefficient the experiment
optimizes is not a quantity the experiment identifies.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from diff_soup.inference.fitting._forward_seam import parameter_handle_leaf_ids
from diff_soup.inference.fitting._path_filter import parameters_at

__all__ = ["SensitivityRun", "sensitivity_solve"]


def _mask_for(base_theta: Any, selectors: tuple, set_name: str, *, problem: Any = None) -> Any:
    """Build a theta mask from Recipe handles or dotted paths into the parameter pytree."""
    if selectors and not all(isinstance(selector, str) for selector in selectors):
        if any(isinstance(selector, str) for selector in selectors):
            raise TypeError(
                f"sensitivity_solve {set_name} cannot mix Recipe handles and dotted paths."
            )
        not_learnable = tuple(
            getattr(handle, "name", repr(handle))
            for handle in selectors
            if getattr(handle, "learnable", None) is False
        )
        if not_learnable:
            raise ValueError(
                f"sensitivity_solve: {set_name} handle(s) {not_learnable} were "
                "declared with learnable=False and therefore are fixed values, "
                "not parameter-pytree leaves. Declare each analyzed or designed "
                "Recipe parameter with learnable=True."
            )
        handle_ids = tuple(
            parameter_handle_leaf_ids(handle, problem=problem) for handle in selectors
        )
        missing = tuple(
            getattr(handle, "name", repr(handle))
            for handle, ids in zip(selectors, handle_ids)
            if not ids
        )
        if missing:
            raise ValueError(
                f"sensitivity_solve: {set_name} handle(s) {missing} expose no "
                "selectable parameter array. Recipe parameters analyzed or "
                "designed here must be declared with learnable=True."
            )
        leaf_ids = frozenset().union(*handle_ids)
        mask = jax.tree.map(lambda leaf: id(leaf) in leaf_ids, base_theta)
        if not any(jax.tree_util.tree_leaves(mask)):
            raise ValueError(
                f"sensitivity_solve: {set_name} handles do not belong to this "
                "Recipe problem. Use handles returned by the same spec declaration."
            )
        return mask
    try:
        return parameters_at(base_theta, selectors)
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"sensitivity_solve: a {set_name} path does not resolve in the "
            f"problem's parameter pytree ({exc}). Pass Recipe handles, or dotted "
            "paths that resolve to array leaves."
        ) from exc


def _leaf_labels(tree: Any) -> tuple:
    """Per-scalar labels for a pytree's leaves, in flattening (ravel) order."""
    labels: list[str] = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        parts: list[str] = []
        for entry in path:
            if hasattr(entry, "key"):
                parts.append(str(entry.key))
            elif hasattr(entry, "name"):
                parts.append(str(entry.name))
            elif hasattr(entry, "idx"):
                parts.append(str(entry.idx))
            else:
                parts.append(str(entry))
        base = ".".join(parts)
        size = int(jnp.size(leaf))
        if size == 1:
            labels.append(base)
        else:
            labels.extend(f"{base}[{i}]" for i in range(size))
    return tuple(labels)


class SensitivityRun:
    """A solve wired for sensitivity analysis. Callable.

    Calling the run with a flat analyzed-parameter vector returns the saved
    frames; ``jax.jacfwd`` of the call gives the sensitivities and any scalar
    criterion of those sensitivities can be reverse-differentiated with
    respect to the design parameters (see the module docstring for the
    composition pattern).

    Attributes:
        x0: Flat default values of the analyzed (``wrt``) parameters.
        names: Recipe handle names for analyzed directions.
        leaf_paths: Lowered paths corresponding to analyzed directions.
        design_leaf_paths: Lowered paths corresponding to design directions.
        d0: Flat default values of the design parameters (empty when no
            ``design`` paths were declared).
        design_names: One label per entry of ``d0``.
        n_directions: Number of sensitivity directions (``x0.size``).
        unravel: Maps a flat analyzed vector back to the analyzed subtree —
            e.g. the ``theta`` to hand a parameter-reading channel.
    """

    def __init__(
        self,
        problem: Any,
        t_span: tuple,
        y0: Union[Array, Callable],
        *,
        base_theta: Any,
        wrt_mask: Any,
        design_mask: Any,
        solver: Any,
        adjoint: Any,
        saveat: Any,
        include_multipliers: bool,
        solve_kwargs: dict,
    ):
        """Bind the solve configuration. Built by [sensitivity_solve][]."""
        self._problem = problem
        self._t_span = t_span
        self._y0 = y0
        self._base_theta = base_theta
        self._wrt_mask = wrt_mask
        self._design_mask = design_mask
        self._solver = solver
        self._adjoint = adjoint
        self._saveat = saveat
        self._include_multipliers = bool(include_multipliers)
        self._solve_kwargs = dict(solve_kwargs)

        wrt_leaves, _ = eqx.partition(base_theta, wrt_mask)
        self.x0, self.unravel = ravel_pytree(wrt_leaves)
        self._unravel = self.unravel
        self.names = _leaf_labels(wrt_leaves)
        self.n_directions = int(self.x0.size)

        if design_mask is None:
            self.d0 = jnp.zeros((0,), dtype=self.x0.dtype)
            self.design_names = ()
            self._design_unravel = None
        else:
            design_leaves, _ = eqx.partition(base_theta, design_mask)
            self.d0, self._design_unravel = ravel_pytree(design_leaves)
            self.design_names = _leaf_labels(design_leaves)

    def _theta_full(self, x: Array, theta: Any) -> Any:
        """Graft the flat analyzed vector ``x`` into the parameter pytree."""
        base = self._base_theta if theta is None else theta
        _, rest = eqx.partition(base, self._wrt_mask)
        return eqx.combine(self._unravel(jnp.asarray(x)), rest)

    def wrt_tangents(self) -> Any:
        """Per-direction tangents of the analyzed subtree.

        Returns:
            A pytree shaped like ``unravel(x0)`` whose leaves carry a leading
            ``n_directions`` axis: entry ``j`` is the analyzed subtree moved
            along basis direction ``j``. Pass it as ``theta_tangents`` to
            [observable_sensitivities][diff_soup.inference.design.fisher.observable_sensitivities]
            for channels that read the analyzed parameters directly.
        """
        basis = jnp.eye(self.n_directions, dtype=self.x0.dtype)
        return jax.vmap(self.unravel)(basis)

    def __call__(self, x: Array, theta: Any = None) -> Array:
        """Solve with analyzed parameters ``x`` and return the saved frames.

        Args:
            x: Flat analyzed-parameter vector (``x0``-shaped).
            theta: Optional replacement for the full parameter pytree —
                typically
                [SensitivityRun.design_theta][diff_soup.inference.design.solve.SensitivityRun.design_theta]
                output. The analyzed entries of ``theta`` are overridden by
                ``x``.

        Returns:
            Saved frames, shape ``(n_frames, n_state)`` — with the bordered-global
            columns appended when the run was built with
            ``include_multipliers=True``.
        """
        sol = self.solution(x, theta)
        ys = sol.ys
        if self._include_multipliers and sol.lambdas is not None:
            ys = jnp.concatenate([ys, sol.lambdas], axis=-1)
        return ys

    def solution(self, x: Optional[Array] = None, theta: Any = None) -> Any:
        """Run the solve and return the full solution object.

        Args:
            x: Flat analyzed-parameter vector (default ``x0``).
            theta: Optional full parameter pytree replacement.

        Returns:
            The solver's solution (frames, bordered globals, statistics).
        """
        x = self.x0 if x is None else x
        theta_full = self._theta_full(x, theta)
        y0 = self._y0(theta_full) if callable(self._y0) else self._y0
        return self._solver.solve(
            self._problem,
            self._t_span,
            y0,
            theta=theta_full,
            saveat=self._saveat,
            adjoint=self._adjoint,
            **self._solve_kwargs,
        )

    def design_theta(self, d: Array, theta: Any = None) -> Any:
        """Graft flat design values into the parameter pytree.

        Args:
            d: Flat design vector (``d0``-shaped).
            theta: Optional base pytree (default: the run's base parameters).

        Returns:
            The full parameter pytree with the design entries set to ``d`` —
            pass it as the run's ``theta`` argument.

        Raises:
            ValueError: If the run was built without ``design`` paths.
        """
        if self._design_unravel is None:
            raise ValueError(
                "design_theta: this run declares no design parameters; pass "
                "design=(...paths...) to sensitivity_solve."
            )
        base = self._base_theta if theta is None else theta
        _, rest = eqx.partition(base, self._design_mask)
        return eqx.combine(self._design_unravel(jnp.asarray(d)), rest)

    def sensitivities(self, x: Optional[Array] = None, theta: Any = None) -> Array:
        """Return the sensitivity matrix ``S[frame, state, direction]``.

        Args:
            x: Flat analyzed-parameter vector (default ``x0``).
            theta: Optional full parameter pytree replacement.

        Returns:
            ``jax.jacfwd`` of the run at ``x``, shape
            ``(n_frames, n_state, n_directions)``.
        """
        x = self.x0 if x is None else x
        return jax.jacfwd(lambda x_: self(x_, theta))(x)


def sensitivity_solve(
    problem: Any,
    t_span: tuple,
    y0: Union[Array, Callable],
    *,
    wrt: tuple,
    saveat: Any,
    solver: Any = None,
    design: tuple = (),
    theta: Any = None,
    include_multipliers: bool = False,
    **solve_kwargs: Any,
) -> SensitivityRun:
    """Wire a problem for sensitivity analysis and experiment design.

    Args:
        problem: Any problem the solver suite integrates.
        t_span: ``(t0, t1)`` integration interval.
        y0: Initial state, or a callable ``y0(theta) -> state`` for initial
            states that depend on the parameters (consistent initial values).
        wrt: Recipe parameter or function handles to analyze, or, for a
            parameter pytree that is not a Recipe's, dotted paths to its
            array leaves.
        saveat: The save grid (``SaveAt``); its frames are the rows the
            Fisher information sums over.
        solver: A configured solver object (default: the stock implicit DAE
            solver). The run always solves in the composable
            forward-sensitivity mode regardless of the solver's own adjoint.
        design: Optional Recipe handles (or dotted paths) for
            design/protocol coefficients. Must be disjoint from ``wrt``.
        theta: Optional base parameter pytree (default: the problem's own).
        include_multipliers: Append the bordered-global columns to the returned
            frames, differential and algebraic globals alike. Dense output interpolates differential globals and holds or
            projects algebraic globals according to the save policy; held algebraic
            columns on an interpolated ``ts`` grid carry a step-sized bias, and that
            case only raises a ``UserWarning``. Exact algebraic-global values require
            observing at the integration endpoint with ``SaveAt(t1=True)``.
        **solve_kwargs: Forwarded to the solver's ``solve`` (e.g. ``max_steps``).

    Returns:
        A [SensitivityRun][diff_soup.inference.design.solve.SensitivityRun]
        whose call ``run(x, theta)`` is differentiable by ``jax.jacfwd`` with
        respect to ``x`` and, through a criterion of those sensitivities, by
        ``jax.grad`` with respect to the design values ``d`` of
        ``run.design_theta(d)``.

    Raises:
        ValueError: If ``wrt`` is empty, a path does not resolve to a
            parameter leaf, or a path appears in both ``wrt`` and ``design``.
    """
    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    from diff_soup._adjoints import ForwardAdjoint
    from diff_soup.problem import default_theta as problem_default_theta

    wrt = tuple(wrt)
    design = tuple(design)
    if not wrt:
        raise ValueError("sensitivity_solve: wrt must name at least one parameter path to analyze.")
    overlap = tuple(selector for selector in wrt if selector in design)
    if overlap:
        raise ValueError(
            "sensitivity_solve: parameters cannot be both analyzed (wrt) and "
            f"designed: {[getattr(item, 'name', item) for item in overlap]}; wrt and design must be disjoint."
        )

    base_theta = problem_default_theta(problem) if theta is None else theta
    wrt_mask = _mask_for(base_theta, wrt, "wrt", problem=problem)
    design_mask = _mask_for(base_theta, design, "design", problem=problem) if design else None
    if design_mask is not None:
        shared = jax.tree.map(lambda a, b: bool(a) and bool(b), wrt_mask, design_mask)
        if any(jax.tree_util.tree_leaves(shared)):
            raise ValueError(
                "sensitivity_solve: wrt and design resolve to overlapping "
                "parameter leaves; the two sets must be disjoint."
            )

    if solver is None:
        from diff_soup import JIDA

        solver = JIDA.stock()

    if include_multipliers and getattr(saveat, "ts", None) is not None:
        warnings.warn(
            "sensitivity_solve: algebraic bordered-global frames on an interpolated "
            "ts grid may be held at accepted-step values rather than projected, so "
            "those channels carry a step-sized bias there. Observe algebraic globals "
            "at the integration endpoint (SaveAt(t1=True)) for exact values.",
            UserWarning,
            stacklevel=2,
        )

    run = SensitivityRun(
        problem,
        (float(t_span[0]), float(t_span[1])),
        y0,
        base_theta=base_theta,
        wrt_mask=wrt_mask,
        design_mask=design_mask,
        solver=solver,
        adjoint=ForwardAdjoint(composable=True),
        saveat=saveat,
        include_multipliers=include_multipliers,
        solve_kwargs=solve_kwargs,
    )

    run.leaf_paths = run.names
    run.design_leaf_paths = run.design_names

    def labels(selectors, mask, fallback):
        if not selectors or any(isinstance(handle, str) for handle in selectors):
            return fallback
        selected, _ = eqx.partition(base_theta, mask)
        owners = [
            (handle.name, parameter_handle_leaf_ids(handle, problem=problem))
            for handle in selectors
        ]
        names = []
        for leaf in jax.tree_util.tree_leaves(selected):
            name = next(name for name, ids in owners if id(leaf) in ids)
            size = int(jnp.size(leaf))
            names.extend((name,) if size == 1 else tuple(f"{name}[{i}]" for i in range(size)))
        return tuple(names)

    run.names = labels(wrt, wrt_mask, run.leaf_paths)
    run.design_names = labels(design, design_mask, run.design_leaf_paths)
    return run
