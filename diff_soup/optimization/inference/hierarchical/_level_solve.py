"""A level system solved by elimination up its tree: the fold, the damped and scaled solves, the attempted step."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.hierarchical._level_system import (
    LevelSystem,
    _addition,
    _dtype,
    _offsets,
)

# ---------------------------------------------------------------------------
# The elimination (inside compiled programs)
# ---------------------------------------------------------------------------


def _split(system: LevelSystem, vector: ArrayLike) -> list:
    """A joint vector as the root part and one ``(N_l, v_l)`` part per level."""
    offsets = _offsets(system.counts, system.widths)
    parts = [vector[: system.widths[0]]]
    for level in range(1, system.depth + 1):
        parts.append(
            vector[offsets[level] : offsets[level + 1]].reshape(
                system.counts[level - 1], system.widths[level]
            )
        )
    return parts


def _node_sum(system: LevelSystem, values: ArrayLike, level: int, above: int) -> jax.Array:
    """Sum per-node values of ``level`` over the nodes below each node of ``above``."""
    if above == 0:
        return jnp.sum(values, axis=0)
    table = system.descendants[level][above]
    mask = system.present[level][above]
    gathered = values[table]
    keep = jnp.reshape(mask, mask.shape + (1,) * (values.ndim - 1))
    return jnp.sum(jnp.where(keep, gathered, jnp.zeros((), values.dtype)), axis=1)


def _ancestor_values(system: LevelSystem, values: ArrayLike, level: int, above: int) -> ArrayLike:
    """The values of each node's level-``above`` ancestor, one row per node of ``level``."""
    return values[system.ancestors[level][above]]


def _lower_solve(factor: ArrayLike, right: ArrayLike) -> jax.Array:
    """``L^-1 B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True))(factor, right)


def _upper_solve(factor: ArrayLike, right: ArrayLike) -> jax.Array:
    """``L^-T B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True, trans="T"))(
        factor, right
    )


def _fold(
    system: LevelSystem, damping: Real | ArrayLike, rhs: Sequence[ArrayLike] | None, lowest: int
) -> tuple:
    """Eliminate the levels ``D .. lowest`` from the experiments up.

    Every node of a level is factored at once, ``L_n = chol(diag~_n + damping I)``;
    its couplings ``Y_n^a = L_n^-1 cross~_n^a`` and right-hand sides
    ``y_n = L_n^-1 R~_n`` are formed, and ``(Y_n^a)^T Y_n^b`` and
    ``(Y_n^a)^T y_n`` are folded into the node's ancestors, summed over the
    nodes below each ancestor in node order.
    """
    widths = system.widths
    depth = system.depth
    dtype = system.root.dtype
    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    root = system.root
    root_rhs = None if rhs is None else rhs[0]
    level_rhs = None if rhs is None else list(rhs)
    factors: list = [None] * (depth + 1)
    couplings: list = [None] * (depth + 1)
    solved: list = [None] * (depth + 1)
    for level in range(depth, lowest - 1, -1):
        width = widths[level]
        if width == 0:
            continue
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping * jnp.eye(width, dtype=dtype)
        )
        factors[level] = factor
        row: list = [None] * level
        for above in range(level):
            if widths[above]:
                row[above] = _lower_solve(factor, cross[level][above])
        couplings[level] = row
        if rhs is not None:
            solved[level] = _lower_solve(factor, level_rhs[level])
        for above in range(level):
            if not widths[above]:
                continue
            for other in range(above + 1):
                if not widths[other]:
                    continue
                product = jnp.einsum("nia,nib->nab", row[above], row[other])
                total = _node_sum(system, product, level, above)
                if above == 0:
                    root = root - total
                elif other == above:
                    diagonal[above] = diagonal[above] - total
                else:
                    cross[above][other] = cross[above][other] - total
            if rhs is not None:
                folded = jnp.einsum("nia,nik->nak", row[above], solved[level])
                total = _node_sum(system, folded, level, above)
                if above == 0:
                    root_rhs = root_rhs - total
                else:
                    level_rhs[above] = level_rhs[above] - total
    return root, root_rhs, diagonal, cross, level_rhs, factors, couplings, solved


@jax.jit
def _solve(system, damping):
    counts, widths = system.counts, system.widths
    depth = system.depth
    g = widths[0]
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)
    matrix = system.constraint_matrix
    n_rows = matrix.shape[0]
    columns = 1 + n_rows
    rhs = [jnp.concatenate([-system.root_gradient[:, None], matrix[:, :g].T], axis=1)]
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        rows = matrix[:, offsets[level] : offsets[level + 1]].T.reshape(count, width, n_rows)
        rhs.append(jnp.concatenate([-system.gradient[level][..., None], rows], axis=2))
    root, root_rhs, _, _, _, factors, couplings, solved = _fold(system, damping, rhs, 1)
    if g:
        root_factor = jnp.linalg.cholesky(root + damping * jnp.eye(g, dtype=dtype))
        half = jax.scipy.linalg.solve_triangular(root_factor, root_rhs, lower=True)
        top = jax.scipy.linalg.solve_triangular(root_factor, half, lower=True, trans="T")
    else:
        root_factor = None
        top = jnp.zeros((0, columns), dtype=dtype)
    steps = [top]
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        if width == 0:
            steps.append(jnp.zeros((count, 0, columns), dtype=dtype))
            continue
        right = solved[level]
        for above in range(level):
            if not widths[above]:
                continue
            if above == 0:
                right = right - jnp.einsum("nia,ak->nik", couplings[level][0], steps[0])
            else:
                above_steps = _ancestor_values(system, steps[above], level, above)
                right = right - jnp.einsum("nia,nak->nik", couplings[level][above], above_steps)
        steps.append(_upper_solve(factors[level], right))
    stacked = jnp.concatenate(
        [steps[0]] + [steps[level].reshape(-1, columns) for level in range(1, depth + 1)], axis=0
    )
    step = stacked[:, 0]
    if n_rows:
        spread = stacked[:, 1:]
        target = system.constraint_rhs - matrix @ system.point
        gram, residual = matrix @ spread, matrix @ step - target
        if system.constraint_active is not None:
            # An inactive row takes a zero multiplier: its row and column of the
            # multipliers' system become the identity's.
            on = system.constraint_active
            identity = jnp.diag(jnp.where(on, 0.0, 1.0))
            gram = jnp.where(on[:, None] & on[None, :], gram, 0.0) + identity
            residual = jnp.where(on, residual, 0.0)
        multipliers = jnp.linalg.solve(gram, residual)
        step = step - spread @ multipliers
    parts = _split(system, step)
    halves: list = [None] * (depth + 1)
    q_squared = jnp.zeros((), dtype=dtype)
    for level in range(depth, 0, -1):
        if not widths[level]:
            continue
        right = parts[level]
        for deeper in range(level + 1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][level], halves[deeper])
            right = right - _node_sum(system, carried, deeper, level)
        halves[level] = _lower_solve(factors[level], right[..., None])[..., 0]
        q_squared = q_squared + jnp.sum(halves[level] * halves[level])
    if g:
        right = parts[0]
        for deeper in range(1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][0], halves[deeper])
            right = right - _node_sum(system, carried, deeper, 0)
        top_half = jax.scipy.linalg.solve_triangular(root_factor, right, lower=True)
        q_squared = q_squared + top_half @ top_half
    return step, q_squared


@jax.jit
def _scaled(system, scale, addition):
    # The system in the coordinates z = scale * s, with ``addition`` added to the
    # diagonal of the scaled matrix.
    inverse = 1.0 / scale
    parts = _split(system, inverse)
    added = _split(system, addition)
    shared = parts[0]
    diagonal: list = [None]
    cross: list = [None]
    gradient: list = [None]
    for level in range(1, system.depth + 1):
        own = parts[level]
        diagonal.append(
            system.diagonal[level] * own[:, :, None] * own[:, None, :]
            + jax.vmap(jnp.diag)(added[level])
        )
        gradient.append(system.gradient[level] * own)
        row = []
        for above in range(level):
            if above == 0:
                other = shared[None, None, :]
            else:
                other = _ancestor_values(system, parts[above], level, above)[:, None, :]
            row.append(system.cross[level][above] * own[:, :, None] * other)
        cross.append(tuple(row))
    return dataclasses.replace(
        system,
        root=system.root * shared[:, None] * shared[None, :] + jnp.diag(added[0]),
        root_gradient=system.root_gradient * shared,
        diagonal=tuple(diagonal),
        cross=tuple(cross),
        gradient=tuple(gradient),
        point=scale * system.point,
        constraint_matrix=system.constraint_matrix * inverse[None, :],
    )


@jax.jit
def _quadratic(system, delta):
    parts = _split(system, delta)
    total = parts[0] @ system.root @ parts[0]
    for level in range(1, system.depth + 1):
        own = parts[level]
        total = total + jnp.einsum("ni,nij,nj->", own, system.diagonal[level], own)
        for above in range(level):
            if not system.widths[above]:
                continue
            if above == 0:
                coupled = jnp.einsum("ni,nij,j->", own, system.cross[level][0], parts[0])
            else:
                other = _ancestor_values(system, parts[above], level, above)
                coupled = jnp.einsum("ni,nij,nj->", own, system.cross[level][above], other)
            total = total + 2.0 * coupled
    return total


@jax.jit
def _vectors(system):
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    diagonal = jnp.concatenate(
        [jnp.diagonal(system.root)]
        + [jnp.diagonal(entry, axis1=1, axis2=2).reshape(-1) for entry in system.diagonal[1:]]
    )
    return gradient, diagonal


# The phases of the multiplier search on the device, as ``_secular_search`` runs it.
_AT_ZERO, _PROBE, _BRACKET, _NEWTON, _FINAL, _NEARLY, _FOUND = range(7)
# The search's iteration limits: the bracket's doublings and the Newton iterations.
_BRACKETS = 80
_NEWTONS = 30


def _norm(vector: ArrayLike) -> jax.Array:
    return jnp.sqrt(jnp.dot(vector, vector))


def _all_finite(vector: ArrayLike) -> jax.Array:
    return jnp.all(jnp.isfinite(vector))


def _search(
    solve: Callable[..., tuple[jax.Array, jax.Array]],
    zero: tuple,
    radius: Real | ArrayLike,
    eps: float,
    tiny: float,
) -> tuple:
    """``_secular_search`` on the device: every branch and limit of the host generator.

    ``solve(damping)`` gives ``(z, q_squared)``; ``zero`` is its answer at
    zero damping. Returns ``(multiplier, z, solves, failed)``: ``solves``
    counts ``zero`` and every solve made, ``failed`` is true when the bracket
    found no finite step inside the radius.
    """
    z0, q0 = zero
    dtype = z0.dtype
    scalar = functools.partial(jnp.asarray, dtype=dtype)
    radius = scalar(radius)
    tolerance = scalar(np.sqrt(eps)) * jnp.maximum(scalar(1.0), radius)
    probe = scalar(np.sqrt(eps))
    one = scalar(1.0)
    tiny = scalar(tiny)
    machine = scalar(float(np.finfo(float).eps))

    def at_zero(state):
        n = _norm(state["z"])
        found = _all_finite(state["z"]) & jnp.isfinite(state["q"]) & (n <= radius)
        factor_bound = ~jnp.isfinite(state["q"])
        return {
            **state,
            "phase": jnp.where(found, _FOUND, jnp.where(factor_bound, _PROBE, _BRACKET)),
            "multiplier": jnp.zeros((), dtype),
            "out": state["z"],
            "lo": jnp.zeros((), dtype),
            "lower_is_factor_bound": factor_bound,
            "hi": one,
            "need": ~found,
            "damping": jnp.where(factor_bound, probe, one),
        }

    def at_probe(state):
        n = _norm(state["z"])
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        found = valid & (n <= radius)
        return {
            **state,
            "phase": jnp.where(found, _FOUND, _BRACKET),
            "multiplier": probe,
            "out": state["z"],
            "lo": probe,
            "lower_is_factor_bound": ~valid,
            "hi": one,
            "need": ~found,
            "damping": one,
        }

    def at_bracket(state):
        n = _norm(state["z"])
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        inside = valid & (n <= radius)
        hi = jnp.where(inside, state["hi"], 2.0 * state["hi"] + 1.0)
        count = state["brackets"] + 1
        failed = ~inside & (count >= _BRACKETS)
        return {
            **state,
            "phase": jnp.where(inside, _NEWTON, jnp.where(failed, _FOUND, _BRACKET)),
            "failed": failed,
            "hi_step": jnp.where(inside, state["z"], state["hi_step"]),
            "current": jnp.where(inside, state["hi"], state["current"]),
            "lo": jnp.where(inside, state["lo"], state["hi"]),
            "lower_is_factor_bound": jnp.where(inside, state["lower_is_factor_bound"], ~valid),
            "hi": hi,
            "brackets": count,
            "need": ~inside & ~failed,
            "damping": hi,
        }

    def at_newton(state):
        n = _norm(state["z"])
        q = state["q"]
        current, lo, hi = state["current"], state["lo"], state["hi"]
        broken = ~jnp.isfinite(n) | ~jnp.isfinite(q)
        inside = ~broken & (n <= radius)
        close = inside & (radius - n <= tolerance)
        lo = jnp.where(broken | ~inside, current, lo)
        hi = jnp.where(inside, current, hi)
        bound = jnp.where(broken, True, jnp.where(inside, state["lower_is_factor_bound"], False))
        newton = current + (n * n / jnp.maximum(q, tiny)) * ((n - radius) / radius)
        stepped = jnp.where(
            jnp.isfinite(newton) & (lo < newton) & (newton < hi), newton, 0.5 * (lo + hi)
        )
        current = jnp.where(broken, 0.5 * (lo + hi), stepped)
        more = ~close & (state["newtons"] < _NEWTONS - 1)
        return {
            **state,
            "phase": jnp.where(more, _NEWTON, _FINAL),
            "hi_step": jnp.where(inside, state["z"], state["hi_step"]),
            "current": current,
            "lo": lo,
            "hi": hi,
            "lower_is_factor_bound": bound,
            "newtons": state["newtons"] + 1,
            "need": more,
            "damping": current,
        }

    def at_final(state):
        lo, hi = state["lo"], state["hi"]
        nearly = (
            state["lower_is_factor_bound"]
            & (lo > 0.0)
            & ((hi - lo) <= 1e-7 * jnp.maximum(one, jnp.abs(lo)))
        )
        return {
            **state,
            "phase": jnp.where(nearly, _NEARLY, _FOUND),
            "multiplier": hi,
            "out": state["hi_step"],
            "need": nearly,
            "damping": lo + jnp.maximum(1e-6 * jnp.abs(lo), machine),
        }

    def at_nearly(state):
        good = _all_finite(state["z"]) & jnp.isfinite(state["q"])
        return {
            **state,
            "phase": jnp.asarray(_FOUND),
            "multiplier": jnp.where(good, state["damping"], state["multiplier"]),
            "out": jnp.where(good, state["z"], state["out"]),
            "need": jnp.asarray(False),
        }

    branches = (at_zero, at_probe, at_bracket, at_newton, at_final, at_nearly)

    def body(state):
        state = jax.lax.switch(state["phase"], branches, state)
        z, q = jax.lax.cond(
            state["need"],
            lambda: solve(state["damping"]),
            lambda: (state["z"], state["q"]),
        )
        return {**state, "z": z, "q": q, "solves": state["solves"] + state["need"]}

    zeros = jnp.zeros_like(z0)
    state = {
        "phase": jnp.asarray(_AT_ZERO),
        "z": z0,
        "q": q0,
        "damping": jnp.zeros((), dtype),
        "need": jnp.asarray(False),
        "lo": jnp.zeros((), dtype),
        "hi": one,
        "current": one,
        "lower_is_factor_bound": jnp.asarray(False),
        "hi_step": zeros,
        "out": zeros,
        "multiplier": jnp.zeros((), dtype),
        "brackets": jnp.asarray(0),
        "newtons": jnp.asarray(0),
        "solves": jnp.asarray(1),
        "failed": jnp.asarray(False),
    }
    state = jax.lax.while_loop(lambda state: state["phase"] != _FOUND, body, state)
    return state["multiplier"], state["out"], state["solves"], state["failed"]


@functools.partial(jax.jit, static_argnames=("eps", "tiny"))
def _attempt(system, scale, addition, radius, *, eps, tiny):
    # One attempt's linear algebra: the scaled system, its solve at zero damping and the
    # decrease the model predicts for that full step, then the multiplier search of
    # radius ``radius`` and the decrease predicted for its step.
    scaled = _scaled(system, scale, addition)
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    inverse = 1.0 / scale

    def solve(damping):
        return _solve(scaled, damping)

    def decrease(z):
        step = inverse * z
        quadratic = _quadratic(system, step) + jnp.sum(
            jnp.where(addition != 0.0, addition * (z * z), 0.0)
        )
        return -(gradient @ step + 0.5 * quadratic)

    zero = solve(jnp.zeros((), scale.dtype))
    finite = _all_finite(zero[0]) & jnp.isfinite(zero[1])
    full = jnp.where(finite, decrease(jnp.where(finite, zero[0], 0.0)), jnp.nan)
    multiplier, scaled_step, solves, failed = _search(solve, zero, radius, eps, tiny)
    return (
        zero[0],
        zero[1],
        full,
        inverse * scaled_step,
        decrease(scaled_step),
        multiplier,
        solves,
        failed,
    )


@functools.partial(jax.jit, static_argnames=("level",))
def _reduce(system, *, level):
    # The gradient is folded as a right-hand side column with the matrix, so the
    # reduced gradient is the outer gradient minus the inner gradients carried up
    # through the inner factors. The equality rows are not folded; the nested fit
    # refuses a row that addresses the eliminated coordinates.
    rhs = [-system.root_gradient[:, None]] + [-entry[..., None] for entry in system.gradient[1:]]
    root, root_rhs, diagonal, cross, level_rhs, _, _, _ = _fold(
        system, jnp.zeros((), dtype=system.root.dtype), rhs, level
    )
    prefix = _offsets(system.counts, system.widths)[level]
    return LevelSystem(
        root=root,
        root_gradient=-root_rhs[:, 0],
        diagonal=tuple(diagonal[:level]),
        cross=(None,) + tuple(tuple(row) for row in cross[1:level]),
        gradient=(None,) + tuple(-entry[..., 0] for entry in level_rhs[1:level]),
        point=system.point[:prefix],
        constraint_matrix=system.constraint_matrix[:, :prefix],
        constraint_rhs=system.constraint_rhs,
        cost=system.cost,
        ancestors=tuple(system.ancestors[:level]),
        descendants=tuple(system.descendants[:level]),
        present=tuple(system.present[:level]),
        fixed_mask=system.fixed_mask[:prefix],
        held_mask=tuple(system.held_mask[:level]),
        counts=tuple(system.counts[: level - 1]),
        widths=tuple(system.widths[:level]),
        constraint_active=system.constraint_active,
    )


def _added_quadratic(addition: ArrayLike | Sequence | None, scaled_step: np.ndarray) -> float:
    """``z^T diag(addition) z`` for a scaled step ``z``; 0 without an addition."""
    if addition is None:
        return 0.0
    with np.errstate(all="ignore"):
        return float(np.asarray(addition, dtype=np.float64) @ (scaled_step * scaled_step))


def scaled_system(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    addition: ArrayLike | Sequence | None = None,
) -> LevelSystem:
    """The system in the coordinates ``z = scale * s``, ``addition`` added to its diagonal.

    Its matrix is ``D H D + diag(addition)`` and its gradient ``D g``, with
    ``D = diag(1 / scale)``; with a unit scale it is the system with the
    addition on the diagonal of its matrix.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        addition: Optional ``(n,)``, nonnegative; ``None`` adds nothing.

    Returns:
        A [LevelSystem][diff_soup.optimization.inference.hierarchical._level_system.LevelSystem]
        of the same shape.
    """
    return _scaled(system, np.asarray(scale, dtype=_dtype(system)), _addition(system, addition))


def solve_damped(system: LevelSystem, damping: Real | ArrayLike) -> tuple:
    """Solve ``(H + damping I) s = -g`` along the tree, under the system's equality rows.

    Args:
        system: The system.
        damping: The nonnegative damping ``lambda``.

    Returns:
        ``(step, q_squared)``: the step ``(n,)`` in joint columns and
        ``s^T (H + damping I)^-1 s``. A matrix that is not positive definite at
        this damping gives non-finite values.
    """
    return _solve(system, np.asarray(damping, dtype=_dtype(system)))


def solve_scaled(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    damping: Real | ArrayLike,
    addition: ArrayLike | Sequence | None = None,
) -> tuple:
    """One damped solve in the coordinates ``z = scale * s``, on the host.

    The matrix is the scaled one with ``addition`` on its diagonal
    (``scaled_system``).

    Returns:
        ``(z, q_squared)`` as a host array and a float.
    """
    scaled = scaled_system(system, scale, addition)
    step, q_squared = _solve(scaled, np.asarray(damping, dtype=_dtype(system)))
    return np.asarray(step), float(q_squared)


def level_quadratic(system: LevelSystem, delta: ArrayLike | Sequence[Real]) -> float:
    """``delta^T H delta`` from the blocks, without a dense matrix.

    Args:
        system: The system.
        delta: ``(n,)``, a step in joint columns.

    Returns:
        The quadratic form as a float.
    """
    return float(_quadratic(system, np.asarray(delta, dtype=_dtype(system))))


def level_vectors(system: LevelSystem) -> tuple:
    """The gradient and the diagonal of the matrix in joint columns, as host arrays."""
    gradient, diagonal = _vectors(system)
    return np.asarray(gradient), np.asarray(diagonal)


def level_secular_step(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    radius: Real | ArrayLike,
    *,
    first: tuple[ArrayLike, Real | ArrayLike] | AttemptSolve | None = None,
    addition: ArrayLike | Sequence | None = None,
) -> tuple:
    """The trust-region step of radius ``radius`` in the scaled norm ``||scale * s||``.

    The multiplier ``lambda`` is zero when the Gauss-Newton step fits;
    otherwise it is found by a safeguarded Newton iteration on the secular
    equation ``||z(lambda)|| = radius`` above the Cholesky bound, each trial a
    damped solve along the tree in the scaled coordinates.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        first: Optional ``(z, q_squared)`` of the solve at ``lambda = 0`` in
            these coordinates, already made; it counts as this step's first
            solve.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``); the model whose step is solved
            and whose decrease is predicted carries it.

    Returns:
        ``(step, predicted, multiplier, solves)``: the step in joint columns,
        the decrease the model predicts, the multiplier, and the number of
        damped solves made.

    Raises:
        ValueError: If no finite step inside the radius is found.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    inverse_scale = 1.0 / scale
    eps = float(np.finfo(dtype).eps)
    search = _secular_search(float(radius), eps, float(np.finfo(float).tiny))
    gradient, _ = level_vectors(system)
    # Every damping the search asks for is one damped solve; the solve at zero damping
    # given as ``first`` answers its first request.
    damping = next(search)
    answer = None if first is None else (np.asarray(first[0]), float(first[1]))
    solves = 0
    try:
        while True:
            if answer is None:
                z, q_squared = _solve(scaled, np.asarray(damping, dtype=dtype))
                answer = np.asarray(z), float(q_squared)
            solves += 1
            damping, answer = search.send(answer), None
    except StopIteration as found:
        multiplier, scaled_step = found.value
    with np.errstate(all="ignore"):
        step = inverse_scale * scaled_step
    quadratic = level_quadratic(system, step) + _added_quadratic(addition, scaled_step)
    predicted = -(float(gradient @ step) + 0.5 * quadratic)
    return step, predicted, float(multiplier), solves


class AttemptSolve(NamedTuple):
    """The linear algebra of one attempt of the joint fit, from one program on the device.

    Indexed as a pair it is the solve at zero damping ``(z, q_squared)`` that
    ``solve_scaled`` gives, so it stands wherever that pair is taken.

    Attributes:
        z: The full step at zero damping in the coordinates ``z = scale * s``.
        q_squared: Its inverse-norm ``z^T (H_z)^-1 z``.
        decrease: The decrease the model predicts for the full step ``z /
            scale``, in cost units, not divided by anything; ``nan`` when the
            solve at zero damping is not finite.
        step: The step of radius ``radius`` in joint columns, as
            ``level_secular_step`` gives it.
        predicted: The decrease the model predicts for ``step``.
        multiplier: The multiplier of ``step``.
        solves: The damped solves the search made, the one at zero damping
            included.
        radius: The radius ``step`` was solved for.
        scale: The scale of the coordinates.
        addition: The diagonal added to the scaled matrix, or ``None``.
        failed: Whether the search found no finite step inside the radius
            (a point off its equality rows whose restoring step exceeds the
            radius, or a scaling no damping cures). ``z``, ``q_squared`` and
            ``decrease`` stand either way; ``step``, ``predicted`` and
            ``multiplier`` are then zero, and the loop that takes the step
            raises there, as ``level_secular_step`` does, so a fit that ends
            before it takes a step is not ended by the search.
    """

    z: np.ndarray
    q_squared: float
    decrease: float
    step: np.ndarray
    predicted: float
    multiplier: float
    solves: int
    radius: float
    scale: np.ndarray
    addition: np.ndarray | None
    failed: bool = False

    def answers(
        self,
        scale: ArrayLike | Sequence[Real],
        radius: Real | ArrayLike,
        addition: ArrayLike | Sequence | None = None,
    ) -> bool:
        """Whether this attempt was solved for ``scale``, ``radius`` and ``addition``."""
        if float(radius) != self.radius or not np.array_equal(np.asarray(scale), self.scale):
            return False
        if addition is None or self.addition is None:
            return addition is None and self.addition is None
        return np.array_equal(np.asarray(addition), self.addition)


def attempt_solve(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    radius: Real | ArrayLike,
    addition: ArrayLike | Sequence | None = None,
) -> AttemptSolve:
    """One attempt's solve at zero damping, its predicted decrease and its trust-region step.

    The same numbers as ``solve_scaled(system, scale, 0.0, addition)``, the
    full step's predicted decrease, and ``level_secular_step(system, scale,
    radius, first=..., addition=addition)``, from one compiled program per
    system shape and one transfer to the host: the multiplier search runs on
    the device, branch for branch as on the host. The numbers agree with that
    chain to rounding.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix.

    Returns:
        The [AttemptSolve][diff_soup.optimization.inference.hierarchical._level_solve.AttemptSolve].
        A search that finds no finite step inside the radius sets its
        ``failed``; nothing is raised here, so the solve at zero damping and
        its predicted decrease serve the tests a loop makes before it takes a
        step.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    added = _addition(system, addition)
    outputs = _attempt(
        system,
        scale,
        added,
        np.asarray(radius, dtype=dtype),
        eps=float(np.finfo(dtype).eps),
        tiny=float(np.finfo(float).tiny),
    )
    z, q_squared, decrease, step, predicted, multiplier, solves, failed = jax.device_get(outputs)
    return AttemptSolve(
        z=np.asarray(z),
        q_squared=float(q_squared),
        decrease=float(decrease),
        step=np.array(step),
        predicted=float(predicted),
        multiplier=float(multiplier),
        solves=int(solves),
        radius=float(radius),
        scale=scale,
        addition=None if addition is None else added,
        failed=bool(failed),
    )


def _secular_search(radius: float, eps: float, tiny: float):
    """The multiplier search of one trust-region problem of radius ``radius``, as a generator.

    It yields the damping of the next solve and receives ``(z, q_squared)``,
    the problem's scaled step at that damping and its inverse-norm; it returns
    ``(multiplier, z)``. ``level_secular_step`` runs one for the whole system,
    ``_lockstep_search`` one per member of a level.
    """

    def norm(vector: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return float(np.linalg.norm(vector))

    def finite(vector: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(vector)))

    scaled_step, zero_q_squared = yield 0.0
    if finite(scaled_step) and np.isfinite(zero_q_squared) and norm(scaled_step) <= radius:
        return 0.0, scaled_step

    lo = 0.0
    lower_is_factor_bound = not np.isfinite(zero_q_squared)
    if lower_is_factor_bound:
        probe = float(np.sqrt(eps))
        probe_step, probe_q_squared = yield probe
        probe_norm = norm(probe_step)
        probe_valid = np.isfinite(probe_norm) and np.isfinite(probe_q_squared)
        if probe_valid and probe_norm <= radius:
            return probe, probe_step
        lo = probe
        lower_is_factor_bound = not probe_valid
    hi = 1.0
    for _ in range(80):
        candidate, q_squared = yield hi
        candidate_norm = norm(candidate)
        valid_factor = np.isfinite(candidate_norm) and np.isfinite(q_squared)
        if valid_factor and candidate_norm <= radius:
            hi_step = candidate
            break
        lo = hi
        lower_is_factor_bound = not valid_factor
        hi = 2.0 * hi + 1.0
    else:
        raise ValueError(
            "TrustRegion could not bracket a finite Gauss-Newton trust-region step; improve "
            "parameter scaling."
        )

    current = hi
    tolerance = float(np.sqrt(eps)) * max(1.0, radius)
    for iteration in range(30):
        candidate_norm = norm(candidate)
        if not np.isfinite(candidate_norm) or not np.isfinite(q_squared):
            lo = current
            lower_is_factor_bound = True
            current = 0.5 * (lo + hi)
            if iteration < 29:
                candidate, q_squared = yield current
            continue
        elif candidate_norm <= radius:
            hi = current
            hi_step = candidate
            if radius - candidate_norm <= tolerance:
                break
        else:
            lo = current
            lower_is_factor_bound = False
        newton = current + (candidate_norm * candidate_norm / max(q_squared, tiny)) * (
            (candidate_norm - radius) / radius
        )
        current = newton if np.isfinite(newton) and lo < newton < hi else 0.5 * (lo + hi)
        if iteration < 29:
            candidate, q_squared = yield current

    multiplier = hi
    scaled_step = hi_step
    if lower_is_factor_bound and lo > 0.0 and (hi - lo) <= 1e-7 * max(1.0, abs(lo)):
        nearly_exact = lo + max(1e-6 * abs(lo), float(np.finfo(float).eps))
        nearly_exact_step, nearly_exact_q_squared = yield nearly_exact
        if finite(nearly_exact_step) and np.isfinite(nearly_exact_q_squared):
            multiplier = nearly_exact
            scaled_step = nearly_exact_step
    return multiplier, scaled_step


def reduce_to(system: LevelSystem, level: int) -> LevelSystem:
    """The system over the levels above ``level``, the levels ``level .. D`` eliminated.

    Its blocks are the Schur complement of the levels ``level .. D`` into the
    blocks above them (no damping). Its gradient is the outer part of the
    gradient with the inner gradients folded in through the same elimination,
    ``g_o - W V^-1 g_i`` (``V`` the eliminated blocks, ``W`` their couplings to
    the outer blocks): the gradient, at a zero outer step, of the quadratic
    model minimized over the eliminated coordinates. Where the inner gradients
    are zero it is the outer part of the gradient. It keeps the outer part of
    the point, the fixed outer coordinates and the outer columns of the equality
    rows, which is exact when no row addresses the eliminated coordinates.

    Args:
        system: The system.
        level: The cut, ``1 .. D``.

    Returns:
        A [LevelSystem][diff_soup.optimization.inference.hierarchical._level_system.LevelSystem]
        of depth ``level - 1`` (the root alone for ``level = 1``).

    Raises:
        ValueError: If ``level`` is not a level of the system.
    """
    level = int(level)
    if not 1 <= level <= system.depth:
        raise ValueError(f"reduce_to cuts at a level 1 .. {system.depth}; got {level}.")
    return _reduce(system, level=level)


__all__ = [
    "AttemptSolve",
    "attempt_solve",
    "level_quadratic",
    "level_secular_step",
    "level_vectors",
    "reduce_to",
    "scaled_system",
    "solve_damped",
    "solve_scaled",
]
