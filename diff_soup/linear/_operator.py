"""Linear operators given by their action, the matrix-free entry of the linear-solver contract.

A [LinearOperator][diff_soup.linear.LinearOperator] describes a square
operator ``M`` by what it does to a vector: ``mv(x) = M x`` and, when the
transposed action is available, ``rmv(x) = M^T x``, together with its static
``shape``, ``dtype`` and structure ``tags``. An iterative (inexact) backend
needs exactly that, and two constructors build one:

* [operator_from_values][diff_soup.linear.operator_from_values] from the
  assembled values of an [Analysis][diff_soup.linear.Analysis]'s pattern,
  with the topology's Dirichlet rows applied exactly as every factor applies
  them;
* [operator_from_jvp][diff_soup.linear.operator_from_jvp] from a residual
  function and a point: ``mv`` is the residual's Jacobian-vector product at
  that point and ``rmv`` its vector-Jacobian product, so the Jacobian is
  never assembled.

An operator is a trace-time object. It is built where it is used (inside the
JAX-level solve of an inexact backend, from the values that solve receives,
or around a residual inside a traced program) and is never a leaf of a
[Factor][diff_soup.linear.Factor] and never an operand of the contract's
primitives: derivatives of a solve flow through the factor's values and the
right-hand side only, never through an operator's closures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from diff_soup.linear._factor import Analysis, FactorContractError, dirichlet_rows
from diff_soup.linear._primitive import pattern_matvec

#: The structure tags an operator may carry.
OPERATOR_TAGS: frozenset[str] = frozenset({"square", "symmetric", "spd", "nonsymmetric"})


def _check_tags(tags: Any, shape: tuple[int, int]) -> frozenset[str]:
    """``tags`` as a frozenset of known, mutually consistent tags for an operator of ``shape``."""
    if isinstance(tags, str):
        tags = (tags,)
    try:
        tags = frozenset(tags)
    except TypeError as exc:
        raise FactorContractError(
            f"LinearOperator tags must be an iterable of strings from {sorted(OPERATOR_TAGS)}; "
            f"got {tags!r}."
        ) from exc
    unknown = tags - OPERATOR_TAGS
    if unknown:
        raise FactorContractError(
            f"LinearOperator tags {sorted(unknown)} are unknown; the tags are "
            f"{sorted(OPERATOR_TAGS)}."
        )
    if "spd" in tags:
        tags = tags | {"symmetric"}
    if {"symmetric", "nonsymmetric"} <= tags:
        raise FactorContractError(
            "LinearOperator tags cannot hold both 'symmetric' and 'nonsymmetric'; keep the one "
            "that describes the operator."
        )
    structural = tags & {"square", "symmetric", "spd"}
    if structural and shape[0] != shape[1]:
        raise FactorContractError(
            f"LinearOperator tags {sorted(structural)} need a square shape; got shape {shape}."
        )
    return tags


class LinearOperator(eqx.Module):
    """A matrix given by its action, for the iterative solves.

    Args:
        mv: The function `x -> M x`.
        rmv: The function `x -> M^T x`, or `None` when no transpose is needed.
        shape: The shape, `(rows, columns)`.
        dtype: The dtype of the entries.
        tags: Structure tags from `"square"`, `"symmetric"`, `"spd"` and `"nonsymmetric"`.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    tags: frozenset = eqx.field(static=True)
    mv: Callable[[Array], Array]
    rmv: Optional[Callable[[Array], Array]]

    def __init__(
        self,
        mv: Callable[[Array], Array],
        rmv: Optional[Callable[[Array], Array]],
        *,
        shape: tuple[int, int],
        dtype: Any,
        tags: Iterable[str] = ("square", "nonsymmetric"),
    ):
        """Build an operator from its actions and its static description.

        Args:
            mv: The callable ``x -> M x``.
            rmv: The callable ``x -> M^T x``, or ``None`` when the transposed
                action is not available.
            shape: ``(rows, columns)`` of ``M``.
            dtype: The dtype of ``M``'s entries.
            tags: Structure tags from ``"square"``, ``"symmetric"``,
                ``"spd"`` and ``"nonsymmetric"``; ``"spd"`` adds
                ``"symmetric"``.

        Raises:
            FactorContractError: If ``mv`` (or a given ``rmv``) is not
                callable, ``shape`` is not two non-negative integers, or the
                tags are unknown or contradict each other or the shape.
        """
        if not callable(mv) or (rmv is not None and not callable(rmv)):
            raise FactorContractError(
                "LinearOperator needs callables mv(x) = M x and rmv(x) = M^T x (rmv may be "
                f"None); got mv={type(mv).__name__}, rmv={type(rmv).__name__}."
            )
        try:
            rows, columns = (int(side) for side in shape)
        except (TypeError, ValueError) as exc:
            raise FactorContractError(
                f"LinearOperator shape must be two integers (rows, columns); got {shape!r}."
            ) from exc
        if rows < 0 or columns < 0:
            raise FactorContractError(
                f"LinearOperator shape must be non-negative; got {(rows, columns)}."
            )
        self.shape = (rows, columns)
        self.dtype = np.dtype(dtype)
        self.tags = _check_tags(tags, self.shape)
        self.mv = mv
        self.rmv = rmv

    def apply(self, x: ArrayLike, *, transpose: bool = False) -> Array:
        """Apply the operator, `M x`, or `M^T x` with `transpose=True`.

        Args:
            x: The vector to apply it to.
            transpose: Apply the transpose instead.

        Returns:
            The product.

        Raises:
            FactorContractError: For `transpose=True` when `rmv` is `None`.
        """
        if not transpose:
            return self.mv(x)
        if self.rmv is None:
            raise FactorContractError(
                "This LinearOperator has no transposed action (rmv is None), so M^T x cannot be "
                "applied; build it with an rmv, for example with operator_from_values or "
                "operator_from_jvp."
            )
        return self.rmv(x)


def apply_assembled(analysis: Analysis, values: Array, x: Array, *, transpose: bool) -> Array:
    """``A_bc x`` (or ``A_bc^T x``) for the values on an analysis's pattern.

    ``A_bc`` is the operator ``values`` describe on ``analysis.topology``
    (duplicate entries add) with every entry of a Dirichlet row replaced by
    zero and ``1`` on that row's diagonal, the operator every factor of the
    pattern represents. The result is linear in ``values`` and in ``x`` and
    differentiates and batches like ``jnp`` code; the values are used in
    their own dtype.

    Args:
        analysis: The [Analysis][diff_soup.linear.Analysis] of the pattern.
        values: Entry values ``(*lanes, nnz)``.
        x: Vector ``(*lanes, n)`` or block ``(*lanes, nrhs, n)``.
        transpose: Apply ``A_bc^T`` instead of ``A_bc``.

    Returns:
        The image, with the shape of ``x``.
    """
    applied = pattern_matvec(analysis, values, x, transpose=transpose)
    _, diagonal = dirichlet_rows(analysis.topology)
    if diagonal.size:
        applied = applied.at[..., diagonal].add(x[..., diagonal])
    return applied


def operator_from_values(
    analysis: Analysis, values: ArrayLike, *, transpose: bool = False
) -> LinearOperator:
    """The operator of an assembled matrix, from its analysis and values.

    Its structure tags follow the `symmetric` and `spd` declarations of the analysis.

    Args:
        analysis: The `Analysis` of the pattern.
        values: One value per pattern entry, shape `(nnz,)`, or `(batch, nnz)`.
        transpose: Build the operator of the transpose instead.

    Returns:
        The `LinearOperator`.
    """
    if not isinstance(analysis, Analysis):
        raise FactorContractError(
            "operator_from_values() needs a diff_soup.linear.Analysis built by "
            f"diff_soup.linear.analyze(); got {type(analysis).__name__}."
        )
    values = jnp.asarray(values)
    if values.ndim not in (1, 2) or values.shape[-1] != analysis.nnz:
        raise FactorContractError(
            f"operator_from_values() needs values of shape ({analysis.nnz},), one per pattern "
            f"entry, with at most one leading lane axis; got shape {tuple(values.shape)}."
        )
    transpose = bool(transpose)
    if analysis.spd:
        tags = ("square", "spd", "symmetric")
    elif analysis.symmetric:
        tags = ("square", "symmetric")
    else:
        tags = ("square", "nonsymmetric")

    def mv(x: Array) -> Array:
        return apply_assembled(analysis, values, x, transpose=transpose)

    def rmv(x: Array) -> Array:
        return apply_assembled(analysis, values, x, transpose=not transpose)

    n = analysis.n
    return LinearOperator(mv, rmv, shape=(n, n), dtype=values.dtype, tags=tags)


def operator_from_jvp(
    residual: Callable[[Array], Array],
    primal: ArrayLike,
    *,
    shape: Optional[tuple[int, int]] = None,
    dtype: Any = None,
    tags: Optional[Iterable[str]] = None,
) -> LinearOperator:
    """The Jacobian of `residual` at `primal`, as a matrix-free operator.

    It linearizes `residual` once, and applying the operator never evaluates `residual` again.
    The Jacobian is never assembled.

    Args:
        residual: A function from a vector `(n,)` to a vector `(m,)`.
        primal: The point `(n,)` where the Jacobian is taken.
        shape: The expected shape `(m, n)`, checked against the residual, or `None`.
        dtype: The dtype of the entries, by default the residual's output dtype.
        tags: Structure tags, by default `("square", "nonsymmetric")` for a square Jacobian.

    Returns:
        The `LinearOperator`.

    Raises:
        FactorContractError: If `primal` or the output is not a vector, or `shape` disagrees.
    """
    primal = jnp.asarray(primal)
    if primal.ndim != 1:
        raise FactorContractError(
            f"operator_from_jvp() needs a vector primal (n,); got shape {tuple(primal.shape)}."
        )
    output, mv = jax.linearize(residual, primal)
    if jnp.ndim(output) != 1:
        raise FactorContractError(
            "operator_from_jvp() needs a residual that maps a vector (n,) to a vector (m,); it "
            f"returned shape {tuple(jnp.shape(output))}."
        )
    found = (int(jnp.shape(output)[0]), int(primal.shape[0]))
    if shape is not None and tuple(int(side) for side in shape) != found:
        raise FactorContractError(
            f"operator_from_jvp() was given shape {tuple(shape)}, but the residual maps a "
            f"vector of length {found[1]} to one of length {found[0]}: its Jacobian has shape "
            f"{found}."
        )
    _, pullback = jax.vjp(residual, primal)

    def rmv(cotangent: Array) -> Array:
        (pulled,) = pullback(cotangent)
        return pulled

    if tags is None:
        tags = ("square", "nonsymmetric") if found[0] == found[1] else ()
    return LinearOperator(
        mv,
        rmv,
        shape=found,
        dtype=jnp.result_type(output) if dtype is None else dtype,
        tags=tags,
    )


__all__ = [
    "LinearOperator",
    "OPERATOR_TAGS",
    "apply_assembled",
    "operator_from_jvp",
    "operator_from_values",
]
