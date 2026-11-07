"""Derivatives of the phased primitives: their tangents, the solve's transpose and its remat rule."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._jax_compat import (
    Var,
    ad,
    internal_partial_eval,
)
from diff_soup.linear._factor import FactorContractError
from diff_soup.linear._phased import (
    ORDER_DTYPE,
    _backend_label,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)
from diff_soup.linear._phased_abstract import _solve_flags, _split_solve_operands


# --------------------------------------------------------------------------- #
# The pattern matvec
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=256)
def _pattern_indices(topology: Any) -> tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Host index arrays of a topology's entries outside its Dirichlet rows.

    Returns ``(nnz, rows, cols, keep)``: the pattern length, the rows and
    columns of the kept entries, and the kept entry positions (``None`` when
    the topology has no Dirichlet rows and every entry is kept). The arrays
    are read-only host numpy, never ``jnp``, so the cache holds no tracer.
    """
    rows = np.asarray(topology.rows, dtype=np.int32)
    cols = np.asarray(topology.cols, dtype=np.int32)
    nnz = int(rows.shape[0])
    bc = topology.bc_indices
    keep = None
    if bc is not None and int(np.asarray(bc).size) > 0:
        keep = np.flatnonzero(~np.isin(rows, np.asarray(bc))).astype(np.int32)
        rows, cols = rows[keep], cols[keep]
        keep.flags.writeable = False
    rows = np.array(rows, copy=True)
    cols = np.array(cols, copy=True)
    rows.flags.writeable = False
    cols.flags.writeable = False
    return nnz, rows, cols, keep


def pattern_matvec(analysis: Any, dvalues: Any, x: Any, *, transpose: bool) -> jax.Array:
    """Apply the operator whose pattern entries are ``dvalues`` to ``x``.

    Computes ``D x`` (or ``D^T x`` with ``transpose=True``), where ``D`` has
    the entries ``dvalues`` on ``analysis.topology`` with every entry of a
    Dirichlet row dropped, which is the derivative of the assembled operator
    ``A_bc`` along a value tangent ``dvalues``. With ``dvalues`` set to the
    values themselves it is ``A_bc x`` without the Dirichlet rows' unit
    diagonal. The index arrays are host numpy taken from the topology, so the
    function is linear in ``dvalues`` and in ``x`` and differentiates and
    batches like any ``jnp`` code.

    A backend whose values describe more than the pattern's entries (a
    low-rank term ``U W^T`` whose entries of ``U`` follow the base values)
    defines a method ``pattern_matvec(analysis, dvalues, x, *, transpose)``
    with this contract, linear in ``dvalues`` and ``x``; it is used in place
    of the pattern's gather and scatter, so derivatives and residuals cover
    the whole operator.

    Args:
        analysis: An analysis carrying the ``topology``.
        dvalues: Entry values of shape ``(*lanes, nnz)``.
        x: Vector ``(*lanes, n)`` or block ``(*lanes, nrhs, n)``.
        transpose: Apply ``D^T`` instead of ``D``.

    Returns:
        An array of the shape of ``x``, in the promoted dtype of ``dvalues``
        and ``x``.

    Raises:
        FactorContractError: If the shapes do not match the topology.
    """
    backend_matvec = getattr(getattr(analysis, "backend", None), "pattern_matvec", None)
    if backend_matvec is not None:
        return backend_matvec(analysis, dvalues, x, transpose=transpose)
    topology = analysis.topology
    n = int(topology.n_dofs)
    nnz, rows, cols, keep = _pattern_indices(topology)
    dvalues = jnp.asarray(dvalues)
    x = jnp.asarray(x)
    if dvalues.ndim < 1 or dvalues.shape[-1] != nnz:
        raise FactorContractError(
            f"pattern_matvec needs values of shape (*lanes, {nnz}), one per pattern entry; "
            f"got shape {list(dvalues.shape)}."
        )
    lanes = tuple(dvalues.shape[:-1])
    block = x.ndim - len(lanes)
    if tuple(x.shape[: len(lanes)]) != lanes or block not in (1, 2) or x.shape[-1] != n:
        raise FactorContractError(
            f"pattern_matvec needs x of shape (*lanes, {n}) or (*lanes, nrhs, {n}) with the "
            f"values' lanes {list(lanes)}; got shape {list(x.shape)}."
        )
    coef = dvalues if keep is None else dvalues[..., keep]
    src, dst = (rows, cols) if transpose else (cols, rows)
    if block == 2:
        coef = coef[..., None, :]
    contributions = coef * x[..., src]
    out = jnp.zeros(contributions.shape[:-1] + (n,), contributions.dtype)
    return out.at[..., dst].add(contributions)


# --------------------------------------------------------------------------- #
# Differentiation
# --------------------------------------------------------------------------- #
def _is_zero(tangent: Any) -> bool:
    """Whether a tangent is symbolically zero (or of the zero-size ``float0`` type)."""
    return type(tangent) is ad.Zero or getattr(tangent, "dtype", None) == jax.dtypes.float0


def _zero_tangents(outs: Sequence[Any]) -> list:
    """Symbolic zero tangents for primal outputs."""
    return [ad.Zero(jax.typeof(out).to_tangent_aval()) for out in outs]


def _refuse_value_tangent(name: str, analysis: Any) -> None:
    """Raise for a value tangent of a factorization that is not a symbolic zero."""
    raise FactorContractError(
        f"The tangent of the matrix values of {name} on the {_backend_label(analysis)} "
        "backend must be a symbolic zero; it is a traced or instantiated array. A factor is "
        "differentiated through the solves that use it, never through the factorization: "
        "bind the factorization on jax.lax.stop_gradient(values) and pass the "
        "differentiated values to the solve."
    )


def _factor_jvp(primals, tangents, **params):
    """Zero tangents; a nonzero value tangent is refused."""
    if not _is_zero(tangents[0]):
        _refuse_value_tangent("phased_factor", params["analysis"])
    outs = phased_factor_p.bind(*primals, **params)
    return outs, _zero_tangents(outs)


def _refactor_jvp(primals, tangents, **params):
    """``select(predicate, dvalues, dheld_values)`` for the kept values; zero elsewhere.

    The primal outputs come from binding the primals, so the factorization
    runs once. The kept values are ``values`` where the predicate holds and
    ``held_values`` elsewhere, so their tangent selects between the two value
    tangents; both symbolically zero gives a zero tangent. The handle, the
    status and the order word have none. The select is linear in both
    tangents, so reverse mode transposes it without a rule of the refactor's
    own and higher orders compose.
    """
    values, _, predicate = primals[:3]
    dvalues, dheld = tangents[0], tangents[1]
    outs = phased_refactor_p.bind(*primals, **params)
    out_tangents = _zero_tangents(outs)
    if not getattr(params["analysis"].backend, "refactor_keeps_values", True):
        # A backend whose factor preconditions a current operator keeps the new
        # values in every lane, so they carry the tangent of the new values.
        if not _is_zero(dvalues):
            out_tangents[0] = ad.instantiate_zeros(dvalues)
        return outs, out_tangents
    if _is_zero(dvalues) and _is_zero(dheld):
        return outs, out_tangents
    dvalues, dheld = ad.instantiate_zeros(dvalues), ad.instantiate_zeros(dheld)
    mask = jnp.broadcast_to(jnp.expand_dims(predicate, -1), jnp.shape(values))
    out_tangents[0] = jax.lax.select(mask, dvalues, dheld)
    return outs, out_tangents


def _invalidate_jvp(primals, tangents, **params):
    """The invalidation on the primal handle; its order word has a zero tangent."""
    del tangents
    outs = phased_invalidate_p.bind(*primals, **params)
    return outs, _zero_tangents(outs)


def _derivative_params(params: dict) -> dict:
    """The parameters of a derivative solve: no initial guess, and no shared-storage trust.

    A derivative solve's order result is dropped (a tangent is not part of
    the primal order chain), so no later operation waits for it; it applies
    the rule of a factor of its own storage, which stays exact whatever runs
    on the storage before it.
    """
    return (
        dict(params, has_x0=False, storage=None)
        if "has_x0" in params
        else dict(params, storage=None)
    )


def _solve_jvp(primals, tangents, **params):
    """``dx = solve(db - dM x)`` on the same handle; the status, the counts and the order word have no tangent.

    The tangents of an inexact solve's ``weights`` and ``x0`` contribute
    nothing: the solve is differentiated as the exact inverse of ``M``. The
    tangent solve takes the order word the primal solve took, binds the same
    ``weights``, no initial guess and no shared storage, and its own order
    result is dropped.
    """
    n_handle = params["n_handle"]
    has_weights, has_x0 = _solve_flags(params)
    values, *operands = primals
    handle, order, b, weights, _ = _split_solve_operands(
        "phased_solve", operands, n_handle, has_weights, has_x0
    )
    dvalues, db = tangents[0], tangents[n_handle + 2]
    outs = list(phased_solve_p.bind(*primals, **params))
    x = outs[0]
    rest_dots = _zero_tangents(outs[1:])
    values_zero, b_zero = _is_zero(dvalues), _is_zero(db)
    if values_zero and b_zero:
        (x_dot,) = _zero_tangents([x])
        return outs, [x_dot, *rest_dots]
    rhs = None if b_zero else db
    if not values_zero:
        effective = (params["orientation"] == "AT") != bool(params["transpose"])
        dm_x = pattern_matvec(params["analysis"], dvalues, x, transpose=effective)
        # x has the result dtype, which is complex whenever the values or b
        # are, so this cast never drops an imaginary part; it brings dM x to
        # the rung's width when the values are wider than the rung (float64
        # values on an fp32 factor), which is the dtype db and the bind need.
        dm_x = dm_x.astype(x.dtype)
        rhs = -dm_x if rhs is None else rhs - dm_x
    extras = (weights,) if has_weights else ()
    x_dot = phased_solve_p.bind(values, *handle, order, rhs, *extras, **_derivative_params(params))[
        0
    ]
    return outs, [x_dot, *rest_dots]


def _solve_transpose(cts, values, *operands, **params):
    """The cotangent of ``b``: a solve with ``transpose`` flipped on the cotangent of ``x``.

    The flipped solve takes the order word of the forward solve (a residual
    in reverse mode) and its own order result is dropped. When that order
    word is itself an output of a linear operation (the forward solve
    followed another solve on a linear right-hand side, as the correction
    solves of a refined tangent do), it has no value in the backward pass:
    JAX counts every output of an equation with a linear input as linear.
    The flipped solve then takes a zero order word. Its value is never read
    and a derivative solve never trusts shared storage, so it stays exact; it
    is ordered after the operations that produced its handle and its
    cotangent. An inexact solve's flipped bind keeps ``weights`` and drops
    ``x0``; neither receives a cotangent.
    """
    has_weights, has_x0 = _solve_flags(params)
    handle, order, b, weights, x0 = _split_solve_operands(
        "phased_solve", operands, params["n_handle"], has_weights, has_x0
    )
    if (
        ad.is_undefined_primal(values)
        or any(ad.is_undefined_primal(leaf) for leaf in handle)
        or not ad.is_undefined_primal(b)
        or any(extra is not None and ad.is_undefined_primal(extra) for extra in (weights, x0))
    ):
        raise FactorContractError(
            "phased_solve is linear in its right-hand side only: it transposes with respect "
            "to b, with the values, the handle, the weights and the initial guess known."
        )
    if ad.is_undefined_primal(order):
        order = jnp.zeros(tuple(order.aval.shape), ORDER_DTYPE)
    x_bar = cts[0]
    if type(x_bar) is ad.Zero:
        b_bar = ad.Zero(b.aval.to_tangent_aval())
    else:
        # b is bound in the result dtype, which x and its cotangent share, so
        # the flipped solve returns the cotangent of b in b's dtype. A caller
        # that cast b on entry gets the conversion back through that cast.
        # The flipped solve never trusts shared storage: reverse mode solves
        # with the factors of every step after later refactors of the storage,
        # so it applies the rule of a factor of its own storage.
        flipped = dict(_derivative_params(params), transpose=not params["transpose"])
        extras = (weights,) if has_weights else ()
        b_bar = phased_solve_p.bind(values, *handle, order, x_bar, *extras, **flipped)[0]
    return [None] * (1 + len(handle)) + [None, b_bar] + [None] * (int(has_weights) + int(has_x0))


# --------------------------------------------------------------------------- #
# Rematerialisation
# --------------------------------------------------------------------------- #
def _saves_outputs(eqn: Any) -> bool:
    """Whether rematerialisation keeps ``eqn`` known and saves its outputs.

    Every factorization, refactorization and invalidation, and every solve on
    a backend whose ``factor_residency`` is not ``"arrays"``: recomputing
    one would factor again, or solve with native storage that later
    operations have since rewritten.
    """
    if eqn.primitive is not phased_solve_p:
        return True
    backend = eqn.params["analysis"].backend
    return getattr(backend, "factor_residency", "arrays") != "arrays"


def _remat_rule(saveable, unks_in, inst_in, eqn):
    """The partial-evaluation rule of ``jax.checkpoint`` for the phased primitives.

    An equation with an unknown operand is staged with its known operands
    as residuals, as any equation is. An equation whose operands are all
    known runs on the known side; when
    [_saves_outputs][diff_soup.linear._phased_ad._saves_outputs] holds its
    outputs are saved (never recomputed), and otherwise (a solve on
    array-held factors) the rematerialisation policy decides, as for any
    equation.
    """
    n_out = len(eqn.outvars)
    residuals = [var for var, inst in zip(eqn.invars, inst_in) if isinstance(var, Var) and not inst]
    if any(unks_in):
        return None, eqn, [True] * n_out, [True] * n_out, residuals
    if not _saves_outputs(eqn):
        policy = internal_partial_eval.ensure_enum(
            saveable(eqn.primitive, *[var.aval for var in eqn.invars], **eqn.params)
        )
        if isinstance(policy, internal_partial_eval.RecomputeType):
            return eqn, eqn, [False] * n_out, [True] * n_out, residuals
    return eqn, None, [False] * n_out, [False] * n_out, []
