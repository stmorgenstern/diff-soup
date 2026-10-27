"""Coefficient tables and backward-difference-table operations for the NDF engine.

The method coefficients come from the published NDF family (Shampine &
Reichelt 1997, SIAM J. Sci. Comput. 18(1)): the harmonic-sum ``gamma``
vector, the ``kappa`` correction coefficients, the per-order local-error
constants, and the leading coefficient ``beta0``. The difference-table
machinery — the signed-binomial change-of-step matrices and the
accepted-step update recurrence — follows the QNDF solver of
OrdinaryDiffEq.jl (MIT license; see ``THIRD_PARTY_NOTICES.md``), which in
turn implements the adaptive-BDF formulation of Alberdi Celaya, Anza
Aguirrezabala & Chatzipantelidis (2014).

Table convention: ``D`` has shape ``(D_ROWS, n)`` with ``D[0]`` the state
itself at the last accepted point and ``D[j]`` its ``j``-th backward
difference on the current uniform step grid, ``j = 1..k+2`` live at order
``k``. Storing the state as row 0 folds the accepted-step state advance
into the same telescoping update as the differences (the identity
``∇ʲy_{n+1} = ∇ʲy_n + ∇ʲ⁺¹y_{n+1}`` applied down to ``j = 0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MAX_ORDER = 5
"""The maximum integration order of the NDF family (orders 1..5)."""

D_ROWS = MAX_ORDER + 3
"""Rows of the difference table: state + differences ``1..k+2`` at ``k = 5``."""

# --- published method coefficients (Shampine & Reichelt 1997) --------------
# Index by order k (index 0 unused). gamma_k = sum_{j=1..k} 1/j.
# The tables are HOST NUMPY, not jnp: the engines are imported lazily by the
# solver entry, so this module may first load inside an active jit trace,
# where any module-level jax operation (including ``jnp.asarray``'s
# device_put) is staged as a tracer and leaks into the global. Convert with
# ``jnp.asarray`` at traced use sites (in particular before traced indexing).
GAMMA = np.asarray(
    [0.0] + [float(sum(1.0 / j for j in range(1, k + 1))) for k in range(1, MAX_ORDER + 1)]
)
"""Harmonic partial sums ``gamma_k``, index by order (index 0 unused)."""

KAPPA = np.asarray([0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0])
"""The published NDF ``kappa`` correction coefficients, index by order."""

ERROR_CONST = KAPPA * GAMMA + np.asarray([1.0] + [1.0 / (k + 1) for k in range(1, MAX_ORDER + 1)])
"""Per-order local-error constants ``kappa_k * gamma_k + 1/(k+1)``."""

BETA0 = np.asarray([1.0] + [float(1.0 / ((1.0 - k_) * g_)) for k_, g_ in zip(KAPPA[1:], GAMMA[1:])])
"""The leading coefficient ``beta0_k = 1/((1 - kappa_k) * gamma_k)``; the
corrector's iteration-matrix scalar is ``1/(dt * beta0_k)``. Index 0 unused."""


def _signed_binomial(rho) -> jax.Array:
    """The ``(MAX_ORDER, MAX_ORDER)`` signed-binomial recurrence matrix.

    Row ``j``/column ``r`` (0-based) is built by ``M[0, r] = -(r+1)*rho`` and
    ``M[j, r] = M[j-1, r] * (j - (r+1)*rho) / (j+1)`` — the change-of-grid
    matrix for backward-difference tables (QNDF ``calc_R``; ``rho = 1``
    gives the self-inverse ``U`` matrix, QNDF ``U!``).
    """
    rho = jnp.asarray(rho)
    r_idx = jnp.arange(1, MAX_ORDER + 1, dtype=rho.dtype)
    rows = [-r_idx * rho]
    for j in range(1, MAX_ORDER):
        rows.append(rows[j - 1] * (j - r_idx * rho) / (j + 1))
    return jnp.stack(rows)


def _signed_binomial_host() -> np.ndarray:
    """The rho = 1 signed-binomial matrix in host numpy.

    Twin of
    [_signed_binomial][diff_soup.timestepping.vsvo.engines.ndf.tables._signed_binomial]
    at ``rho = 1`` (the self-inverse ``U``
    matrix), computed without jnp so a first import inside a jit trace cannot
    leak a tracer into the module global.
    """
    r_idx = np.arange(1.0, MAX_ORDER + 1.0)
    rows = [-r_idx]
    for j in range(1, MAX_ORDER):
        rows.append(rows[j - 1] * (j - r_idx) / (j + 1))
    return np.stack(rows)


_U = _signed_binomial_host()


def difference_rescale_matrix(rho: jax.Array, k: jax.Array) -> jax.Array:
    """The masked ``R(rho) @ U`` matrix that moves differences onto a new step.

    Acts on the difference block (rows ``1..MAX_ORDER`` of ``D``). Entries
    with row or column index ``>= k`` are replaced by the identity so orders
    above the active one pass through unchanged; ``k`` may be traced.
    """
    rho = jnp.asarray(rho)
    ru = _signed_binomial(rho).astype(rho.dtype) @ _U.astype(rho.dtype)
    idx = jnp.arange(MAX_ORDER)
    active = (idx[:, None] < k) & (idx[None, :] < k)
    eye = jnp.eye(MAX_ORDER, dtype=ru.dtype)
    return jnp.where(active, ru, eye)


def rescale_d(d: jax.Array, rho: jax.Array, k: jax.Array) -> jax.Array:
    """Move the difference table onto step ``rho * dt``.

    ``D[0]`` (the state) is grid-independent and untouched; difference rows
    ``1..k`` transform by the signed-binomial matrix, rows above ``k`` pass
    through.
    """
    m = difference_rescale_matrix(rho, k).astype(d.dtype)
    diff_block = d[1 : MAX_ORDER + 1]
    return d.at[1 : MAX_ORDER + 1].set(m.T @ diff_block)


def update_d(d: jax.Array, dd: jax.Array, k: jax.Array) -> jax.Array:
    """Fold an accepted step's correction ``dd = y_new - predictor`` into ``D``.

    ``dd`` is the order-``k+1`` difference at the new point; the telescoping
    identity ``∇ʲy_{n+1} = ∇ʲy_n + ∇ʲ⁺¹y_{n+1}`` then rebuilds every lower
    row, including the state row 0 (QNDF ``update_D!``, with the fold
    carried through row 0 in SciPy's ``D[0]``-holds-the-state convention). Row ``k+2`` receives the new
    highest difference before the fold. ``k`` may be traced.
    """
    dd = jnp.asarray(dd, dtype=d.dtype)
    d = d.at[k + 2].set(dd - d[k + 1])
    d = d.at[k + 1].set(dd)
    for i in range(MAX_ORDER, -1, -1):
        folded = d[i] + d[i + 1]
        d = d.at[i].set(jnp.where(i <= k, folded, d[i]))
    return d


def seed_d(y0: jax.Array, ydot0: jax.Array, h0: jax.Array) -> jax.Array:
    """The first-step table: the state and its scaled derivative.

    ``D[1] = h0 * ydot0`` is the first backward difference of the
    (as-yet-unbuilt) two-point history, exact for the locally-linear model
    the order-1 method starts from; higher rows are zero.
    """
    y0 = jnp.asarray(y0)
    n = y0.shape[0]
    d = jnp.zeros((D_ROWS, n), dtype=y0.dtype)
    d = d.at[0].set(y0)
    return d.at[1].set(jnp.asarray(h0, dtype=y0.dtype) * jnp.asarray(ydot0, dtype=y0.dtype))


__all__ = [
    "MAX_ORDER",
    "D_ROWS",
    "GAMMA",
    "KAPPA",
    "ERROR_CONST",
    "BETA0",
    "difference_rescale_matrix",
    "rescale_d",
    "update_d",
    "seed_d",
]
