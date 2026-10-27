"""Safe-math primitives: double-``where`` guards for sqrt/div/power.

Each primitive satisfies two contracts:

1. VALUE-identical to the unguarded op for every healthy input — same bits
   (``safe_sqrt(s) == sqrt(s)`` exactly for ``s > 0``).
2. Both branch linearizations finite at the singular point (the double-``where``
   pattern) — so reverse mode through checkpointed / IFT adjoints never
   materializes ``0 * inf = nan``.

Every norm in the package (WRMS, error norms, convergence rates) routes through
these; physics kernels use them only at KNOWN singular points — never sprinkled
speculatively. Lives in ``utils/`` — the package floor, imports nothing above it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def safe_sqrt(s: jax.Array) -> jax.Array:
    """``sqrt`` with finite linearization at 0.

    VALUE-identical to ``jnp.sqrt`` for ``s > 0``; returns ``s`` (i.e. 0) at
    ``s == 0`` with a finite cotangent. ``s`` must be ``>= 0`` or
    ``nan`` by construction.

    Args:
        s: Non-negative argument (scalar or array).

    Returns:
        ``sqrt(s)`` where ``s > 0``, ``s`` elsewhere; grad finite at 0.
    """
    s = jnp.asarray(s)
    pos = s > 0.0
    one = jnp.asarray(1.0, dtype=s.dtype)
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, s, one)), s)


def safe_div(num: jax.Array, den: jax.Array, *, tol: float = 0.0) -> jax.Array:
    """``num / den``, 0 where ``|den| <= tol``; masked-branch VJP stays finite.

    VALUE-identical to ``num / den`` where ``|den| > tol``.

    Args:
        num: Numerator (scalar or array).
        den: Denominator (scalar or array).
        tol: Magnitude below which the quotient is masked to 0. Default 0.0.

    Returns:
        ``num / den`` where ``|den| > tol``, 0 elsewhere; grad finite at 0.
    """
    num, den = jnp.asarray(num), jnp.asarray(den)
    ok = jnp.abs(den) > tol
    one = jnp.asarray(1.0, dtype=den.dtype)
    return jnp.where(ok, num / jnp.where(ok, den, one), jnp.zeros_like(num * one))


def safe_power(base: jax.Array, exponent) -> jax.Array:
    """``base ** exponent`` for ``base > 0``; 0 with finite grad at ``base <= 0``.

    VALUE-identical to ``jnp.power`` for ``base > 0``.

    Args:
        base: Positive base (scalar or array).
        exponent: Exponent (scalar or array; not guarded).

    Returns:
        ``base ** exponent`` where ``base > 0``, 0 elsewhere; grad finite at 0.
    """
    base = jnp.asarray(base)
    pos = base > 0.0
    one = jnp.asarray(1.0, dtype=base.dtype)
    return jnp.where(pos, jnp.power(jnp.where(pos, base, one), exponent), jnp.zeros_like(base))
