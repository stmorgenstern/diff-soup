"""Backward-differentiation coefficient tables for the NDF and BDF families.

The two families share the gamma (harmonic partial sum) and denominator tables
and differ only by the kappa vector: NDF uses the Klopfenstein-Shampine kappas
(the family used by variable-order NDF/QNDF integrators), BDF sets kappa to zero. Both store the reciprocal
``inv_ga = 1/(gamma*(1-kappa))``; the iteration pencil is ``M - hinvGak*J`` with
``hinvGak = h*inv_ga`` and the leading coefficient is ``cj = 1/hinvGak``.
"""

import jax.numpy as jnp

GAMMA = (1.0, 3.0 / 2.0, 11.0 / 6.0, 25.0 / 12.0, 137.0 / 60.0)
KAPPA_NDF = (-37.0 / 200.0, -1.0 / 9.0, -0.0823, -0.0415, 0.0)
KAPPA_BDF = (0.0, 0.0, 0.0, 0.0, 0.0)
DEN = (2.0, 3.0, 4.0, 5.0, 6.0)

_KAPPA = {"ndf": KAPPA_NDF, "bdf": KAPPA_BDF}


def coeffs(family, dtype):
    """Return ``(gamma, inv_ga, erconst)`` for ``family`` in ``{"ndf","bdf"}``.

    Parameters
    ----------
    family:
        ``"ndf"`` for the Klopfenstein-Shampine NDF family or ``"bdf"`` for
        the standard BDF family.
    dtype:
        JAX floating-point dtype for the returned arrays.

    Returns:
    -------
    gamma:
        Harmonic partial sum coefficients, shape ``(5,)``.
    inv_ga:
        Reciprocal leading coefficients ``1/(gamma*(1-kappa))``, shape ``(5,)``.
    erconst:
        Local-error constants ``kappa*gamma + 1/den``, shape ``(5,)``.
    """
    if family not in _KAPPA:
        raise ValueError(f"coefficient family must be 'ndf' or 'bdf', got {family!r}")
    gamma = jnp.asarray(GAMMA, dtype=dtype)
    kappa = jnp.asarray(_KAPPA[family], dtype=dtype)
    den = jnp.asarray(DEN, dtype=dtype)
    inv_ga = 1.0 / (gamma * (1.0 - kappa))
    erconst = kappa * gamma + 1.0 / den
    return gamma, inv_ga, erconst


def cj_from_hinvgak(hinvGak):
    """Materialize the leading coefficient ``cj = 1/hinvGak``.

    The iteration matrix pencil is ``M - hinvGak*J`` where ``hinvGak = h*inv_ga``.
    The leading coefficient ``cj`` equals ``gamma*(1-kappa)/h`` and is always
    stored as the reciprocal of ``hinvGak``, never computed by inverting that
    expression directly.
    """
    return 1.0 / hinvGak
