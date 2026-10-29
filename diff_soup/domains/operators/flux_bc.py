"""General flux boundary conditions (Neumann / Robin).

``FluxBC`` models ``du/dn = flux_fn(u_boundary, t)`` in the most general
form.  The convenience factory function
[RobinBC][diff_soup.domains.operators.flux_bc.RobinBC] returns a configured
[FluxBC][diff_soup.domains.operators.flux_bc.FluxBC] for the Robin case.

``FluxBC.flux_fn`` is a **runtime** (non-static) field so that learnable
callables (e.g. ``eqx.nn.MLP`` modules) expose their array leaves through
``eqx.partition``.  Constant-value BCs are wrapped in ``_ConstFlux(eqx.Module)``,
whose ``value`` is an array leaf.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union, final

import equinox as eqx
import jax
import jax.numpy as jnp

Array = jax.Array


class _ConstFlux(eqx.Module):
    """Zero-leaf wrapper for a constant Neumann flux value.

    Used as the default ``flux_fn`` of
    [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC] so that ``flux_fn``
    stays a runtime (non-static) field.  The ``value`` field is a JAX array leaf
    (traced under AD / vmappable).
    """

    value: jax.Array

    def __call__(self, u, t):  # noqa: ARG002
        """Return the constant flux value (u and t unused)."""
        return self.value


class _ConstRobinFlux(eqx.Module):
    """Two-leaf wrapper for a Robin flux: ``(g - alpha*u) / beta``.

    Replaces bare ``lambda u, t: (_g - _a*u)/_b`` captures in
    [RobinBC][diff_soup.domains.operators.flux_bc.RobinBC].
    """

    alpha: jax.Array
    beta: jax.Array
    g: jax.Array

    def __call__(self, u, t):  # noqa: ARG002
        """Return Robin flux: ``(g - alpha * u) / beta``."""
        return (self.g - self.alpha * u) / self.beta


@final
class FluxBC(eqx.Module):
    r"""General boundary flux condition: ``du/dn = flux_fn(u_boundary, t)``.

    ``@final``: subclassing is forbidden. Use the
    convenience factory function
    [RobinBC][diff_soup.domains.operators.flux_bc.RobinBC] for the Robin case.

    The most general flux-type BC.  Subsumes Neumann (constant flux),
    Robin (linear in ``u``), Newton's cooling, and arbitrary nonlinear
    conditions.

    ``flux_fn`` returns ``du/dn`` (the **derivative**, not the physical
    flux ``D * du/dn``).  The operator multiplies by ``D`` internally,
    so the BC is independent of the diffusivity.

    ``flux_fn`` must be JAX-differentiable: it receives traced scalars
    and must return a traced scalar.  This ensures the sparse Jacobian
    (colored vmap JVP) automatically captures the BC's contribution.

    Boundary identification
    -----------------------
    * **Grid1D**: use ``side=`` (``'low'`` or ``'high'``); ``axis`` defaults to 0.
    * **GridND**: use ``axis=`` + ``side=``.

    The operator bank applies a ``FluxBC`` on structured grids only. A flux
    through a tagged mesh boundary is a boundary integral that the problem
    assembles from ``spec.bc.flux(var, tag=...)``; passing a ``FluxBC`` to
    ``ops.div_flux`` / ``ops.laplacian`` on a mesh raises a
    ``ConfigurationError`` that says so.

    Attributes:
        tag: Boundary tag (int or name). The operator bank refuses a
            ``FluxBC`` on a mesh; see above.
        axis: Spatial axis (0, 1, 2) for structured grids.
        side: ``'low'`` or ``'high'`` for structured grids.
        flux_fn: Callable ``(u_boundary, t) -> scalar`` returning ``du/dn``.
            Must be JAX-traceable.  ``u_boundary`` is the solution value
            at the boundary node/cell (a traced JAX scalar).

    Example:
        Newton's cooling law ``du/dn = h_conv * (T_inf - u)``::

            bc = FluxBC(side='high', flux_fn=lambda u, t: h_conv * (T_inf - u))
    """

    tag: Optional[Union[int, str]] = eqx.field(static=True, default=None)
    axis: Optional[int] = eqx.field(static=True, default=None)
    side: Optional[str] = eqx.field(static=True, default=None)
    # A runtime (non-static) field, so learnable eqx.Module flux callables
    # expose their array leaves through eqx.partition.  Constant-value BCs are
    # wrapped by _ConstFlux, whose value is an array leaf.
    flux_fn: Callable = eqx.field(default_factory=lambda: _ConstFlux(value=jnp.asarray(0.0)))


def _validate_side(cls_name: str, side: Optional[str]) -> None:
    """Reject a structured-grid ``side`` that is not ``'low'`` / ``'high'``."""
    if side is not None and side not in ("low", "high"):
        raise ValueError(f"{cls_name}: side must be 'low' or 'high', got {side!r}")


def RobinBC(
    alpha: float,
    beta: float,
    g: float,
    *,
    tag: Optional[Union[int, str]] = None,
    axis: Optional[int] = None,
    side: Optional[str] = None,
) -> FluxBC:
    r"""Robin flux BC: ``alpha*u + beta*du/dn = g``.

    Rearranges to ``du/dn = (g - alpha*u) / beta``.

    Args:
        alpha: Coefficient of ``u``.
        beta: Coefficient of ``du/dn``.  Must be nonzero.
        g: RHS constant.
        tag: Boundary tag forwarded to
        [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC].
        axis: Boundary axis forwarded to
        [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC].
        side: Boundary side forwarded to
        [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC].

    Returns:
        A [FluxBC][diff_soup.domains.operators.flux_bc.FluxBC] instance
        with Robin flux_fn.
    """
    _a, _b, _g = float(alpha), float(beta), float(g)
    if _b == 0.0:
        raise ValueError(f"RobinBC: beta must be nonzero (du/dn coefficient), got beta={beta}")
    _validate_side("RobinBC", side)
    # _ConstRobinFlux carries alpha/beta/g as traced leaves.
    return FluxBC(
        tag=tag,
        axis=axis,
        side=side,
        flux_fn=_ConstRobinFlux(
            alpha=jnp.asarray(_a),
            beta=jnp.asarray(_b),
            g=jnp.asarray(_g),
        ),
    )


class _NormalDerivativeBC(eqx.Module):
    """A prescribed outward normal derivative on one grid face, already evaluated.

    Built for each residual evaluation from ``spec.bc.neumann`` and merged into
    the ``bcs`` of ``laplacian`` / ``div_flux`` applied to the conditioned field.
    The operator imposes it through a ghost value, so the face flux is the
    operator's own diffusivity times ``value``.

    Attributes:
        axis: Grid axis normal to the face.
        side: ``"low"`` or ``"high"``.
        value: The outward normal derivative, a scalar or one value per face cell
            (the grid shape with ``axis`` removed).
        mask: ``1`` at the face cells the condition holds on and ``0`` elsewhere,
            or ``None`` for the whole face.
    """

    axis: int = eqx.field(static=True)
    side: str = eqx.field(static=True)
    value: jax.Array
    mask: Optional[jax.Array] = None


class _MeshNormalDerivative(eqx.Module):
    """A prescribed outward normal derivative on tagged mesh facets, already evaluated.

    The mesh counterpart of [_NormalDerivativeBC][diff_soup.domains.operators.flux_bc._NormalDerivativeBC].
    ``laplacian`` / ``div_flux`` applied to the conditioned field add the natural
    boundary term ``int D value N_i dS`` with their own diffusivity ``D``.

    Attributes:
        facets: Indices of the selected rows of ``mesh.boundary_facets()``.
        value: The outward normal derivative, a scalar or one value per selected facet.
    """

    facets: tuple = eqx.field(static=True)
    value: jax.Array
