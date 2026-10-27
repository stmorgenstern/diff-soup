"""distribution.py — population (distribution-space) observables.

Observe a *family* (a population of members — particles, grains, …) only
through a distribution-space summary of one member field.  This is the
inversion pattern for XRD / MPM problems where the goal is to recover
the distribution parameters of a family.

* [PopulationObservable][diff_soup.inference.data.distribution.PopulationObservable] ([ObservationOperator][diff_soup.inference.data.base.ObservationOperator] subclass):
  gathers the family member slab from the state, reduces each member to
  a scalar value (``member_reduce``) or pools a member field across all
  members, then maps the value sample to an observation vector through a
  *kernel*.
* [SoftHistogram][diff_soup.inference.data.distribution.SoftHistogram] (kernel): a differentiable Gaussian-KDE-at-bin-centers
  density (or soft counts).  XRD peak-shape forward models can be implemented
  as a kernel plug-in using the same protocol.
* [Moments][diff_soup.inference.data.distribution.Moments] (kernel): smooth raw moments.  Cheap, but low moments
  cannot detect bimodality, so ``Moments`` is recommended for diagnostics /
  auxiliary rows rather than as the primary misfit.
* [CDFBins][diff_soup.inference.data.distribution.CDFBins] (kernel, submodule-level): a smooth empirical CDF on a
  fixed grid (Cramér-von-Mises / W₁-flavored rows — better-conditioned tails
  than density bins).

**Family accessor + escape hatch.**  ``PopulationObservable`` uses the
``problem.family_view(name) -> FamilyView`` accessor when available.  When
that accessor is not present, pass ``extract_fn=(u) -> (count, member_state)``
as the fallback gather function.

Kernel contract: ``kernel(values, weights) -> obs`` — ``values`` is the
``(n_values,)`` sample, ``weights`` the ``(n_values,)`` member/quadrature
weights (or ``None`` → uniform).  Kernels inherit the sample's dtype and
contain no hard fp64 casts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diff_soup.inference.data.base import ObservationOperator

# A small additive floor for histogram/CDF normalization (keeps the density finite
# when the soft counts sum to ~0).  Derived from the sample dtype, not a literal cast.
_NORM_FLOOR = 1.0e-30


# ---------------------------------------------------------------------------
# Distribution kernels: value-sample -> observation vector
# ---------------------------------------------------------------------------


class SoftHistogram(eqx.Module):
    """Differentiable Gaussian-KDE-at-bin-centers soft histogram.

    Each member value contributes a Gaussian bump (width ``bandwidth``) summed at
    the bin centers; the result is a density (``normalize=True``, default) or soft
    counts.  Differentiable in every member value.

    Bandwidth is the gradient-smoothness knob: below ~1 bin width the gradients
    localize and may vanish, so a floor of ~1.5·Δbin is recommended.

    Attributes:
        edges: Bin edges, shape ``(n_bins + 1,)``.
        bandwidth: Gaussian kernel bandwidth (should be ≳ 1 bin width).
        normalize: ``True`` (default) returns a density; ``False`` returns soft counts.
    """

    edges: Float[Array, " n_bins_plus_1"]
    bandwidth: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True, default=True)

    def __call__(
        self, values: Float[Array, " n_values"], weights: Optional[Array] = None
    ) -> Float[Array, " n_bins"]:
        """Map a value sample to a soft histogram (density or counts).

        Args:
            values: The ``(n_values,)`` member value sample.
            weights: Optional ``(n_values,)`` member weights (``None`` → uniform).

        Returns:
            The ``(n_bins,)`` density (``normalize``) or soft counts.
        """
        edges = jnp.asarray(self.edges, dtype=values.dtype)
        centers = 0.5 * (edges[:-1] + edges[1:])
        z = (values[:, None] - centers[None, :]) / self.bandwidth
        bumps = jnp.exp(-0.5 * z * z)
        if weights is not None:
            bumps = bumps * jnp.asarray(weights, dtype=values.dtype)[:, None]
        counts = jnp.sum(bumps, axis=0)
        if self.normalize:
            dbin = edges[1] - edges[0]
            total = jnp.sum(counts) * dbin
            return counts / (total + jnp.asarray(_NORM_FLOOR, dtype=values.dtype))
        return counts


class Moments(eqx.Module):
    """Smooth raw moments of the value sample.

    Returns ``E[v^p]`` for each ``p`` in `orders` (weighted mean of powers).
    Smooth and cheap, but low moments cannot detect bimodality — recommended for
    diagnostics and auxiliary rows rather than as the primary misfit.

    Attributes:
        orders: The moment orders to return (default ``(1, 2)``).
    """

    orders: tuple = eqx.field(static=True, default=(1, 2))

    def __call__(
        self, values: Float[Array, " n_values"], weights: Optional[Array] = None
    ) -> Float[Array, " n_orders"]:
        """Map a value sample to its raw moments.

        Args:
            values: The ``(n_values,)`` member value sample.
            weights: Optional ``(n_values,)`` member weights (``None`` → uniform).

        Returns:
            The ``(len(orders),)`` weighted raw moments.
        """
        if weights is None:
            w = jnp.ones_like(values)
        else:
            w = jnp.asarray(weights, dtype=values.dtype)
        wsum = jnp.sum(w) + jnp.asarray(_NORM_FLOOR, dtype=values.dtype)
        return jnp.stack([jnp.sum(w * values ** int(p)) / wsum for p in self.orders])


class CDFBins(eqx.Module):
    """Smooth empirical CDF on a fixed grid (submodule-level).

    Returns the (weighted) fraction of member values below each grid point, smoothed
    by a logistic so it is differentiable — a Cramér-von-Mises / W₁-flavored row set
    with better-conditioned tails than density bins; recommended when histograms are
    sparse.

    Attributes:
        grid: The ``(n_grid,)`` query points (ascending).
        bandwidth: Logistic smoothing width (should be ≳ the grid spacing).
    """

    grid: Float[Array, " n_grid"]
    bandwidth: float = eqx.field(static=True)

    def __call__(
        self, values: Float[Array, " n_values"], weights: Optional[Array] = None
    ) -> Float[Array, " n_grid"]:
        """Map a value sample to a smooth empirical CDF on `grid`.

        Args:
            values: The ``(n_values,)`` member value sample.
            weights: Optional ``(n_values,)`` member weights (``None`` → uniform).

        Returns:
            The ``(n_grid,)`` smooth CDF.
        """
        grid = jnp.asarray(self.grid, dtype=values.dtype)
        # sigmoid((grid - v)/bw) ≈ 1{v <= grid}, differentiable in v.
        z = (grid[None, :] - values[:, None]) / self.bandwidth
        ind = jax.nn.sigmoid(z)
        if weights is None:
            return jnp.mean(ind, axis=0)
        w = jnp.asarray(weights, dtype=values.dtype)
        wsum = jnp.sum(w) + jnp.asarray(_NORM_FLOOR, dtype=values.dtype)
        return jnp.sum(w[:, None] * ind, axis=0) / wsum


# ---------------------------------------------------------------------------
# PopulationObservable: family slab -> value sample -> kernel
# ---------------------------------------------------------------------------


class PopulationObservable(ObservationOperator):
    """Observe a family only through a distribution-space summary.

    ``predict(u, t, theta)`` (``t``/``theta`` unused):

    1. Gather the family member slab — via the ``problem.family_view(name)``
       accessor (when bound through [with_family_view][diff_soup.inference.data.distribution.PopulationObservable.with_family_view]) or the
       `extract_fn` fallback (``(u) -> (count, member_state)``).
    2. Reduce the slab to a value sample — ``vmap(member_reduce)`` over members,
       or pool the `field` column across all members.
    3. Map the value sample to an observation vector through `kernel`.

    Attributes:
        family: The Family IR node name — the accessor key.
        kernel: A distribution kernel ([SoftHistogram][diff_soup.inference.data.distribution.SoftHistogram] / [Moments][diff_soup.inference.data.distribution.Moments] /
            [CDFBins][diff_soup.inference.data.distribution.CDFBins]) mapping ``(values, weights) -> obs``.
        member_reduce: ``(member_state,) -> scalar(s)`` per member; ``None`` pools
            every value (the `field` column).
        field: Which member field column to pool (index) when ``member_reduce`` is
            ``None``; ``None`` pools the entire flattened member state.
        extract_fn: Fallback gather function — a pure
            ``(u) -> (count, member_state)`` callable used when no
            ``family_view`` accessor is available.  Takes precedence when set.
        member_weights: Optional ``(count,)`` volume / quadrature weights (static).
            ``None`` → uniform.
    """

    family: str = eqx.field(static=True)
    kernel: Any  # a kernel eqx.Module (its array leaves, e.g. edges, ride as leaves)
    member_reduce: Optional[Callable] = eqx.field(static=True, default=None)
    field: Optional[Union[int, slice]] = eqx.field(static=True, default=None)
    extract_fn: Optional[Callable] = eqx.field(static=True, default=None)
    member_weights: Optional[Any] = None

    def _slab(self, u: Array) -> tuple[Any, Optional[Array]]:
        """Gather ``(member_state, member_weights)`` from the state ``u``.

        Returns:
            ``(slab, weights)`` — the ``(count, member_state_size)`` slab and optional
            ``(count,)`` weights.

        Raises:
            ValueError: If neither ``extract_fn`` nor a bound ``family_view`` is set.
        """
        if self.extract_fn is not None:
            _count, slab = self.extract_fn(u)
            w = None if self.member_weights is None else jnp.asarray(self.member_weights)
            return slab, w
        raise ValueError(
            "PopulationObservable: no family accessor is available. Pass "
            "extract_fn=(u) -> (count, member_state) to supply a custom gather "
            "function, or call with_family_view() with a lowered problem that "
            "exposes a family_view accessor."
        )

    def _values(self, slab: Any) -> Array:
        """Reduce a member slab to the ``(n_values,)`` value sample."""
        if self.member_reduce is not None:
            return jnp.reshape(jax.vmap(self.member_reduce)(slab), (-1,))
        arr = jnp.asarray(slab)
        if self.field is not None and arr.ndim >= 2:
            arr = arr[:, self.field]
        return jnp.reshape(arr, (-1,))

    def predict(self, nodal_field: Array, t: Any, theta: Any) -> Array:
        """Map a state to the distribution-space observation.

        Args:
            nodal_field: The solver state ``u`` carrying the family slab.
            t: Unused; the family summary does not depend on the frame time.
            theta: Unused; the family summary does not depend on parameters.

        Returns:
            The kernel's observation vector.
        """
        del t, theta
        slab, weights = self._slab(nodal_field)
        values = self._values(slab)
        # Pool-mode weights need to match the pooled value count; the degrade-path
        # member_weights are per-member, so only forward them when they align.
        if (
            weights is not None
            and self.member_reduce is not None
            and weights.shape[0] == values.shape[0]
        ):
            return self.kernel(values, weights)
        return self.kernel(values, None)

    def with_family_view(self, family_view: Any) -> "PopulationObservable":
        """Bind a ``FamilyView`` accessor as the slab gather.

        Returns a copy whose `extract_fn` is the accessor's ``extract`` and
        whose `member_weights` are the accessor's ``member_weights``, so a
        lowered problem that exposes ``family_view(name)`` drives the same
        ``predict`` path as the ``extract_fn`` fallback.

        Args:
            family_view: A ``FamilyView`` (frozen: ``.extract(u)`` returns the
                ``(count, member_state_size)`` slab; also exposes ``.count``,
                ``.field_slice(f)``, and ``.member_weights``).

        Returns:
            A bound [PopulationObservable][diff_soup.inference.data.distribution.PopulationObservable].
        """
        field = self.field
        if isinstance(self.field, str) and hasattr(family_view, "field_slice"):
            field = family_view.field_slice(self.field)
        # ``extract_fn`` / ``field`` / ``family`` are static (treedef) fields, so
        # ``eqx.tree_at`` cannot target them (it traverses leaves only).
        # Reconstruct the module directly with the bound accessor.
        # ``FamilyView.extract(u)`` returns the ``(count, member_state_size)`` slab;
        # adapt here to the ``(count, slab)`` tuple contract.
        count = family_view.count
        return PopulationObservable(
            family=self.family,
            kernel=self.kernel,
            member_reduce=self.member_reduce,
            field=field,
            extract_fn=lambda u, _fv=family_view: (count, _fv.extract(u)),
            member_weights=getattr(family_view, "member_weights", None),
        )
