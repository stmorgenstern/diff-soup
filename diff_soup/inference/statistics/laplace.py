"""laplace.py — rank-safe Gaussian (Laplace) posterior around a fitted mode.

A Laplace approximation summarizes a posterior as a Gaussian centered at the
mode with precision equal to a curvature matrix: the data curvature (expected
Fisher information, observed negative-log-likelihood Hessian, or Gauss-Newton
approximation — the ``curvature_kind`` label keeps them distinct) plus an
optional Gaussian prior precision. [laplace_posterior][diff_soup.inference.statistics.laplace.laplace_posterior] builds the
approximation rank-safely:

* an SPD total precision factorizes (Cholesky) into a valid
  [LaplacePosterior][diff_soup.inference.statistics.laplace.LaplacePosterior];
* singular data curvature closed by a proper prior is still a valid
  posterior, and [identifiability][diff_soup.inference.identifiability] keeps
  reporting which directions the *data* left undetermined;
* a singular total precision never becomes invented finite variances — the
  caller receives an
  [ImproperPosteriorError][diff_soup.inference.statistics.report.ImproperPosteriorError] carrying
  the [IdentifiedSubspace][diff_soup.inference.statistics.report.IdentifiedSubspace], or the
  subspace itself (``on_singular="raise"`` or ``"subspace"``).

Singularity is detected by an eigenvalue threshold on the total precision
before any factorization is attempted, so failure is a deliberate statistical
statement rather than a numerical accident. Every posterior query
([LaplacePosterior.covariance_action][diff_soup.inference.statistics.laplace.LaplacePosterior.covariance_action],
[LaplacePosterior.marginal_sd][diff_soup.inference.statistics.laplace.LaplacePosterior.marginal_sd], [LaplacePosterior.sample][diff_soup.inference.statistics.laplace.LaplacePosterior.sample],
[LaplacePosterior.log_det_precision][diff_soup.inference.statistics.laplace.LaplacePosterior.log_det_precision]) goes through triangular solves
against the stored Cholesky factor — no matrix inverse is ever formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.statistics.estimation import EstimationProblem
from diff_soup.inference.statistics.information import (
    _coordinate_names,
    _information_terms,
    _observed_curvature,
    _point,
    _recorded_data_jacobian,
    _window_split,
    assemble_information,
)
from diff_soup.inference.statistics.measurement import require_likelihoods
from diff_soup.inference.statistics.parameters import COORDINATES
from diff_soup.inference.statistics.report import IdentifiedSubspace, ImproperPosteriorError

__all__ = [
    "LaplacePosterior",
    "laplace",
    "laplace_posterior",
]

CURVATURE_KINDS = ("expected", "observed", "ggn")

# Relative eigenvalue threshold on the total precision: directions at or below
# null_tol * lambda_max carry no usable curvature and make the posterior improper.
_NULL_TOL = 1e-10

# Relative tolerance for rejecting a materially asymmetric curvature input.
_SYMMETRY_RTOL = 1e-8

# Stationarity tolerance on the Newton decrement sqrt(g^T H^-1 g) at a fit. The log joint
# at the exact mode is about half its square above the fit's value, 0.005 at this bound,
# which bounds the error the fit's point adds to a Laplace log evidence.
_STATIONARITY_TOL = 0.1


@dataclass(frozen=True)
class Marginal:
    """Mean, standard deviation, and Gaussian interval for a fitted handle."""

    mean: Any
    sd: Any

    def interval(self, level=0.95):
        """Return the central pointwise interval at the requested level."""
        if not 0 < level < 1:
            raise ValueError("interval level must lie between zero and one.")
        z = NormalDist().inv_cdf((1 + level) / 2)
        return self.mean - z * self.sd, self.mean + z * self.sd


class _CovarianceLookup:
    def __init__(self, posterior):
        self.posterior = posterior

    def __getitem__(self, handles):
        left, right = (self.posterior.parameter_schema.entry(handle) for handle in handles)
        covariance = self.posterior.covariance_dense()
        entries = {}
        for a in left.occurrences:
            for b in right.occurrences:
                block = covariance[jnp.ix_(jnp.asarray(a.columns), jnp.asarray(b.columns))]
                entries[a.owner, b.owner] = block.reshape(()) if block.size == 1 else block
        return entries[None, None] if tuple(entries) == ((None, None),) else entries


class LaplacePosterior(eqx.Module):
    """A proper Gaussian (Laplace) posterior with its curvature provenance.

    Built by [laplace_posterior][diff_soup.inference.statistics.laplace.laplace_posterior], which guarantees the stored precision
    is symmetric positive definite. All covariance queries are solves against
    the Cholesky factor; no inverse is ever formed.

    Attributes:
        mode: The MAP/mode the Gaussian is centered at, shape ``(n,)``,
            expressed in ``coordinate``.
        coordinate: The coordinate the density lives in — ``"x"`` (physical)
            or ``"z"`` (inference).
        curvature_kind: Which curvature the data term is — ``"expected"``
            (Fisher information), ``"observed"`` (NLL Hessian), or ``"ggn"``
            (Gauss-Newton). A label, never silently interchanged.
        precision_factor: Lower Cholesky factor ``L`` of the total posterior
            precision ``P = L Lᵀ`` (data curvature plus prior precision).
        prior_spectrum: Eigenvalues of the prior precision, descending, or
            ``None`` when no prior was used.
        names: Optional per-parameter labels, or ``()``.
        prior_precision: The prior contribution used in the stored total
            curvature, or ``None``. Retained so evidence calculations can
            verify the supplied normalized prior instead of trusting a label.
        log_likelihood_at_mode: Normalized log likelihood at ``mode`` when it
            was explicitly supplied, otherwise ``None``.
        likelihood_normalized: Whether ``log_likelihood_at_mode`` is known to
            include every density normalizer, or ``None`` when unknown.
        curvature_regularization: Numerical ridge added to the reported
            curvature. ``None`` means its status is unknown; a clean,
            unregularized construction records ``0.0`` explicitly.
        mode_gradient_norm: Size of the log-joint gradient ``g`` at ``mode``
            when checked, which `laplace` records as the Newton decrement
            ``sqrt(g^T P^-1 g)``, otherwise ``None``.
        stationarity_tolerance: Tolerance used for the stationarity check.
        provenance: Caller-supplied experiment, channel, fit, and geometry
            metadata.
        warnings: Geometry warnings accumulated while constructing the mode.
    """

    mode: Array
    coordinate: str = eqx.field(static=True)
    curvature_kind: str = eqx.field(static=True)
    precision_factor: Array
    prior_spectrum: Optional[Array]
    names: tuple = eqx.field(static=True, default=())
    prior_precision: Optional[Array] = None
    log_likelihood_at_mode: Optional[Array] = None
    likelihood_normalized: Optional[bool] = eqx.field(static=True, default=None)
    curvature_regularization: Optional[float] = eqx.field(static=True, default=None)
    mode_gradient_norm: Optional[float] = eqx.field(static=True, default=None)
    stationarity_tolerance: Optional[float] = eqx.field(static=True, default=None)
    provenance: Any = None
    warnings: tuple = eqx.field(static=True, default=())

    parameter_schema: Any = None
    _data_curvature: Optional[Array] = None
    _evidence_prior: Any = None

    def __getitem__(self, handle):
        """Return the marginal for a shared handle, or marginals keyed by owner."""
        if self.parameter_schema is None:
            raise TypeError("handle lookup requires a posterior from laplace(problem, fit).")
        entry = self.parameter_schema.entry(handle)
        sd = self.marginal_sd()
        values = {}
        for occurrence in entry.occurrences:
            columns = jnp.asarray(occurrence.columns)
            mean = self.mode[columns]
            spread = sd[columns]
            if entry.size == 1:
                mean, spread = mean.reshape(()), spread.reshape(())
            values[occurrence.owner] = Marginal(mean, spread)
        return values[None] if tuple(values) == (None,) else values

    @property
    def cov(self):
        """Covariance entries indexed as ``post.cov[left_handle, right_handle]``."""
        return _CovarianceLookup(self)

    @property
    def dim(self) -> int:
        """Number of parameters the posterior covers."""
        return int(self.mode.shape[0])

    def covariance_action(self, v: Any) -> Array:
        """Return ``P⁻¹ v`` via two triangular solves (no formed inverse).

        Args:
            v: Vector of shape ``(n,)`` or stack of column vectors ``(n, k)``.

        Returns:
            The posterior covariance applied to ``v``, same shape as ``v``.
        """
        v = jnp.asarray(v)
        w = solve_triangular(self.precision_factor, v, lower=True)
        return solve_triangular(self.precision_factor.T, w, lower=False)

    def covariance_dense(self) -> Array:
        """Return the explicit ``(n, n)`` posterior covariance.

        A small-problem convenience: the full matrix is assembled from
        triangular solves against the identity (still no ``inv()``), so it
        costs ``n`` solves and ``n²`` storage. For large ``n`` prefer
        [LaplacePosterior.covariance_action]
        [diff_soup.inference.statistics.laplace.LaplacePosterior.covariance_action]
        and [LaplacePosterior.marginal_sd]
        [diff_soup.inference.statistics.laplace.LaplacePosterior.marginal_sd].
        """
        cov = self.covariance_action(jnp.eye(self.dim, dtype=self.mode.dtype))
        return 0.5 * (cov + cov.T)

    def marginal_sd(self, names: Any = None) -> Array:
        """Return posterior marginal standard deviations.

        The covariance diagonal is read off the inverse Cholesky factor
        (``diag(P⁻¹)ᵢ = ‖L⁻¹ eᵢ‖²``), never from a formed inverse.

        Args:
            names: ``None`` for all parameters in order; otherwise a single
                label or index, or an iterable of labels/indices, selecting
                marginals. String labels require the posterior to carry
                ``names``.

        Returns:
            Standard deviations, shape ``(n,)`` or ``(len(names),)``.

        Raises:
            ValueError: If a string label is used on a posterior built
                without ``names``, or a label is unknown.
        """
        linv = solve_triangular(
            self.precision_factor, jnp.eye(self.dim, dtype=self.mode.dtype), lower=True
        )
        sd = jnp.sqrt(jnp.sum(linv**2, axis=0))
        if names is None:
            return sd
        if isinstance(names, (str, int, np.integer)):
            names = (names,)
        indices = []
        for label in names:
            if isinstance(label, str):
                if not self.names:
                    raise ValueError(
                        "marginal_sd: selection by name needs per-parameter labels; "
                        "build the posterior with names=(...) or select by index."
                    )
                if label not in self.names:
                    raise ValueError(
                        f"marginal_sd: unknown parameter name {label!r}; "
                        f"known names are {self.names}."
                    )
                indices.append(self.names.index(label))
            else:
                indices.append(int(label))
        return sd[jnp.asarray(indices)]

    def sample(self, key: Array, n: int) -> Array:
        """Draw ``n`` posterior samples, shape ``(n, dim)``.

        Uses ``mode + L⁻ᵀ ε`` with ``ε ~ N(0, I)``: since the covariance is
        ``P⁻¹ = L⁻ᵀ L⁻¹``, the solve colors the noise exactly.

        Args:
            key: A ``jax.random`` PRNG key.
            n: Number of draws.

        Returns:
            Samples of shape ``(n, dim)``.
        """
        eps = jax.random.normal(key, (int(n), self.dim), dtype=self.mode.dtype)
        colored = solve_triangular(self.precision_factor.T, eps.T, lower=False)
        return self.mode[None, :] + colored.T

    def log_det_precision(self) -> Array:
        """Return ``log det P`` from the Cholesky diagonal (exact, stable)."""
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.precision_factor)))

    @property
    def log_evidence(self) -> float:
        """The Laplace approximation to the log marginal likelihood, as `ds.inference.evidence` returns it."""
        from diff_soup.inference.statistics.evidence import laplace_evidence

        prior = self._evidence_prior
        if isinstance(prior, str):
            raise ValueError(prior)
        if prior is None:
            raise ValueError(
                "log_evidence needs a posterior from ds.inference.laplace(problem, fit); for "
                "a posterior built by hand use laplace_evidence(posterior, prior)."
            )
        return laplace_evidence(self, prior).log_evidence


# What satisfies the finiteness check, per source of curvature.
_OBSERVED_REMEDY = (
    'Pass curvature="expected" to use the Gauss-Newton curvature from the whitened '
    "sensitivity, which does not differentiate through the solve twice, or rescale or "
    "re-fit those parameters so their second derivatives evaluate finitely."
)
_SENSITIVITY_REMEDY = (
    "The forward sensitivity did not evaluate finitely there: check the forward solve "
    "at those parameters (a failed segment or an overflowing scale) or rescale them."
)
_PRIOR_REMEDY = "The prior declared on those parameters must have a finite precision at the mode."


def _require_finite(
    matrix: np.ndarray,
    *,
    what: str,
    remedy: str,
    names: tuple = (),
    square: bool = True,
) -> None:
    """Reject a curvature or Jacobian holding non-finite entries, naming the coordinates.

    NaN entries pass a symmetry comparison silently and reach
    ``numpy.linalg`` only as a bare ``LinAlgError``, so the offending
    parameter coordinates are reported here instead.

    Args:
        matrix: The array to check; its columns index parameter coordinates.
        what: How to name the array to the user, e.g. ``"observed curvature"``.
        remedy: The sentence stating what would satisfy the check.
        names: Optional per-coordinate labels, length equal to the number of
            columns; coordinates are reported by index when this is empty.
        square: Whether the rows index parameter coordinates too — true of a
            curvature matrix, false of a whitened Jacobian.
    """
    bad = ~np.isfinite(matrix)
    if not bool(bad.any()):
        return
    touched = bad.any(axis=0)
    if square:
        touched = touched | bad.any(axis=1)
    indices = [int(index) for index in np.flatnonzero(touched)]
    if names:
        where = ", ".join(str(names[index]) for index in indices)
    else:
        where = ("index " if len(indices) == 1 else "indices ") + ", ".join(
            str(index) for index in indices
        )
    count = int(bad.sum())
    raise ValueError(
        f"laplace: the {what} holds {count} non-finite "
        f"{'entry' if count == 1 else 'entries'}, in the "
        f"{'rows or columns' if square else 'columns'} of {where}. {remedy}"
    )


def _as_symmetric(
    matrix: Any,
    n: int,
    *,
    what: str,
    label: str,
    remedy: str,
    names: tuple = (),
    dtype: Any = float,
) -> np.ndarray:
    """Validate an ``(n, n)`` symmetric matrix and return its symmetrized copy.

    ``what`` names the argument in shape and symmetry errors; ``label`` and
    ``remedy`` describe the matrix and its fix when it is not finite.
    """
    a = np.asarray(matrix, dtype=dtype)
    if a.ndim != 2 or a.shape != (n, n):
        raise ValueError(
            f"laplace_posterior: {what} must be a square ({n}, {n}) matrix "
            f"matching the mode; got shape {a.shape}."
        )
    _require_finite(a, what=label, remedy=remedy, names=names)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    if float(np.max(np.abs(a - a.T))) > _SYMMETRY_RTOL * max(scale, 1.0):
        raise ValueError(
            f"laplace_posterior: {what} must be symmetric; the largest "
            f"asymmetry exceeds the relative tolerance {_SYMMETRY_RTOL:.0e}. "
            "Pass the symmetric curvature (or its whitened Jacobian)."
        )
    return 0.5 * (a + a.T)


def _descending_eigh(matrix: np.ndarray) -> tuple:
    """Return (eigenvalues, eigenvector columns) of a symmetric matrix, descending."""
    w, v = np.linalg.eigh(matrix)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def laplace_posterior(
    mode: Any,
    *,
    data_curvature: Any = None,
    whitened_jacobian: Any = None,
    prior_precision: Any = None,
    coordinate: str = "x",
    curvature_kind: str,
    names: tuple = (),
    on_singular: str = "raise",
    log_likelihood_at_mode: Any = None,
    likelihood_normalized: Optional[bool] = None,
    curvature_regularization: Optional[float] = None,
    mode_gradient_norm: Optional[float] = None,
    stationarity_tolerance: Optional[float] = None,
    provenance: Any = None,
    warnings: tuple = (),
) -> Union[LaplacePosterior, IdentifiedSubspace]:
    """Build a rank-safe Gaussian (Laplace) posterior at a fitted mode.

    The total posterior precision is the data curvature plus the prior
    precision (when given). Its spectrum is checked before factorization:
    directions with relative curvature at or below ``1e-10 · λ_max`` make the
    posterior improper, and no finite variances are invented for them.

    Args:
        mode: The MAP/mode, shape ``(n,)``, in ``coordinate``.
        data_curvature: Symmetric ``(n, n)`` data curvature matrix. Exactly
            one of ``data_curvature`` and ``whitened_jacobian`` must be given.
        whitened_jacobian: The ``1/σ``-weighted sensitivity matrix, shape
            ``(n_rows, n)``; the data curvature is its Gram matrix ``Jᵀ J``,
            with the spectrum taken from the SVD of ``J`` (more accurate than
            eigendecomposing the formed Gram matrix).
        prior_precision: Optional symmetric ``(n, n)`` Gaussian prior
            precision (e.g. ``GaussianPrior(...).precision()``), added to the
            data curvature.
        coordinate: The coordinate the mode and curvature live in — ``"x"``
            (physical, default) or ``"z"`` (inference).
        curvature_kind: What the data curvature is — ``"expected"`` (Fisher
            information), ``"observed"`` (NLL Hessian), or ``"ggn"``
            (Gauss-Newton). Stored verbatim so downstream consumers know
            which approximation they are looking at.
        names: Optional per-parameter labels, length ``n``.
        on_singular: What to do when the total precision is singular (or
            indefinite): ``"raise"`` raises
            [ImproperPosteriorError]
            [diff_soup.inference.statistics.report.ImproperPosteriorError]
            carrying the identified subspace; ``"subspace"`` returns the
            [IdentifiedSubspace][diff_soup.inference.statistics.report.IdentifiedSubspace]
            instead of a posterior.
        log_likelihood_at_mode: Optional normalized log likelihood evaluated
            at ``mode``. Evidence calculations require this value.
        likelihood_normalized: Explicit declaration that the stored log
            likelihood includes its density normalization. ``None`` records
            that this was not checked.
        curvature_regularization: Numerical ridge added to the total
            curvature, with ``0.0`` declaring an unregularized factor and
            ``None`` recording unknown status.
        mode_gradient_norm: Optional norm of the log-joint gradient at the
            reported mode.
        stationarity_tolerance: Tolerance used to judge
            ``mode_gradient_norm``.
        provenance: Experiment, channel, likelihood, and fit metadata retained
            for downstream reports.
        warnings: Geometry warnings retained with the posterior.

    Returns:
        A [LaplacePosterior][diff_soup.inference.statistics.laplace.LaplacePosterior], or an
        [IdentifiedSubspace][diff_soup.inference.statistics.report.IdentifiedSubspace] when
        the total precision is singular and ``on_singular="subspace"``.

    Raises:
        ValueError: If the inputs are malformed (neither or both curvature
            sources, wrong shapes, non-finite entries — reported against the
            parameter coordinates that carry them — material asymmetry,
            unknown coordinate / curvature kind / singular policy, wrong
            ``names`` length).
        ImproperPosteriorError: If the total precision leaves parameter
            directions undetermined and ``on_singular="raise"``; the error's
            ``subspace`` attribute describes what the data and prior do
            determine.
    """
    if coordinate not in COORDINATES:
        raise ValueError(
            f"laplace_posterior: coordinate must be one of {COORDINATES} — 'x' for "
            f"the physical parameters, 'z' for the inference coordinate; got "
            f"{coordinate!r}."
        )
    if curvature_kind not in CURVATURE_KINDS:
        raise ValueError(
            f"laplace_posterior: curvature_kind must be one of {CURVATURE_KINDS}; "
            f"got {curvature_kind!r}."
        )
    if on_singular not in ("raise", "subspace"):
        raise ValueError(
            "laplace_posterior: on_singular must be 'raise' (ImproperPosteriorError) "
            f"or 'subspace' (return the IdentifiedSubspace); got {on_singular!r}."
        )
    if (data_curvature is None) == (whitened_jacobian is None):
        raise ValueError(
            "laplace_posterior: pass exactly one of data_curvature and whitened_jacobian."
        )

    # The posterior is computed at the mode's precision: a single-precision
    # fit gives a single-precision posterior.
    mode_arr = np.asarray(mode)
    dtype = mode_arr.dtype if np.issubdtype(mode_arr.dtype, np.floating) else np.dtype(float)
    mode_arr = mode_arr.astype(dtype)
    if mode_arr.ndim != 1:
        raise ValueError(
            f"laplace_posterior: mode must be a 1-D parameter vector; got shape {mode_arr.shape}."
        )
    n = mode_arr.shape[0]

    names = tuple(names)
    if names and len(names) != n:
        raise ValueError(f"laplace_posterior: {len(names)} names for {n} parameters.")

    if whitened_jacobian is not None:
        j = np.asarray(whitened_jacobian, dtype=dtype)
        if j.ndim != 2 or j.shape[1] != n:
            raise ValueError(
                f"laplace_posterior: whitened_jacobian must have shape (n_rows, {n}) "
                f"matching the mode; got {j.shape}."
            )
        _require_finite(
            j, what="whitened Jacobian", remedy=_SENSITIVITY_REMEDY, names=names, square=False
        )
        data = j.T @ j
    else:
        data = _as_symmetric(
            data_curvature,
            n,
            what="data_curvature",
            label=f"{curvature_kind} curvature",
            remedy=_OBSERVED_REMEDY if curvature_kind == "observed" else _SENSITIVITY_REMEDY,
            names=names,
            dtype=dtype,
        )

    prior_spectrum = None
    prior = None
    total = data
    if prior_precision is not None:
        prior = _as_symmetric(
            prior_precision,
            n,
            what="prior_precision",
            label="prior precision",
            remedy=_PRIOR_REMEDY,
            names=names,
            dtype=dtype,
        )
        prior_spectrum = _descending_eigh(prior)[0]
        total = data + prior

    total = 0.5 * (total + total.T)
    w_tot, v_tot = _descending_eigh(total)
    lam_max = float(w_tot[0]) if n else 0.0
    identified = w_tot > _NULL_TOL * max(lam_max, 0.0)
    rank = int(np.sum(identified))
    if rank < n:
        subspace = IdentifiedSubspace(
            directions=v_tot[:, :rank],
            variances=1.0 / w_tot[:rank] if rank else np.zeros(0),
            null_directions=v_tot[:, rank:],
            names=names,
        )
        if on_singular == "subspace":
            return subspace
        prior_clause = (
            "the data curvature plus the prior precision"
            if prior_precision is not None
            else "the data curvature alone (no prior was given)"
        )
        raise ImproperPosteriorError(
            f"laplace_posterior: the total posterior precision — {prior_clause} — "
            f"leaves {n - rank} of {n} parameter directions undetermined "
            f"(relative curvature threshold {_NULL_TOL:.0e}), so no proper "
            "Gaussian posterior exists. Close the undetermined directions with a "
            "proper prior, or pin or re-parameterize them; the determined "
            "directions and their variances are attached as the error's "
            "subspace attribute.",
            subspace,
        )

    factor = np.linalg.cholesky(total)
    likelihood_normalized = None if likelihood_normalized is None else bool(likelihood_normalized)
    curvature_regularization = (
        None if curvature_regularization is None else float(curvature_regularization)
    )
    mode_gradient_norm = None if mode_gradient_norm is None else float(mode_gradient_norm)
    stationarity_tolerance = (
        None if stationarity_tolerance is None else float(stationarity_tolerance)
    )
    return LaplacePosterior(
        mode=jnp.asarray(mode_arr),
        coordinate=coordinate,
        curvature_kind=curvature_kind,
        precision_factor=jnp.asarray(factor),
        prior_spectrum=None if prior_spectrum is None else jnp.asarray(prior_spectrum),
        _data_curvature=jnp.asarray(data),
        names=names,
        prior_precision=None if prior is None else jnp.asarray(prior),
        log_likelihood_at_mode=(
            None if log_likelihood_at_mode is None else jnp.asarray(log_likelihood_at_mode)
        ),
        likelihood_normalized=likelihood_normalized,
        curvature_regularization=curvature_regularization,
        mode_gradient_norm=mode_gradient_norm,
        stationarity_tolerance=stationarity_tolerance,
        provenance=provenance,
        warnings=tuple(warnings),
    )


class _DeclaredPrior:
    """The joint prior a MAP problem declares, as the normalized density `laplace_evidence` takes."""

    coordinate = "x"
    proper = True
    normalized = True

    def __init__(self, problem: EstimationProblem):
        self._problem = problem
        self.size = int(problem.n_params)

    def log_prob(self, value: Any) -> Array:
        """The normalized log density of the declared priors at ``value``."""
        return self._problem.log_prior(value)


def _evidence_refusal(problem: EstimationProblem) -> str | None:
    """Why ``problem`` has no Laplace evidence, or ``None`` when it has one."""
    from diff_soup.inference.statistics.estimation import MAP

    split = _window_split(problem)
    if split is not None:
        return (
            "evidence needs a single-shooting fit, because the window starts of a "
            "multiple-shooting fit carry no normalized prior; refit with single shooting "
            f"from fit.params[:{split[0].size}]."
        )
    if not isinstance(problem.objective, MAP):
        return (
            "evidence needs a normalized prior: build the problem with objective=MAP() and "
            "a prior on every fitted handle."
        )
    return None


def _negative_log_joint_gradient(problem: EstimationProblem, point: Any, terms: tuple) -> Any:
    """The gradient of the negative log likelihood plus the negative log prior at ``point``.

    ``terms`` is the whitened data ``(residual, jacobian)`` at ``point`` when both are
    known, so the data gradient is ``J^T r`` and no solve runs; otherwise the
    problem's own objective gradient is evaluated once.
    """
    residual, jacobian = terms
    if residual is not None and jacobian is not None:
        data = jnp.asarray(jacobian).T @ jnp.asarray(residual)
        prior = jax.grad(lambda x: -problem.log_prior(x))(jnp.asarray(point))
        return np.asarray(data + prior, dtype=float)
    # The MAP objective's prior rows differ from the negative log prior by a constant.
    ad = "forward" if "forward" in problem.supported_compositions else "reverse"
    return np.asarray(problem.loss_and_grad_fn(ad)(jnp.asarray(point))[1], dtype=float)


def _laplace_facade(
    problem: EstimationProblem,
    fit: OptResult | None = None,
    *,
    at: Any = None,
    curvature: str = "observed",
    on_singular: str = "raise",
) -> LaplacePosterior | IdentifiedSubspace:
    """Laplace posterior of an ``InferenceProblem`` (the estimation-problem route of ``laplace``).

    Supply either ``fit`` or ``at``; a fit must carry the same
    parameter schema as ``problem``. A fit of the same study that recorded the
    data Jacobian at its parameters supplies that Jacobian and its log
    likelihood, so the expected curvature runs no forward solve. A
    decision-variable multishoot fit has its window starts integrated out.
    On a MAP problem the posterior records the Newton decrement of the fit's
    point and the declared priors, which is what its log evidence needs.
    """
    if (fit is None) == (at is None):
        raise ValueError("laplace needs exactly one of fit or at.")
    if curvature not in CURVATURE_KINDS:
        raise ValueError(f"laplace curvature must be one of {CURVATURE_KINDS}.")
    source = fit if fit is not None else at
    point = _point(problem, source)
    split = _window_split(problem)
    mode = point if split is None else point[jnp.asarray(split[0])]
    # The residual Jacobian settles both the rank question below and the
    # expected/Gauss-Newton curvature, so the observed Hessian — much the more
    # expensive of the two — is assembled only where it is actually used.
    info, residual = _information_terms(
        problem,
        at=source if isinstance(source, OptResult) else point,
        kind="expected" if curvature == "observed" else curvature,
    )
    recorded_fit = (
        isinstance(source, OptResult)
        and _recorded_data_jacobian(problem, source, point) is not None
    )
    log_likelihood = (
        source.log_likelihood
        if recorded_fit and source.log_likelihood is not None
        else problem.log_likelihood(point)
    )
    provenance = dict(info.provenance)
    provenance["curvature_kind"] = curvature
    kwargs = {
        "prior_precision": info.prior_precision,
        "coordinate": "x",
        "curvature_kind": curvature,
        "names": _coordinate_names(problem),
        "on_singular": on_singular,
        "log_likelihood_at_mode": log_likelihood,
        # A multishoot objective also carries the continuity rows, which no
        # measurement normalizes.
        "likelihood_normalized": split is None,
        "curvature_regularization": 0.0,
        "provenance": provenance,
    }
    # A lowered study gives J^T J summed over its lanes in place of the residual Jacobian.
    if info.whitened_jacobian is None:
        expected = {"data_curvature": np.asarray(info.expected, dtype=float)}
    else:
        expected = {"whitened_jacobian": info.whitened_jacobian}
    if info.prior_precision is None:
        if info.whitened_jacobian is None:
            gram = expected["data_curvature"]
            _require_finite(
                gram, what="expected curvature", remedy=_SENSITIVITY_REMEDY, names=kwargs["names"]
            )
            squares = np.clip(np.linalg.eigvalsh(0.5 * (gram + gram.T)), 0.0, None)[::-1]
        else:
            jacobian = np.asarray(info.whitened_jacobian, dtype=float)
            _require_finite(
                jacobian,
                what="whitened Jacobian",
                remedy=_SENSITIVITY_REMEDY,
                names=kwargs["names"],
                square=False,
            )
            squares = np.linalg.svd(jacobian, compute_uv=False) ** 2
        largest = float(squares[0]) if squares.size else 0.0
        rank = int(np.sum(squares > _NULL_TOL * largest))
        if rank < mode.shape[0]:
            # The residual Jacobian is the structural rank source. An observed
            # Hessian can acquire tiny positive curvature in a truly null
            # direction from solver noise or nonzero residual terms; that does
            # not make an MLE posterior proper.
            return laplace_posterior(mode, **expected, **kwargs)
    if curvature == "observed":
        observed = assemble_information(
            observed=_observed_curvature(problem, point), coordinate="x"
        ).observed
        posterior = laplace_posterior(mode, data_curvature=observed, **kwargs)
    else:
        posterior = laplace_posterior(mode, **expected, **kwargs)
    if not isinstance(posterior, LaplacePosterior):
        return posterior
    refusal = _evidence_refusal(problem)
    if refusal is not None:
        object.__setattr__(posterior, "_evidence_prior", refusal)
        return posterior
    gradient = _negative_log_joint_gradient(problem, point, (residual, info.whitened_jacobian))
    factor = np.asarray(posterior.precision_factor, dtype=float)
    decrement = float(np.linalg.norm(np.linalg.solve(factor, gradient)))
    object.__setattr__(posterior, "mode_gradient_norm", decrement)
    object.__setattr__(posterior, "stationarity_tolerance", _STATIONARITY_TOL)
    object.__setattr__(posterior, "_evidence_prior", _DeclaredPrior(problem))
    return posterior


def laplace(problem: Any, fit: Any, *, curvature: str = "observed", on_singular: str = "raise"):
    """Build the Laplace approximation, the Gaussian centered on a fit with the inverse curvature as covariance.

    For a `MAP` problem the curvature includes the prior's, and `log_evidence` is the Laplace
    log evidence. A multiple-shooting fit that fits its window starts has them integrated out.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] whose measurements
            all declare a likelihood.
        fit: The [`OptResult`][diff_soup.inference.OptResult] to center on, a parameter vector,
            or a mapping from every fitted handle to its value.
        curvature: `"observed"` or `"expected"`, as `kind=` of `information`.
        on_singular: What happens when the data leave a direction undetermined, `"raise"` to
            raise `ImproperPosteriorError` or `"subspace"` to return the determined directions.

    Returns:
        A `LaplacePosterior` indexed by handle, or an `IdentifiedSubspace` under
            `on_singular="subspace"` when a direction is undetermined.
    """
    if not isinstance(problem, EstimationProblem):
        raise TypeError("laplace requires an InferenceProblem.")
    require_likelihoods(problem, "laplace")
    if curvature == "ggn":
        raise ValueError('laplace curvature="ggn" was removed; use curvature="expected".')
    if curvature not in ("observed", "expected"):
        raise ValueError('laplace curvature must be "observed" or "expected".')
    post = _laplace_facade(problem, at=fit, curvature=curvature, on_singular=on_singular)
    if isinstance(post, LaplacePosterior):
        object.__setattr__(post, "parameter_schema", problem.parameter_schema)
    return post
