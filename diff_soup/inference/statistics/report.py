"""report.py — host-side identifiability analysis of a Fisher information matrix.

Everything here runs on host numpy after the differentiable pipeline has
produced its FIM or Jacobian: eigendecompositions are safe here precisely
because nothing in this module is ever differentiated. Run
`identifiability_report` BEFORE optimizing a design — an exact FIM null
direction means THIS experiment's data carry no information about that
parameter combination, so it must be pinned, re-parameterized, or targeted by
a redesigned protocol before the combination can be fitted.

Every rank or null statement here is **experiment-local**: it holds at the
evaluated parameter point, inputs, time grid, and observables. A different
protocol may or may not resolve a reported null; deciding that a combination
is unidentifiable for *every* admissible protocol is a structural analysis
this report does not perform.

* `identifiability_report` — spectrum (raw + regularized), correlation
  matrix, sloppy-direction loadings, condition number, and a sensitivity-rank
  detector that names degenerate parameter combinations in plain words.
* `gauss_newton_covariance` — the Gauss-Newton posterior covariance
  ``(Jᵀ Σ⁻¹ J)⁻¹`` from a residual Jacobian.
* `fit_posterior` — the same covariance for a completed fit, reusing the
  inference problem's own residual Jacobian.
* `pointwise_variance` — the pointwise uncertainty band of a
  function-valued parameter from its coefficient covariance.
* `function_report` — metric-aware modes of a function-valued block:
  the generalized eigenproblem of the (nuisance-profiled) block information
  against the basis Gram matrix or the prior precision, with pointwise
  uncertainty, contraction, and labeled pointwise/simultaneous bands.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

__all__ = [
    "NullDirection",
    "IdentifiabilityReport",
    "IdentifiedSubspace",
    "ImproperPosteriorError",
    "FunctionBand",
    "FunctionReport",
    "identifiability_report",
    "function_report",
    "gauss_newton_covariance",
    "fit_posterior",
    "pointwise_variance",
]


@dataclasses.dataclass(frozen=True)
class IdentifiedSubspace:
    """The parameter directions a rank-deficient experiment does determine. Frozen.

    Attributes:
        directions: Orthonormal identified directions as columns, shape
            ``(n, rank)``, best-determined first.
        variances: Gauss-Newton variance of each identified direction,
            shape ``(rank,)``.
        null_directions: Orthonormal undetermined directions as columns,
            shape ``(n, n - rank)``.
        names: Per-parameter labels for the rows of both bases.
    """

    directions: np.ndarray
    variances: np.ndarray
    null_directions: np.ndarray
    names: tuple = ()


class ImproperPosteriorError(ValueError):
    """A requested covariance does not exist without more information.

    Raised when the data leave parameter directions completely undetermined
    and no proper prior closes them: inventing finite variances there would
    misstate the uncertainty. The `subspace` attribute holds the
    `IdentifiedSubspace` describing what the data do determine.
    """

    def __init__(self, message: str, subspace: IdentifiedSubspace):
        """Store the message and the identified subspace the data determine."""
        super().__init__(message)
        self.subspace = subspace


@dataclasses.dataclass(frozen=True)
class NullDirection:
    """One parameter direction this experiment's data do not inform. Frozen.

    Attributes:
        eigenvalue: The (near-zero) FIM eigenvalue of the direction.
        loadings: The direction's unit loadings over the parameters.
        description: Plain-words statement naming the degenerate combination
            at the evaluated experiment.
    """

    eigenvalue: float
    loadings: np.ndarray
    description: str


@dataclasses.dataclass(frozen=True)
class IdentifiabilityReport:
    """Host-side identifiability evidence for one Fisher information matrix. Frozen.

    Attributes:
        names: The per-parameter direction labels.
        eigenvalues: Raw FIM eigenvalues, descending.
        eigenvalues_regularized: Eigenvalues of ``FIM + ridge·I``, descending.
        loadings: Orthonormal eigenvector columns matching ``eigenvalues``
            (column ``k`` is the direction of eigenvalue ``k``; the last
            columns are the sloppy directions).
        correlation: Parameter correlation matrix from the regularized
            covariance (unit diagonal).
        condition_number: ``λ_max / λ_min`` of the raw spectrum (``inf`` when
            the FIM is singular).
        sensitivity_rank: Number of eigenvalues above the relative null
            threshold — the count of parameter combinations THIS experiment's
            data determine at the evaluated point (a local sensitivity rank,
            not a structural-identifiability claim).
        null_directions: One `NullDirection` per eigenvalue at or below
            the threshold, worst first.
        notes: Plain-words remarks (e.g. that the ridge dominates a direction).
        evaluated_at: Provenance of the evaluation — parameter point label,
            channels/observables, and time-grid summary, as supplied by the
            caller. Empty when not provided.
    """

    names: tuple
    eigenvalues: np.ndarray
    eigenvalues_regularized: np.ndarray
    loadings: np.ndarray
    correlation: np.ndarray
    condition_number: float
    sensitivity_rank: int
    null_directions: tuple
    notes: tuple
    evaluated_at: dict = dataclasses.field(default_factory=dict)


def _combination_words(loadings: np.ndarray, names: tuple, share: float = 0.2) -> str:
    """Render a unit direction as a plain-words parameter combination."""
    terms = [f"{float(w):+.2f}·{names[i]}" for i, w in enumerate(loadings) if abs(float(w)) > share]
    return " ".join(terms) if terms else "a combination below the loading threshold"


# Relative magnitude below which a negative FIM eigenvalue counts as roundoff
# and is clipped to zero (with a report note); anything more negative is a
# materially indefinite input and is rejected.
_NEGATIVE_CLIP_RTOL = 1e-12


def identifiability_report(
    fim: Any = None,
    *,
    whitened_jacobian: Any = None,
    names: Optional[tuple] = None,
    ridge: float = 1e-10,
    null_tol: float = 1e-10,
    evaluated_at: Optional[dict] = None,
) -> IdentifiabilityReport:
    """Analyze the information carried by one experiment, on the host.

    Prefer passing ``whitened_jacobian`` (the ``1/σ``-weighted sensitivity
    matrix): its SVD resolves directions a formed FIM loses to floating-point
    rounding, because forming ``Jᵀ J`` squares the condition number. A formed
    FIM is accepted for compatibility and for information that only exists in
    matrix form.

    Args:
        fim: Symmetric PSD matrix, shape ``(n, n)`` (any array-like).
            Eigenvalues more negative than ``1e-12 · λ_max`` are rejected as
            materially indefinite; negative values within that tolerance are
            clipped to zero with a report note.
        whitened_jacobian: Whitened sensitivity matrix, shape ``(n_rows, n)``.
            Exactly one of ``fim`` and ``whitened_jacobian`` must be given.
            Reported eigenvalues are the squared singular values, so the
            spectrum has the same information units as the FIM path.
        names: Optional per-parameter labels (default ``theta[i]``).
        ridge: Diagonal regularization for the regularized spectrum and the
            correlation matrix.
        null_tol: Relative eigenvalue threshold: eigenvalues at or below
            ``null_tol · λ_max`` count as uninformed by this experiment.
        evaluated_at: Optional provenance mapping recorded verbatim on the
            report (e.g. parameter-point label, channel names, time-grid
            summary), tying every rank/null statement to the experiment that
            produced it.

    Returns:
        A frozen `IdentifiabilityReport`.

    Raises:
        ValueError: If neither or both inputs are given, ``fim`` is not a
            square PSD matrix, or ``names`` has the wrong length.
    """
    if (fim is None) == (whitened_jacobian is None):
        raise ValueError("identifiability_report: pass exactly one of fim and whitened_jacobian.")

    notes = []
    if whitened_jacobian is not None:
        j = np.asarray(whitened_jacobian, dtype=float)
        if j.ndim != 2:
            raise ValueError(
                f"identifiability_report: whitened_jacobian must be 2-D; got {j.shape}."
            )
        n = j.shape[1]
        _, s, vt = np.linalg.svd(j, full_matrices=True)
        eigenvalues = np.zeros(n)
        eigenvalues[: s.size] = s**2
        loadings = vt.T
    else:
        f = np.asarray(fim, dtype=float)
        if f.ndim != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(f"identifiability_report: fim must be square; got {f.shape}.")
        n = f.shape[0]
        f = 0.5 * (f + f.T)
        w, v = np.linalg.eigh(f)
        order = np.argsort(w)[::-1]
        eigenvalues = w[order]
        loadings = v[:, order]
        lam_max_raw = float(eigenvalues[0]) if n else 0.0
        neg = eigenvalues < 0.0
        if np.any(eigenvalues < -_NEGATIVE_CLIP_RTOL * max(lam_max_raw, 0.0)):
            raise ValueError(
                "identifiability_report: fim is materially indefinite "
                f"(most negative eigenvalue {eigenvalues[-1]:.2e}); pass a PSD "
                "information matrix or the whitened_jacobian it came from."
            )
        if np.any(neg):
            notes.append(
                f"clipped {int(np.sum(neg))} roundoff-negative eigenvalue(s) to "
                f"zero (within relative tolerance {_NEGATIVE_CLIP_RTOL:.0e})"
            )
            eigenvalues = np.clip(eigenvalues, 0.0, None)

    if names is None:
        names = tuple(f"theta[{i}]" for i in range(n))
    names = tuple(names)
    if len(names) != n:
        raise ValueError(f"identifiability_report: {len(names)} names for {n} parameters.")

    w_reg = eigenvalues + ridge

    lam_max = float(eigenvalues[0]) if n else 0.0
    lam_min = float(eigenvalues[-1]) if n else 0.0
    cond = float("inf") if lam_min <= 0.0 else lam_max / lam_min

    cov = (loadings / w_reg) @ loadings.T
    d = np.sqrt(np.clip(np.diag(cov), np.finfo(float).tiny, None))
    correlation = cov / np.outer(d, d)
    np.fill_diagonal(correlation, 1.0)

    threshold = null_tol * max(lam_max, 0.0)
    null_mask = eigenvalues <= threshold
    sensitivity_rank = int(np.sum(~null_mask))
    nulls = []
    for k in np.where(null_mask)[0][::-1]:
        combo = _combination_words(loadings[:, k], names)
        nulls.append(
            NullDirection(
                eigenvalue=float(eigenvalues[k]),
                loadings=loadings[:, k].copy(),
                description=(
                    f"this experiment's data do not inform the parameter "
                    f"combination {combo} (FIM eigenvalue {eigenvalues[k]:.2e}) "
                    "at the evaluated parameter point, inputs, and time grid; "
                    "pin or re-parameterize the combination to fit it here, or "
                    "redesign the protocol and re-evaluate; a different "
                    "experiment may or may not resolve it"
                ),
            )
        )

    if lam_max > 0.0 and ridge > null_tol * lam_max:
        notes.append(
            f"the ridge ({ridge:.1e}) exceeds the null threshold; regularized "
            "quantities may mask near-null directions"
        )

    return IdentifiabilityReport(
        names=names,
        eigenvalues=eigenvalues,
        eigenvalues_regularized=w_reg,
        loadings=loadings,
        correlation=correlation,
        condition_number=cond,
        sensitivity_rank=sensitivity_rank,
        null_directions=tuple(nulls),
        notes=tuple(notes),
        evaluated_at=dict(evaluated_at) if evaluated_at else {},
    )


def gauss_newton_covariance(
    jac: Any,
    *,
    sigma: Any = None,
    null_tol: float = 1e-10,
    names: tuple = (),
) -> np.ndarray:
    """Return the Gauss-Newton covariance ``(Jᵀ Σ⁻¹ J)⁻¹``, rank-aware.

    The covariance is assembled from the SVD of the whitened Jacobian, never
    from a matrix inverse. If the Jacobian leaves parameter directions
    undetermined, the covariance does not exist and this raises instead of
    silently inventing finite variances for those directions.

    Args:
        jac: Residual Jacobian, shape ``(n_rows, n_params)``. Pass the raw
            (unweighted) Jacobian together with ``sigma``, or an already
            ``1/σ``-weighted Jacobian with ``sigma=None``.
        sigma: Optional per-row noise standard deviations, broadcastable to
            ``(n_rows,)``.
        null_tol: Relative information threshold: directions whose squared
            singular value is at or below ``null_tol · s_max²`` count as
            undetermined.
        names: Optional per-parameter labels, attached to the identified
            subspace on failure.

    Returns:
        The ``(n_params, n_params)`` covariance (host numpy).

    Raises:
        ImproperPosteriorError: If any parameter direction is undetermined;
            the error's ``subspace`` attribute describes the directions the
            data do determine and their variances.
    """
    j = np.asarray(jac, dtype=float)
    if sigma is not None:
        j = j / np.broadcast_to(np.asarray(sigma, dtype=float), (j.shape[0],))[:, None]
    n = j.shape[1]
    _, s_thin, vt = np.linalg.svd(j, full_matrices=True)
    s = np.zeros(n)
    s[: s_thin.size] = s_thin
    s_max = float(s[0]) if n else 0.0
    identified = s**2 > null_tol * s_max**2
    rank = int(np.sum(identified))
    if rank < n:
        v = vt.T
        subspace = IdentifiedSubspace(
            directions=v[:, :rank],
            variances=1.0 / s[:rank] ** 2 if rank else np.zeros(0),
            null_directions=v[:, rank:],
            names=tuple(names),
        )
        raise ImproperPosteriorError(
            f"gauss_newton_covariance: the data leave {n - rank} of {n} "
            f"parameter directions undetermined (relative information "
            f"threshold {null_tol:.0e}), so no finite covariance exists. "
            "Close the undetermined directions with a proper prior, or pin or "
            "re-parameterize them; the determined directions and their "
            "variances are attached as the error's subspace attribute.",
            subspace,
        )
    return (vt.T / s**2) @ vt


def fit_posterior(problem: Any, x: Any, *, sigma: Any = None) -> np.ndarray:
    """Return the Gauss-Newton posterior covariance of a completed fit.

    Reuses the inference problem's own residual Jacobian at the converged
    parameters (the same forward-mode path the least-squares solver used).
    Fit residuals already carry their ``1/σ`` series weights, so ``sigma`` is
    normally left ``None``.

    Args:
        problem: The inference problem the optimizer solved.
        x: The converged flat parameter vector (e.g. ``OptResult.x``).
        sigma: Optional extra per-row standard deviations.

    Returns:
        The ``(n_theta, n_theta)`` posterior covariance (host numpy).

    Raises:
        ValueError: If the problem exposes no residual Jacobian.
        ImproperPosteriorError: If the fit leaves parameter directions
            undetermined (see `gauss_newton_covariance`).
    """
    from diff_soup.inference.fitting.diagnostics import _theta_jacobian

    jac = _theta_jacobian(problem, np.asarray(x))
    if jac is None or jac.size == 0:
        raise ValueError(
            "fit_posterior: the problem exposes no residual closures to "
            "differentiate; fit it with the standard least-squares path first."
        )
    theta_names = getattr(problem, "theta_names", None)
    n_theta = len(theta_names) if theta_names is not None else jac.shape[1]
    return gauss_newton_covariance(jac[:, :n_theta], sigma=sigma)


def pointwise_variance(basis: Any, covariance: Any) -> np.ndarray:
    """Return the pointwise standard deviation of a function-valued parameter.

    For a function ``f(x) = Σ_i θ_i φ_i(x)`` with coefficient covariance ``C``,
    the pointwise uncertainty is ``σ_f(x) = sqrt(φ(x)ᵀ C φ(x))`` — the
    uncertainty band to draw around the recovered function.

    Args:
        basis: Basis matrix ``Φ`` with ``Φ[k, i] = φ_i(x_k)``, shape
            ``(n_points, n_coefficients)``.
        covariance: Coefficient covariance, shape ``(n_coefficients, n_coefficients)``.

    Returns:
        The ``(n_points,)`` pointwise standard deviations (host numpy).
    """
    phi = np.asarray(basis, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    var = np.einsum("xi,ij,xj->x", phi, cov, phi)
    return np.sqrt(np.clip(var, 0.0, None))


# ---------------------------------------------------------------------------
# Metric-aware function reports
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FunctionBand:
    """An uncertainty band around a function estimate, labeled by calibration.

    The band is centered on zero deviation — add it to the estimated function
    curve. ``kind="pointwise"`` covers each point marginally at ``level``;
    ``kind="simultaneous"`` covers the whole curve jointly (a draw-based
    sup-t band), which is necessarily wider.

    Attributes:
        lower: Lower deviation at each evaluation point.
        upper: Upper deviation at each evaluation point.
        kind: ``"pointwise"`` or ``"simultaneous"``.
        level: The coverage level the band targets.
        calibration: How the band was calibrated, in plain words.
    """

    lower: np.ndarray
    upper: np.ndarray
    kind: str
    level: float
    calibration: str


@dataclasses.dataclass(frozen=True)
class FunctionReport:
    """Metric-aware identifiability of a function-valued parameter block. Frozen.

    Built by `function_report` from the generalized eigenproblem
    ``H v = λ M v``: ``H`` is the block's (nuisance-profiled) information and
    ``M`` the chosen metric — the basis Gram matrix (``metric="gram"``, so λ
    is information per unit of function L² norm) or the prior precision
    (``metric="prior"``, so λ is the information gain relative to the prior).
    The modes are M-orthonormal coefficient vectors, best-informed first;
    modes whose eigenvalue vanishes at the tolerance are the experiment's
    null *functions*.

    Every statement is experiment-local: it holds at the evaluated parameter
    point, inputs, time grid, and observables of the information matrix that
    built it.

    Attributes:
        block: The parameter block label.
        names: Per-coefficient labels of the block.
        metric: ``"gram"`` or ``"prior"``.
        scale: ``"raw"`` (the basis expansion itself — the log scale for an
            exp-transformed function) or ``"natural"`` (the evaluation
            Jacobian of the block's output transform is applied).
        eigenvalues: Generalized eigenvalues, descending.
        modes: M-orthonormal eigenmode coefficient vectors as columns,
            aligned with ``eigenvalues``.
        null_functions: The trailing mode columns whose eigenvalue is at or
            below the null tolerance — function shapes this experiment does
            not inform.
        provenance: Curvature kind, nuisance handling, and tolerances used.
    """

    block: str
    names: tuple
    metric: str
    scale: str
    eigenvalues: np.ndarray
    modes: np.ndarray
    null_functions: np.ndarray
    provenance: dict
    _basis: Any = dataclasses.field(repr=False, default=None)
    _at: Optional[np.ndarray] = dataclasses.field(repr=False, default=None)
    _null_tol: float = dataclasses.field(repr=False, default=1e-10)

    # -- evaluation helpers -------------------------------------------------

    def _basis_rows(self, cs: Any) -> np.ndarray:
        if self._basis is None:
            raise ValueError(
                "this FunctionReport was built without a basis, so it cannot "
                "evaluate curves; rebuild it with function_report(..., "
                "basis=<the block's Legendre>)."
            )
        cs = np.atleast_1d(np.asarray(cs, dtype=float))
        lo, hi = getattr(self._basis, "interval", (0.0, 1.0))
        x = 2.0 * (cs - lo) / (hi - lo) - 1.0
        return np.polynomial.legendre.legvander(x, len(self.names) - 1)

    def _jacobian(self, cs: Any, rows: np.ndarray) -> np.ndarray:
        """The evaluation Jacobian |d transform / d expansion| at each point."""
        if self.scale != "natural":
            return np.ones(rows.shape[0])
        transform = getattr(self._basis, "transform", None)
        if transform is None:
            return np.ones(rows.shape[0])
        import jax

        cs = np.atleast_1d(np.asarray(cs, dtype=float))
        raw = rows @ self._at
        jac = [float(jax.grad(lambda p: transform(c, p))(float(p0))) for c, p0 in zip(cs, raw)]
        return np.abs(np.asarray(jac))

    def _weighted_rows(self, cs: Any) -> np.ndarray:
        rows = self._basis_rows(cs)
        return rows * self._jacobian(cs, rows)[:, None]

    def _mode_variances(self) -> np.ndarray:
        """Per-mode variance weights: ``1/λ`` (gram) or ``1/(λ+1)`` (prior)."""
        lam = np.clip(self.eigenvalues, 0.0, None)
        if self.metric == "prior":
            return 1.0 / (lam + 1.0)
        var = np.full(lam.shape, np.inf)
        lam_max = float(lam[0]) if lam.size else 0.0
        keep = lam > self._null_tol * max(lam_max, 0.0)
        var[keep] = 1.0 / lam[keep]
        return var

    # -- the report surface --------------------------------------------------

    def mode_curves(self, cs: Any) -> np.ndarray:
        """Evaluate every mode's function shape, shape ``(n_modes, n_points)``."""
        return (self._weighted_rows(cs) @ self.modes).T

    def pointwise_sd(self, cs: Any) -> np.ndarray:
        """The pointwise function standard deviation at each evaluation point.

        ``metric="gram"``: the data-only profiled uncertainty — infinite
        wherever a null function has support (the experiment does not bound
        the function value there; no finite number is invented).
        ``metric="prior"``: the posterior uncertainty ``(H + Λ₀)^{-1}``,
        finite everywhere the prior is proper.
        """
        q = self._weighted_rows(cs) @ self.modes
        var_modes = self._mode_variances()
        finite = np.isfinite(var_modes)
        var = np.einsum("xi,i->x", q[:, finite] ** 2, var_modes[finite])
        sd = np.sqrt(var)
        if not np.all(finite):
            null_mass = np.sum(q[:, ~finite] ** 2, axis=1)
            total = np.sum(q**2, axis=1)
            sd = np.where(null_mass > 1e-12 * np.maximum(total, 1e-300), np.inf, sd)
        return sd

    def pointwise_contraction(self, cs: Any) -> np.ndarray:
        """The posterior/prior pointwise standard-deviation ratio.

        ``1`` means the experiment taught nothing about the function value at
        that point (prior-dominated); values near ``0`` mean the data dominate.

        Raises:
            ValueError: Unless the report was built with ``metric="prior"``
                (contraction is relative to the prior by definition).
        """
        if self.metric != "prior":
            raise ValueError(
                "pointwise_contraction compares posterior to prior, so it needs "
                'a report built with metric="prior" (and an information result '
                "carrying the prior precision)."
            )
        q = self._weighted_rows(cs) @ self.modes
        lam = np.clip(self.eigenvalues, 0.0, None)
        var_post = np.einsum("xi,i->x", q**2, 1.0 / (lam + 1.0))
        var_prior = np.sum(q**2, axis=1)
        return np.sqrt(var_post / var_prior)

    def bands(
        self,
        cs: Any,
        *,
        kind: str = "pointwise",
        level: float = 0.95,
        draws: int = 2000,
        seed: int = 0,
    ) -> FunctionBand:
        """An uncertainty band for the function over the evaluation points.

        ``kind="pointwise"`` uses the normal quantile of `pointwise_sd`
        at each point separately. ``kind="simultaneous"`` is draw-based: it
        samples coefficient vectors from the report's covariance, standardizes
        each sampled curve by the pointwise standard deviation, and calibrates
        the band width on the ``level`` quantile of the per-draw supremum —
        joint coverage over the whole grid.

        Raises:
            ValueError: For an unknown kind, a level outside ``(0, 1)``, or a
                simultaneous band where a null function makes the pointwise
                uncertainty infinite (close it with ``metric="prior"``).
        """
        from scipy.stats import norm

        if kind not in ("pointwise", "simultaneous"):
            raise ValueError(f"band kind must be 'pointwise' or 'simultaneous'; got {kind!r}.")
        if not 0.0 < float(level) < 1.0:
            raise ValueError(f"band level must satisfy 0 < level < 1; got {level!r}.")
        sd = self.pointwise_sd(cs)
        if kind == "pointwise":
            upper = float(norm.ppf(0.5 + level / 2.0)) * sd
            return FunctionBand(
                lower=-upper,
                upper=upper,
                kind=kind,
                level=float(level),
                calibration="pointwise normal quantile of the profiled sd",
            )
        if not np.all(np.isfinite(sd)):
            raise ValueError(
                "a simultaneous band is undefined where a null function makes "
                "the pointwise uncertainty infinite; close the data-null "
                'directions with a proper prior (metric="prior") first.'
            )
        var_modes = self._mode_variances()
        rows = self._weighted_rows(cs)
        rng = np.random.default_rng(seed)
        samples = self.modes @ (
            np.sqrt(var_modes)[:, None] * rng.standard_normal((len(self.names), int(draws)))
        )
        curves = rows @ samples  # (n_points, draws)
        sup = np.max(np.abs(curves) / sd[:, None], axis=0)
        q = float(np.quantile(sup, level))
        return FunctionBand(
            lower=-q * sd,
            upper=q * sd,
            kind=kind,
            level=float(level),
            calibration=f"draw-based sup-t calibration over the grid ({int(draws)} draws)",
        )


def _resolve_block_columns(names: tuple, block: Any) -> tuple:
    """Resolve a block spec to ``(label, column indices)`` over ``names``."""
    if isinstance(block, str):
        cols = [
            i
            for i, n in enumerate(names)
            if n == block or (n.startswith(f"{block}[") and n.endswith("]"))
        ]
        if not cols:
            raise ValueError(
                f"function_report block {block!r} matches no parameter scalars; "
                f"the information matrix labels its parameters {list(names)}."
            )
        return block, tuple(cols)
    cols = tuple(int(i) for i in block)
    if not cols:
        raise ValueError("function_report block indices must name at least one column.")
    return "block", cols


def function_report(
    information: Any,
    block: Any,
    *,
    metric: str = "gram",
    basis: Any = None,
    gram: Any = None,
    nuisance: str = "schur",
    scale: str = "raw",
    at: Any = None,
    null_tol: float = 1e-10,
) -> FunctionReport:
    """Report how well the data determine one function-valued parameter, measured in the function's own norm.

    It takes the function's coefficients out of an information matrix and reports the
    eigenvalues of their information measured against `metric`.

    Args:
        information: The result of `information`.
        block: The name of the function handle, or the column indices of its coefficients.
        metric: `"gram"` to measure against the Gram matrix of the basis, or `"prior"` against
            the prior precision of the function.
        basis: The function's basis, such as a `Legendre`, needed for the curves and used for
            the Gram matrix when `gram` is `None`.
        gram: The Gram matrix of the function's basis.
        nuisance: `"schur"` to eliminate the other parameters, or `"slice"` to hold them fixed.
        scale: `"raw"` for curves of the raw function, or `"natural"` for curves through the
            basis's output transform.
        at: The function's coefficient values, needed for `scale="natural"`.
        null_tol: The eigenvalue, relative to the largest, at or below which a mode counts as
            undetermined.

    Returns:
        A `FunctionReport` with the `eigenvalues`, the `modes`, and the undetermined
            `null_functions`.
    """
    import scipy.linalg

    if metric not in ("gram", "prior"):
        raise ValueError(f"function_report metric must be 'gram' or 'prior'; got {metric!r}.")
    if nuisance not in ("schur", "slice"):
        raise ValueError(
            f"function_report nuisance must be 'schur' (profile) or 'slice'; got {nuisance!r}."
        )
    if scale not in ("raw", "natural"):
        raise ValueError(f"function_report scale must be 'raw' or 'natural'; got {scale!r}.")
    if scale == "natural" and at is None:
        raise ValueError(
            "scale='natural' applies the output transform's evaluation "
            "Jacobian, which depends on the function itself; pass at=<the "
            "block's coefficient values>."
        )

    curvature_kind = None
    for kind in ("expected", "ggn", "observed"):
        matrix = getattr(information, kind, None)
        if matrix is not None:
            curvature_kind = kind
            break
    if curvature_kind is None:
        raise ValueError(
            "function_report needs a stored curvature matrix; the information "
            "result carries none of expected/ggn/observed."
        )
    h_full = np.asarray(matrix, dtype=float)
    n = h_full.shape[0]
    names = tuple(getattr(information, "provenance", {}).get("theta_names", ())) or tuple(
        f"theta[{i}]" for i in range(n)
    )
    label, cols = _resolve_block_columns(names, block)
    cols_arr = np.asarray(cols, dtype=int)
    nuis = np.asarray([i for i in range(n) if i not in set(cols)], dtype=int)

    h_bb = h_full[np.ix_(cols_arr, cols_arr)]
    nuisance_rank_deficient = False
    if nuisance == "schur" and nuis.size:
        h_nn = h_full[np.ix_(nuis, nuis)]
        h_bn = h_full[np.ix_(cols_arr, nuis)]
        w, v = np.linalg.eigh(0.5 * (h_nn + h_nn.T))
        w_max = float(w[-1]) if w.size else 0.0
        keep = w > max(w_max, 0.0) * 1e-12
        nuisance_rank_deficient = bool(np.any(~keep))
        pinv = (v[:, keep] / w[keep]) @ v[:, keep].T
        h_eff = h_bb - h_bn @ pinv @ h_bn.T
    else:
        h_eff = h_bb
    h_eff = 0.5 * (h_eff + h_eff.T)

    if metric == "gram":
        if gram is None and basis is not None:
            gram = basis.gram()
        if gram is None:
            raise ValueError(
                'metric="gram" needs the block\'s function metric; pass '
                "basis=<the block's Legendre> or an explicit gram= matrix."
            )
        m = np.asarray(gram, dtype=float)
    else:
        prior_precision = getattr(information, "prior_precision", None)
        if prior_precision is None:
            raise ValueError(
                'metric="prior" compares information to the prior, so the '
                "information result must carry prior_precision (build it with "
                "a prior)."
            )
        m = np.asarray(prior_precision, dtype=float)[np.ix_(cols_arr, cols_arr)]
    m = 0.5 * (m + m.T)
    if m.shape != h_eff.shape:
        raise ValueError(
            f"the metric block has shape {m.shape} but the parameter block has "
            f"{len(cols)} coefficient(s); they must match."
        )
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"the {metric} metric block is not positive definite; a function "
            "metric must be an SPD matrix (a basis Gram matrix or a proper "
            "prior precision)."
        ) from exc

    eigenvalues, modes = scipy.linalg.eigh(h_eff, m)
    eigenvalues = eigenvalues[::-1]
    modes = np.ascontiguousarray(modes[:, ::-1])
    lam_max = max(float(eigenvalues[0]), 0.0)
    n_null = int(np.sum(eigenvalues <= null_tol * lam_max))

    return FunctionReport(
        block=label,
        names=tuple(names[i] for i in cols),
        metric=metric,
        scale=scale,
        eigenvalues=eigenvalues,
        modes=modes,
        null_functions=np.ascontiguousarray(modes[:, len(cols) - n_null :]),
        provenance={
            "curvature_kind": curvature_kind,
            "nuisance": nuisance,
            "nuisance_names": tuple(names[i] for i in nuis),
            "nuisance_rank_deficient": nuisance_rank_deficient,
            "null_tol": float(null_tol),
            "n_null": n_null,
        },
        _basis=basis,
        _at=None if at is None else np.asarray(at, dtype=float),
        _null_tol=float(null_tol),
    )
