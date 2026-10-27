"""information.py — one container that keeps curvature kinds distinct.

Several different matrices are casually called "the Hessian" in fitting
folklore; they are different objects with different meanings, and this module
never conflates them:

* the **whitened sensitivity** ``J`` (``Σ^{-1/2}``-weighted residual
  Jacobian) — the primitive most analyses should start from;
* the **expected data Fisher information** ``E[-∇² log p(y|θ)] = Jᵀ J`` —
  an average over hypothetical data, always positive semidefinite;
* the **observed curvature** ``-∇² log p(y_obs|θ)`` — evaluated at the data
  actually seen; away from the noise floor it differs from the expected
  information by residual-weighted second-derivative terms and can be
  indefinite;
* the **generalized Gauss-Newton (GGN) curvature** — the first-order part of
  the observed curvature, positive semidefinite by construction;
* the **prior precision** and the **posterior precision** approximation
  (data curvature plus prior precision).

[InformationResult][diff_soup.inference.statistics.information.InformationResult] stores each kind separately, alongside the
coordinate the curvature lives in and the provenance/tolerances of the
evaluation. It is an analysis container, not a differentiable object: its
methods run host-side numpy (rank, delegation to
[identifiability_report][diff_soup.inference.statistics.report.identifiability_report]), but arrays
passed in are stored as given — JAX arrays stay JAX arrays.

Build instances with [assemble_information][], which validates shapes and
symmetry, forms the expected information from the whitened sensitivity when
needed, and assembles the posterior precision when a prior is present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup.inference.fitting.inference_problem import _is_structural_schema
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.statistics.estimation import MAP, EstimationProblem
from diff_soup.inference.statistics.experiment import (
    _channel_likelihood_view,
)
from diff_soup.inference.statistics.measurement import require_likelihoods

__all__ = [
    "InformationResult",
    "assemble_information",
    "information",
]

# Relative asymmetry within which an input matrix is silently symmetrized
# (attributed to roundoff); anything worse is rejected. Recorded on every
# result under tolerances["symmetry_rtol"].
# Forward-over-forward Hessians through adaptive implicit solves accumulate
# solver-level directional noise. Asymmetry below this relative scale is
# numerical and is removed explicitly; larger disagreement remains an error.
# The bound is loose enough for a float64 Hessian composed through an
# adaptive implicit solve, and tight enough that a curvature whose two
# mixed partials genuinely disagree is still refused.
_SYMMETRY_RTOL = 1e-6

# Relative magnitude below which a negative eigenvalue of a supplied
# information matrix counts as roundoff during rank computation; anything
# more negative means the matrix is materially indefinite and cannot serve
# as an information matrix.
_INDEFINITE_RTOL = 1e-12

# Fixed vocabulary and ordering of kinds(); "prior" and "posterior" name the
# precision fields, the rest name the like-named curvature fields.
_KIND_FIELDS = (
    ("whitened_jacobian", "whitened_jacobian"),
    ("expected", "expected"),
    ("observed", "observed"),
    ("ggn", "ggn"),
    ("prior", "prior_precision"),
    ("posterior", "posterior_precision"),
)

_DEFAULT_NULL_TOL = 1e-10


class InformationResult(eqx.Module):
    """Local curvature at one evaluation point, each kind kept distinct.

    Every field is optional except the labels: an instance carries exactly
    the kinds that were computed, and [kinds][diff_soup.inference.statistics.information.InformationResult.kinds] names them. Nothing here
    is ever labeled "the Hessian" without qualification — the expected data
    Fisher information, the observed negative-log-likelihood curvature, and
    the generalized Gauss-Newton curvature are stored and reported as three
    different objects.

    Construct instances through [assemble_information][], which performs
    the shape/symmetry validation and derived-field assembly; the raw
    constructor stores fields verbatim.

    Attributes:
        whitened_jacobian: Whitened sensitivity matrix ``Σ^{-1/2} ∂r/∂θ``,
            shape ``(n_rows, n)``, or ``None``. When present it is the rank
            source: its SVD resolves directions that forming a Gram matrix
            loses to floating-point rounding.
        expected: Expected data Fisher information ``E[-∇² log p(y|θ)]``,
            shape ``(n, n)``, or ``None``.
        observed: Observed negative-log-likelihood curvature
            ``-∇² log p(y_obs|θ)`` at the data actually seen, or ``None``.
            May be indefinite away from the noise floor; it is never used as
            a rank source.
        ggn: Generalized Gauss-Newton curvature, or ``None``.
        prior_precision: Prior precision matrix, or ``None``.
        posterior_precision: Data curvature plus prior precision, assembled
            when both are known, else ``None``. The data term is the GGN when
            present, else the expected information, else the observed
            curvature.
        coordinate: Coordinate the curvature lives in: ``"z"`` (inference)
            or ``"x"`` (physical).
        provenance: Channel/experiment provenance of the evaluation
            (parameter-point label, channel names, time-grid summary, ...).
        tolerances: Numerical-rank and clipping tolerances used. Always
            records ``symmetry_rtol``; a recorded ``null_tol`` becomes the
            default rank/report threshold.
    """

    whitened_jacobian: Optional[Array] = None
    expected: Optional[Array] = None
    observed: Optional[Array] = None
    ggn: Optional[Array] = None
    prior_precision: Optional[Array] = None
    posterior_precision: Optional[Array] = None
    coordinate: str = eqx.field(static=True, default="x")
    provenance: dict = eqx.field(static=True, default_factory=dict)
    tolerances: dict = eqx.field(static=True, default_factory=dict)

    # -- rank ---------------------------------------------------------------

    def rank_source(self) -> str:
        """Name the field [rank][diff_soup.inference.statistics.information.InformationResult.rank] computes from.

        Returns:
            ``"whitened_jacobian"`` when the whitened sensitivity is stored
            (rank from its SVD), else ``"expected"``, else ``"ggn"`` (rank
            from eigenvalues of the formed matrix).

        Raises:
            ValueError: If none of those fields is populated. The observed
                curvature is deliberately not a fallback — it can be
                indefinite and does not measure information.
        """
        if self.whitened_jacobian is not None:
            return "whitened_jacobian"
        if self.expected is not None:
            return "expected"
        if self.ggn is not None:
            return "ggn"
        raise ValueError(
            "InformationResult.rank: no rank source is stored; populate "
            "whitened_jacobian, expected, or ggn. The observed curvature is "
            "not a rank source because it can be indefinite."
        )

    def rank(self, *, null_tol: Optional[float] = None) -> int:
        """Return the local sensitivity rank of the data information.

        Counts the directions whose information exceeds ``null_tol`` times
        the largest. The source is chosen by [rank_source][diff_soup.inference.statistics.information.InformationResult.rank_source]: the SVD of
        the whitened sensitivity when stored (squared singular values, so the
        threshold has information units), else the eigenvalues of the
        ``expected`` (or ``ggn``) matrix — a formed matrix squares the
        condition number, so directions below roughly ``1e-16`` relative are
        unresolvable on that path.

        Args:
            null_tol: Relative information threshold. Defaults to the
                container's recorded ``tolerances["null_tol"]``, else 1e-10.

        Returns:
            The number of informed directions at the evaluated experiment.

        Raises:
            ValueError: If no rank source is stored, or the source matrix is
                materially indefinite (more negative than roundoff allows).
        """
        source = self.rank_source()
        tol = self._resolved_null_tol(null_tol)
        if source == "whitened_jacobian":
            j = np.asarray(self.whitened_jacobian, dtype=float)
            s = np.linalg.svd(j, compute_uv=False)
            eigenvalues = np.zeros(j.shape[1])
            eigenvalues[: s.size] = s**2
        else:
            m = np.asarray(getattr(self, source), dtype=float)
            w = np.linalg.eigvalsh(0.5 * (m + m.T))
            lam_max = max(float(w[-1]), 0.0) if w.size else 0.0
            if w.size and float(w[0]) < -_INDEFINITE_RTOL * lam_max:
                raise ValueError(
                    f"InformationResult.rank: the {source} matrix is "
                    f"materially indefinite (most negative eigenvalue "
                    f"{float(w[0]):.2e}); an information matrix must be "
                    "positive semidefinite up to roundoff."
                )
            eigenvalues = np.clip(w, 0.0, None)
        lam_max = float(eigenvalues.max()) if eigenvalues.size else 0.0
        return int(np.sum(eigenvalues > tol * lam_max))

    # -- introspection -------------------------------------------------------

    def kinds(self) -> tuple:
        """Name the populated kinds, in a fixed order.

        Returns:
            A subset of ``("whitened_jacobian", "expected", "observed",
            "ggn", "prior", "posterior")``, where ``"prior"`` and
            ``"posterior"`` name the precision fields and the rest name the
            like-named curvature fields.
        """
        return tuple(kind for kind, field in _KIND_FIELDS if getattr(self, field) is not None)

    # -- helpers -------------------------------------------------------------

    def _resolved_null_tol(self, null_tol: Optional[float]) -> float:
        """Resolve an explicit threshold against the recorded default."""
        if null_tol is not None:
            return float(null_tol)
        return float(self.tolerances.get("null_tol", _DEFAULT_NULL_TOL))


def _symmetrized(matrix: Any) -> Any:
    """Average a composed curvature with its transpose.

    A Hessian composed by differentiating twice through an adaptive implicit
    solve carries directional roundoff, so its two mixed partials differ in
    the last few digits. Averaging removes that difference exactly; the
    symmetry check in :func:`assemble_information` remains the guard on
    curvature a caller supplies directly.
    """
    return 0.5 * (matrix + matrix.T)


def _validated_matrix(value: Any, name: str, n: Optional[int]) -> tuple:
    """Validate one square symmetric input; return (symmetrized, size).

    Symmetrizes within the documented relative roundoff tolerance and rejects
    anything worse. Preserves the array type (JAX arrays stay JAX arrays);
    non-array inputs are coerced to host numpy.
    """
    if not (hasattr(value, "ndim") and hasattr(value, "T")):
        value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(
            f"assemble_information: {name} must be a square matrix; got shape {value.shape}."
        )
    if n is not None and value.shape[0] != n:
        raise ValueError(
            f"assemble_information: {name} has {value.shape[0]} parameters "
            f"but earlier inputs have {n}; all inputs must share one "
            "parameter dimension."
        )
    host = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(host))) if host.size else 0.0
    asymmetry = float(np.max(np.abs(host - host.T))) if host.size else 0.0
    # The tolerance is relative to the matrix's own scale, so a matrix of
    # small entries is held to the same relative standard as a large one. A
    # matrix whose largest entry is below the smallest normal float carries
    # no resolvable asymmetry and is taken as symmetric.
    if scale > float(np.finfo(float).tiny):
        relative_asymmetry = asymmetry / scale
    else:
        relative_asymmetry = 0.0
    if relative_asymmetry > _SYMMETRY_RTOL:
        raise ValueError(
            f"assemble_information: {name} is not symmetric (relative "
            f"asymmetry {relative_asymmetry:.2e} "
            f"exceeds {_SYMMETRY_RTOL:.0e}); pass a symmetric matrix."
        )
    if asymmetry > 0.0:
        value = 0.5 * (value + value.T)
    return value, value.shape[0]


def assemble_information(
    *,
    whitened_jacobian: Any = None,
    expected: Any = None,
    observed: Any = None,
    ggn: Any = None,
    prior_precision: Any = None,
    coordinate: str = "x",
    provenance: Optional[dict] = None,
    tolerances: Optional[dict] = None,
) -> InformationResult:
    """Validate and assemble an [InformationResult][diff_soup.inference.statistics.information.InformationResult].

    Assembly rules:

    * When ``expected`` is not supplied but ``whitened_jacobian`` is, the
      expected data Fisher information is formed as ``Jᵀ J``. The formed
      matrix is a dense convenience for small parameter counts only; rank and
      spectrum analysis still come from the SVD of ``J``, which the formed
      matrix cannot match (forming the Gram matrix squares the condition
      number).
    * ``posterior_precision`` is assembled as data curvature plus
      ``prior_precision`` when a prior is given and at least one data
      curvature exists; the data term is ``ggn`` when present, else
      ``expected``, else ``observed``. Without a prior it stays ``None``.
    * Every matrix input must be square, of one shared parameter dimension,
      and symmetric: asymmetry within 1e-6 of the matrix's own largest entry
      is attributed to roundoff and symmetrized, anything worse is rejected.

    Arrays are stored as given (after symmetrization) — JAX arrays are not
    copied to host.

    Args:
        whitened_jacobian: Whitened sensitivity matrix, shape ``(n_rows, n)``.
        expected: Expected data Fisher information, shape ``(n, n)``.
        observed: Observed negative-log-likelihood curvature, shape
            ``(n, n)`` — the curvature at the data actually seen, distinct
            from the expected information and from the GGN.
        ggn: Generalized Gauss-Newton curvature, shape ``(n, n)``.
        prior_precision: Prior precision matrix, shape ``(n, n)``.
        coordinate: ``"z"`` (inference coordinate) or ``"x"`` (physical).
        provenance: Channel/experiment provenance recorded verbatim.
        tolerances: Numerical-rank/clipping tolerances; a ``null_tol`` entry
            becomes the default threshold of ``rank()`` and ``report()``.
            The symmetry policy actually applied is always recorded as
            ``symmetry_rtol``.

    Returns:
        The frozen [InformationResult][diff_soup.inference.statistics.information.InformationResult].

    Raises:
        ValueError: On an empty call, an unknown coordinate, a non-square or
            shape-mismatched input, or a materially asymmetric matrix.
    """
    if coordinate not in ("z", "x"):
        raise ValueError(
            f"assemble_information: coordinate must be 'z' or 'x'; got {coordinate!r}."
        )
    if all(
        value is None for value in (whitened_jacobian, expected, observed, ggn, prior_precision)
    ):
        raise ValueError(
            "assemble_information: nothing to assemble; pass at least one of "
            "whitened_jacobian, expected, observed, ggn, prior_precision."
        )

    n: Optional[int] = None
    if whitened_jacobian is not None:
        if not (hasattr(whitened_jacobian, "ndim") and hasattr(whitened_jacobian, "T")):
            whitened_jacobian = np.asarray(whitened_jacobian, dtype=float)
        if whitened_jacobian.ndim != 2:
            raise ValueError(
                "assemble_information: whitened_jacobian must be 2-D "
                f"(n_rows, n_params); got shape {whitened_jacobian.shape}."
            )
        n = whitened_jacobian.shape[1]

    validated = {}
    for name, value in (
        ("expected", expected),
        ("observed", observed),
        ("ggn", ggn),
        ("prior_precision", prior_precision),
    ):
        if value is None:
            validated[name] = None
            continue
        validated[name], n = _validated_matrix(value, name, n)

    if validated["expected"] is None and whitened_jacobian is not None:
        validated["expected"] = whitened_jacobian.T @ whitened_jacobian

    posterior_precision = None
    if validated["prior_precision"] is not None:
        data = next(
            (
                validated[name]
                for name in ("ggn", "expected", "observed")
                if validated[name] is not None
            ),
            None,
        )
        if data is not None:
            posterior_precision = data + validated["prior_precision"]

    recorded_tolerances = dict(tolerances) if tolerances else {}
    recorded_tolerances["symmetry_rtol"] = _SYMMETRY_RTOL

    return InformationResult(
        whitened_jacobian=whitened_jacobian,
        expected=validated["expected"],
        observed=validated["observed"],
        ggn=validated["ggn"],
        prior_precision=validated["prior_precision"],
        posterior_precision=posterior_precision,
        coordinate=coordinate,
        provenance=dict(provenance) if provenance else {},
        tolerances=recorded_tolerances,
    )


def _resolve_at(predictor: Any, at: Any) -> Any:
    """Resolve the evaluation point of an experiment analysis to a flat vector.

    ``None`` evaluates at the declaration's own parameter values; a mapping
    overrides named scalars of the flat theta vector; an array is used as the
    flat vector directly.
    """
    import jax.numpy as jnp

    x0 = jnp.asarray(predictor.x0)
    if at is None:
        return x0
    if isinstance(at, dict):
        names = list(predictor.theta_names)
        unknown = sorted(set(at) - set(names))
        if unknown:
            raise ValueError(
                f"information/laplace at= names unknown parameters {unknown}; "
                f"the learnable scalars are {sorted(names)}."
            )
        x = np.asarray(x0, dtype=float).copy()
        for name, value in at.items():
            value = np.asarray(value, dtype=float)
            if value.shape != ():
                raise ValueError(
                    f"at[{name!r}] must be a scalar override; got shape {value.shape}."
                )
            x[names.index(name)] = float(value)
        return jnp.asarray(x)
    at = jnp.asarray(at)
    if at.shape != x0.shape:
        raise ValueError(
            f"information/laplace at= must be a flat vector of shape {x0.shape} "
            f"(order {predictor.theta_names}); got {at.shape}."
        )
    return at


def _expected_information_core(predictor: Any, x: Any, response: tuple) -> tuple:
    """Evaluate predictions/Jacobians and assemble exact expected information.

    A likelihood's expected-information action is evaluated at the prediction
    and then held fixed while its square-root action whitens the raw prediction
    Jacobian. This is the chain rule ``J.T @ I(mu) @ J``. Differentiating a
    diagnostic standardized residual would instead differentiate the weights
    themselves and is only valid for fixed-covariance location families.

    Returns ``(predictions, jacobians, whitened_jacobian, expected)``.
    """
    import jax
    import jax.numpy as jnp

    names = tuple(c.name for c in response)

    def selected(x_flat):
        predictions, failed = predictor(x_flat)
        values = tuple(predictions[name] for name in names)
        return values, (values, failed)

    jacobians, (values, failed) = jax.jacfwd(selected, has_aux=True)(x)
    if bool(failed):
        raise RuntimeError(
            "information: the forward solve failed at the evaluation point, so "
            "expected information is undefined there."
        )

    rows = []
    for channel, predicted, jacobian in zip(response, values, jacobians):
        likelihood, context = _channel_likelihood_view(channel, predicted)
        metric = likelihood.expected_information(predicted, context=context)
        whitened = jax.vmap(
            lambda direction: metric.whiten(direction),
            in_axes=-1,
            out_axes=-1,
        )(jacobian)
        rows.append(jnp.reshape(whitened, (-1, x.shape[0])))

    whitened_jacobian = jnp.concatenate(rows, axis=0)
    expected = whitened_jacobian.T @ whitened_jacobian
    predictions = dict(zip(names, values))
    jacobian_by_name = dict(zip(names, jacobians))
    return predictions, jacobian_by_name, whitened_jacobian, expected


def _coordinate_names(problem: EstimationProblem) -> tuple[str, ...]:
    """Return one unambiguous label per flat physical coordinate."""
    names: list[str | None] = [None] * problem.n_params
    for entry in problem.parameter_schema.entries:
        for occurrence in entry.occurrences:
            owner = "" if occurrence.owner is None else f"@{occurrence.owner}"
            for component, column in enumerate(occurrence.columns):
                suffix = "" if entry.size == 1 else f"[{component}]"
                names[int(column)] = f"{entry.handle.name}{owner}{suffix}"
    if any(name is None for name in names):
        raise ValueError("parameter schema does not label every fitted coordinate")
    return tuple(name for name in names if name is not None)


def _point(problem: EstimationProblem, at: Any) -> jax.Array:
    """Resolve an explicit vector or same-schema fit result."""
    if at is None:
        return jnp.asarray(problem.x0)
    if isinstance(at, OptResult):
        schema = at.parameter_schema
        if schema is None:
            raise ValueError(
                "UQ analysis needs an OptResult carrying a parameter schema, or a flat "
                "parameter vector; this OptResult carries neither."
            )
        # A structural schema records one problem's own column layout rather than
        # the content of a declaration, so it is never comparable with a Study
        # fingerprint; the shape check below is what validates such a point.
        if (
            not _is_structural_schema(schema)
            and schema.fingerprint != problem.parameter_schema.fingerprint
        ):
            raise ValueError(
                f"UQ analysis OptResult carries parameter schema fingerprint "
                f"{schema.fingerprint!r}, but this problem's schema fingerprint is "
                f"{problem.parameter_schema.fingerprint!r}."
            )
        at = at.params
    if isinstance(at, Mapping):
        point = jnp.asarray(problem.x0)
        assigned = set()
        for handle, value in at.items():
            entry = problem.parameter_schema.entry(handle)
            assigned.add(entry.handle.identity)
            for occurrence in entry.occurrences:
                block = value[occurrence.owner] if occurrence.owner is not None else value
                block = jnp.asarray(block).reshape(-1)
                if block.size != entry.size:
                    raise ValueError(f"point for {entry.handle.name!r} needs {entry.size} values.")
                point = point.at[jnp.asarray(occurrence.columns)].set(block)
        if assigned != {entry.handle.identity for entry in problem.parameter_schema.entries}:
            raise ValueError("an analysis point requires a value for every fitted handle.")
        return point
    point = jnp.asarray(at)
    if point.shape != problem.x0.shape:
        raise ValueError(f"UQ analysis point has shape {point.shape}; expected {problem.x0.shape}.")
    return point


def _recorded_data_jacobian(problem: EstimationProblem, at: Any, point: Any) -> Any:
    """The whitened data Jacobian a fit recorded at ``point``, or ``None``.

    A fit of the same study (``info["uq"]["study_fingerprint"]``) whose
    parameters are exactly ``point`` and whose optimizer recorded the data
    Jacobian there (``info["final_data_jacobian"]``) carries the Jacobian
    ``problem.data_residual_jacobian(point)`` returns.
    """
    if not isinstance(at, OptResult):
        return None
    jacobian = at.info.get("final_data_jacobian")
    uq = at.info.get("uq")
    if jacobian is None or not isinstance(uq, Mapping):
        return None
    if uq.get("study_fingerprint") != problem.study.fingerprint:
        return None
    params = np.asarray(at.params)
    if params.shape != np.shape(point) or not np.array_equal(params, np.asarray(point)):
        return None
    if np.shape(jacobian)[1:] != params.shape:
        return None
    return jnp.asarray(jacobian)


def _information_facade(
    problem: EstimationProblem,
    *,
    at: Any = None,
    kind: str = "expected",
) -> InformationResult:
    """Information of an ``InferenceProblem`` (the estimation-problem route of ``information``).

    Expected/GGN information is the Gram matrix of the exact whitened
    residual Jacobian; a lowered study sums it over its lanes
    (``expected_information``) and forms no residual Jacobian. At a fit that
    recorded the data Jacobian at its parameters, that Jacobian is used and no
    forward solve runs for it. Observed information is the real
    ``jax.jacfwd(jax.jacfwd(nll))`` Hessian retained by
    [InferenceProblem][diff_soup.inference.InferenceProblem], on a lowered
    study the sum of its lanes' exact Hessians.
    """
    if kind not in ("expected", "ggn", "observed", "all"):
        raise ValueError(
            f"information kind must be 'expected', 'ggn', 'observed', or 'all'; got {kind!r}."
        )
    point = _point(problem, at)
    # A lowered study sums J^T J over its lanes; it forms no residual Jacobian.
    lanes = getattr(problem, "expected_information", None)
    expected = lanes(point) if callable(lanes) else None
    if expected is None:
        jacobian = _recorded_data_jacobian(problem, at, point)
        if jacobian is None:
            _residual, jacobian = problem.data_residual_jacobian(point)
        expected = jacobian.T @ jacobian
    else:
        jacobian = None
    observed = (
        _symmetrized(problem.observed_information(point)) if kind in ("observed", "all") else None
    )
    prior = problem.prior_precision(point) if isinstance(problem.objective, MAP) else None
    return assemble_information(
        whitened_jacobian=jacobian,
        expected=expected,
        ggn=expected if kind in ("ggn", "all") else None,
        observed=observed,
        prior_precision=prior,
        coordinate="x",
        provenance={
            "study_fingerprint": problem.study.fingerprint,
            "objective": type(problem.objective).__name__,
            "at": tuple(float(value) for value in np.asarray(point)),
            "theta_names": _coordinate_names(problem),
            "curvature_kind": kind,
        },
    )


def information(problem: Any, *, at: Any, kind: str = "observed") -> InformationResult:
    """Assemble the information matrix, the curvature of the negative log likelihood, at a point.

    For a `MAP` problem the result also holds the prior's curvature and their sum, the posterior
    precision.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] whose measurements
            all declare a likelihood.
        at: The point, an [`OptResult`][diff_soup.inference.OptResult], a parameter vector, or a
            mapping from every fitted handle to its value.
        kind: `"observed"` for the curvature at the data seen, or `"expected"` for the curvature
            averaged over data drawn from the model.

    Returns:
        An `InformationResult` holding the matrix, whose `rank()` counts the directions the data
            determine.
    """
    if not isinstance(problem, EstimationProblem):
        raise TypeError("information requires an InferenceProblem.")
    require_likelihoods(problem, "information")
    if kind == "ggn":
        raise ValueError('information kind="ggn" was removed; use kind="expected".')
    if kind not in ("observed", "expected"):
        raise ValueError('information kind must be "observed" or "expected".')
    return _information_facade(problem, at=at, kind=kind)
