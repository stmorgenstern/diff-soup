"""profile.py — profile likelihood over fitted problems.

The profile of a scalar target ``g(x)`` is the fitted objective re-minimized
over every other parameter while ``g`` is held at a grid of values: the
curvature of the resulting one-dimensional curve is what the data actually
say about ``g`` once every compensation the model allows has been used.
Where the Laplace ellipsoid linearizes, the profile walks the true objective,
so it exposes curved valleys, one-sided intervals, and practically
unidentified (open) directions that a covariance matrix cannot represent.

Three target kinds are supported by [profile][diff_soup.inference.statistics.profile.profile]:

* a parameter name (one flat-theta scalar),
* a direction vector in flat-theta space (the profiled scalar is ``vᵀx``),
* [At][diff_soup.inference.statistics.profile.At] — the value of a learnable function block at a point, e.g.
  ``At("j0", c=0.4)`` (linear in the block's coefficients).

Nuisance re-optimization runs through the standard ``fit`` engine on a
reduced-coordinate view of the problem — the pinned scalar is eliminated
exactly (``x = x_p + N u``), never penalized — with each grid point
warm-started from its neighbour. Interval endpoints come from the standard
likelihood-ratio threshold; the asymptotic assumptions behind that reading
are recorded on the result, and directions the walk could not close are
reported as open rather than extrapolated.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

from diff_soup.inference.approx.legendre import Legendre
from diff_soup.inference.fitting._forward_seam import _path_key_tokens
from diff_soup.inference.fitting.bounds import resolve_decision_bounds
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.hierarchical.level_solvers import TrustRegion
from diff_soup.inference.solvers.levenberg_marquardt import LevenbergMarquardt
from diff_soup.inference.statistics.estimation import MAP, EstimationProblem
from diff_soup.inference.statistics.experiment import _CompiledExperiment
from diff_soup.inference.statistics.identifiability import (
    _require_problem_schema,
    _schema_coordinates,
)
from diff_soup.inference.statistics.measurement import require_likelihoods

__all__ = ["At", "ProfileRefit", "ProfileResult", "profile"]


class At(eqx.Module):
    """A function-value profile target: a learnable block evaluated at a point.

    ``At("j0", c=0.4)`` profiles the value of the function whose coefficients
    live in parameter block ``"j0"``, evaluated at the physical input
    ``c = 0.4``. The value must be *linear* in the block's coefficients (the
    raw basis expansion, before any output transform such as ``exp``); the
    profiled scalar is then ``row @ coeffs`` with ``row`` the basis functions
    evaluated at ``c``.

    The evaluation row is resolved in this order:

    1. ``row`` — an explicit coefficient row, used verbatim;
    2. ``basis`` — a [Legendre][diff_soup.inference.approx.legendre.Legendre]
       whose interval scaling maps ``c`` onto the Legendre argument;
    3. discovery — the inference problem's model pytree is searched for a single
       [Legendre][diff_soup.inference.approx.legendre.Legendre] leaf reached through a path containing ``name``.

    Attributes:
        name: The parameter block whose coefficients are targeted.
        c: The physical evaluation point.
        basis: Optional explicit basis object carrying the interval scaling.
        row: Optional explicit evaluation row over the block's coefficients.
    """

    name: str = eqx.field(static=True)
    c: float = eqx.field(static=True)
    basis: Any = None
    row: Optional[Any] = None

    def __init__(self, name: str, *, c: Any, basis: Any = None, row: Any = None):
        """Validate the target declaration.

        Raises:
            TypeError: If ``name`` is not a non-empty string.
            ValueError: If ``c`` is not a finite scalar or ``row`` is not a
                finite 1-D vector.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"At 'name' must be a non-empty block name; got {name!r}.")
        c = float(c)
        if not np.isfinite(c):
            raise ValueError(f"At 'c' must be a finite evaluation point; got {c!r}.")
        if row is not None:
            row = np.asarray(row, dtype=np.float64)
            if row.ndim != 1 or row.size == 0 or not np.all(np.isfinite(row)):
                raise ValueError(
                    "At 'row' must be a finite 1-D vector with one entry per "
                    f"coefficient of the block; got {row!r}."
                )
        self.name = name
        self.c = c
        self.basis = basis
        self.row = row

    def label(self) -> str:
        """A short human-readable label for reports."""
        return f"{self.name}(c={self.c:g})"


@dataclasses.dataclass(frozen=True)
class ProfileRefit:
    """One profile grid point's re-optimization record.

    Attributes:
        t: The pinned value of the profiled scalar.
        nll: The re-minimized objective value (best basin under multistart).
        n_steps: Iterations of the warm-started refit (the first basin).
        converged: Whether the best basin's solve converged.
        basin_nlls: Objective values of every start (warm start first). More
            than one distinct value means the nuisance landscape has several
            basins at this grid point — the profile takes the best, never an
            average, and the spread is the disagreement to report.
    """

    t: float
    nll: float
    n_steps: int
    converged: bool
    basin_nlls: tuple


@dataclasses.dataclass(frozen=True)
class ProfileResult:
    """The profile curve of one scalar target with its interval machinery.

    Attributes:
        target: Human-readable label of the profiled scalar.
        grid: Profiled values, ascending; includes the fitted value.
        nll: Re-minimized objective at each grid value (aligned with ``grid``).
        reference_nll: The objective at the overall best point found; the
            likelihood-ratio statistic is ``2 * (nll - reference_nll)``.
        levels: The confidence levels the walk was asked to cover.
        refits: One
            [ProfileRefit][diff_soup.inference.statistics.profile.ProfileRefit]
            per non-center grid point, in walk order (lower wing
            outward, then upper wing outward).
        open_directions: The sides (``"lower"``/``"upper"``) whose walk ended
            before the curve reached the largest requested threshold — the data
            do not close the interval on that side, and
            [interval][diff_soup.inference.statistics.profile.ProfileResult.interval]
            reports it infinite.
            [wing_end][diff_soup.inference.statistics.profile.ProfileResult]
            says why each side ended.
        wing_end: Why each direction's walk ended, keyed ``"lower"`` and
            ``"upper"``, with one of:

            * ``"crossed"`` — the objective rise passed the largest level's
              threshold, so that side is closed;
            * ``"bounds"`` — the walk reached the limit declared for the
              target;
            * ``"max_steps"`` — the per-direction refit budget ran out;
            * ``"nonconvergence"`` — a refit did not converge within the
              optimizer's per-refit step budget, and the shorter-step retries
              did not recover it;
            * ``"solver_failure"`` — a refit returned a non-finite objective.

            The last two mean the side stopped on an uncertified refit rather
            than on evidence: raise the nuisance optimizer's ``max_steps``, or
            refit from a better start, before reading that side as
            unidentified. The same mapping is repeated in
            ``provenance["wing_end"]``, and ``provenance["retries"]`` counts
            the retries each side spent.
        provenance: How this curve was produced: the target, the
            likelihood-ratio assumptions behind
            [threshold][diff_soup.inference.statistics.profile.ProfileResult.threshold],
            each wing's end reason and retry count, the multistart policy, and
            whether the objective includes regularization rows.
    """

    target: str
    grid: np.ndarray
    nll: np.ndarray
    reference_nll: float
    levels: tuple
    refits: tuple
    open_directions: tuple
    wing_end: dict
    provenance: dict

    @property
    def delta_nll(self) -> np.ndarray:
        """The objective rise over the best point, aligned with ``grid``."""
        return self.nll - self.reference_nll

    def threshold(self, level: float) -> float:
        """The likelihood-ratio ``delta_nll`` threshold for one scalar.

        Uses the asymptotic chi-squared calibration with one degree of
        freedom recorded in ``provenance["assumptions"]``:
        ``chi2.ppf(level, 1) / 2``.
        """
        _check_level(level)
        return float(chi2.ppf(level, 1) / 2.0)

    def interval(self, level: float) -> tuple:
        """The profile-likelihood interval endpoints at ``level``.

        Endpoints interpolate the signed root of the likelihood-ratio
        statistic, ``sign(t - t_best) * sqrt(2 delta_nll)``, which is linear
        in ``t`` for a quadratic objective — so quadratic problems reproduce
        the Wald interval exactly. A side whose curve never reaches the
        threshold is reported as infinite (see
        [open_directions][diff_soup.inference.statistics.profile.ProfileResult.open_directions]).
        """
        _check_level(level)
        z = float(np.sqrt(chi2.ppf(level, 1)))
        root = np.sqrt(np.maximum(2.0 * self.delta_nll, 0.0))
        i0 = int(np.argmin(self.nll))

        def _cross(indices) -> float:
            prev = None
            for j in indices:
                if root[j] >= z and prev is not None:
                    ti, ri = self.grid[prev], root[prev]
                    tj, rj = self.grid[j], root[j]
                    return float(ti + (tj - ti) * (z - ri) / (rj - ri))
                if root[j] >= z:
                    return float(self.grid[j])
                prev = j
            return np.nan

        lo = _cross(range(i0, -1, -1))
        hi = _cross(range(i0, len(self.grid)))
        return (
            float(lo) if np.isfinite(lo) else -np.inf,
            float(hi) if np.isfinite(hi) else np.inf,
        )


def _check_level(level: float) -> None:
    """Reject confidence levels outside the open unit interval."""
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"confidence level must satisfy 0 < level < 1; got {level!r}.")


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _theta_names(problem: Any) -> tuple:
    names = tuple(getattr(problem, "theta_names", ()) or ())
    if not names:
        raise ValueError(
            "profile needs the problem to expose 'theta_names' so targets "
            "can be resolved to flat-vector columns."
        )
    return names


def _block_columns(problem: Any, name: str) -> list:
    """Flat-vector columns of a named parameter block."""
    group_names = tuple(getattr(problem, "param_group_names", ()) or ())
    if name in group_names:
        cols_by_group = tuple(getattr(problem, "param_group_cols", ()) or ())
        return list(np.asarray(cols_by_group[group_names.index(name)], dtype=int))
    names = _theta_names(problem)
    cols = [
        i
        for i, n in enumerate(names)
        if n == name or (n.startswith(f"{name}[") and n.endswith("]"))
    ]
    if not cols:
        raise ValueError(
            f"target block {name!r} matches no parameter group and no flat-theta "
            f"scalars; available scalars are {list(names)} and parameter groups "
            f"are {list(group_names)}."
        )
    return cols


def _discover_basis(problem: Any, name: str) -> Optional[Legendre]:
    """Find the single Legendre leaf reached through a path containing ``name``."""
    if problem is None:
        return None
    try:
        leaves = jax.tree_util.tree_flatten_with_path(
            problem, is_leaf=lambda n: isinstance(n, Legendre)
        )[0]
    except Exception:  # noqa: BLE001 — non-pytree problems simply have no leaves
        return None
    hits = [
        leaf
        for path, leaf in leaves
        if isinstance(leaf, Legendre) and name in _path_key_tokens(path)
    ]
    return hits[0] if len(hits) == 1 else None


def _at_row(target: At, problem: Any, n_cols: int) -> np.ndarray:
    """The evaluation row of an [At][diff_soup.inference.statistics.profile.At] target.

    The row spans the target's block columns.
    """
    if target.row is not None:
        row = np.asarray(target.row, dtype=np.float64)
    else:
        basis = target.basis
        if basis is None:
            basis = _discover_basis(getattr(problem, "problem", None), target.name)
        if basis is None:
            raise ValueError(
                f"At({target.name!r}, c={target.c:g}) could not resolve its basis: "
                "the inference problem's model has no unique Legendre leaf under that "
                "name. Pass basis=<the block's Legendre> or row=<explicit "
                "evaluation row>."
            )
        lo, hi = getattr(basis, "interval", (0.0, 1.0))
        x = 2.0 * (target.c - lo) / (hi - lo) - 1.0
        n_terms = int(np.asarray(basis.coeffs).shape[0])
        row = np.polynomial.legendre.legvander(np.asarray([x]), n_terms - 1)[0]
    if row.shape != (n_cols,):
        raise ValueError(
            f"At({target.name!r}) resolved an evaluation row of length "
            f"{row.shape[0]} but the block owns {n_cols} coefficient column(s); "
            "the row needs one entry per coefficient."
        )
    return row


def _resolve_target(problem: Any, target: Any) -> tuple:
    """Resolve a profile target to ``(a, label, coord_index)``.

    ``a`` is the full flat-space vector of the profiled scalar ``t = a @ x``;
    ``coord_index`` is the column index when the target is one coordinate
    (which unlocks exact box-bound reduction), else ``None``.
    """
    names = _theta_names(problem)
    n = len(names)
    if isinstance(target, str):
        if target not in names:
            raise ValueError(
                f"profile target {target!r} is not a flat-theta scalar of this "
                f"problem; available scalars are {list(names)}."
            )
        i = names.index(target)
        a = np.zeros(n)
        a[i] = 1.0
        return a, target, i
    if isinstance(target, At):
        cols = _block_columns(problem, target.name)
        row = _at_row(target, problem, len(cols))
        a = np.zeros(n)
        a[np.asarray(cols, dtype=int)] = row
        if np.count_nonzero(a) == 1:
            return a, target.label(), int(np.flatnonzero(a)[0])
        return a, target.label(), None
    a = np.asarray(target, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(
            f"a direction target must be a 1-D vector with one entry per "
            f"flat-theta scalar ({n}); got shape {np.shape(target)}."
        )
    if not np.all(np.isfinite(a)) or not np.any(a != 0.0):
        raise ValueError("a direction target must be finite and nonzero.")
    # A public handle-aware facade can resolve a scalar handle directly to
    # its flat column even when the problem's ``theta_names`` are merely
    # lowering artefacts (for example ``"values"``).  Preserve that exact
    # coordinate information for an explicitly supplied positive one-hot
    # direction.  This is not just a nicer label: it lets the reduced nuisance
    # problem delete the corresponding box-bound row exactly.  General scaled
    # or signed directions remain linear targets because their target-space
    # bounds are not the decision-coordinate bounds.
    nonzero = np.flatnonzero(a)
    coord_index = int(nonzero[0]) if nonzero.size == 1 and a[int(nonzero[0])] == 1.0 else None
    return a, "direction", coord_index


# ---------------------------------------------------------------------------
# The reduced-coordinate view the fit engine solves
# ---------------------------------------------------------------------------


class _PinnedView:
    """A problem view with one linear functional of ``x`` pinned.

    The decision coordinate is the reduced nuisance vector ``u``; the full
    vector is recovered by ``lift(u)``, which holds ``a @ x = t`` exactly.
    The surface is closed and explicit (no attribute delegation): every
    callable an engine may read is re-expressed in the reduced coordinate, so
    a forgotten passthrough can never hand the engine a full-coordinate
    closure by accident.
    """

    is_decision_variable = False
    is_sharded = False
    theta_only_steps = 0
    batched_r_fn = None
    batched_rj_fn = None
    evaluate_lsq_fn = None
    evaluate_cost_fn = None
    bounds = None
    prob_static = None
    unflatten_fn = None

    def __init__(
        self,
        inner: Any,
        lift: Callable,
        basis_matrix: np.ndarray,
        u0: np.ndarray,
        reduced_bounds: Any,
    ):
        self._inner = inner
        self._lift = lift
        self.x0 = np.asarray(u0, dtype=float)
        self.theta_names = tuple(f"nuisance[{i}]" for i in range(len(self.x0)))
        self.param_group_names = ()
        self.param_group_cols = ()
        self._declared_bounds = reduced_bounds
        self.n_segments = int(getattr(inner, "n_segments", 1))

        N = jnp.asarray(basis_matrix)
        inner_r = getattr(inner, "stacked_r_fn", None)
        inner_rj = getattr(inner, "stacked_rj_fn", None)
        if inner_r is not None and inner_rj is not None:
            # Reuse the inner problem's cached compiled kernels; the chain
            # rule for the affine map is exact: J_u = J_x @ N.
            self.stacked_r_fn = lambda u: inner_r(lift(u))
            self.stacked_rj_fn = lambda u: _chain(inner_rj, lift, N, u)
            self.segment_fns = (self.stacked_r_fn,)
        else:
            self.stacked_r_fn = None
            self.stacked_rj_fn = None
            self.segment_fns = tuple(
                (lambda fn: lambda u: fn(lift(u)))(fn) for fn in inner.segment_fns
            )

        reg = getattr(inner, "regularization", None)
        if reg is None or isinstance(reg, NoReg):
            self.regularization = NoReg()
        else:
            self.regularization = _ReducedRegularizer(reg, lift)

        inner_count = getattr(inner, "count_failed", None)
        if inner_count is not None:
            self.count_failed = lambda u: inner_count(np.asarray(lift(jnp.asarray(u))))

        inner_unflatten = getattr(inner, "unflatten_fn", None)
        if inner_unflatten is not None:
            self.unflatten_fn = lambda u: inner_unflatten(lift(jnp.asarray(u)))
            self.prob_static = getattr(inner, "prob_static", None)


def _chain(inner_rj: Callable, lift: Callable, basis: Any, u: Any) -> tuple:
    r, jac = inner_rj(lift(u))
    return r, jac @ basis


class _ReducedRegularizer:
    """The inner regularizer re-expressed in the reduced coordinate."""

    @property
    def baseline_reached_tend(self) -> bool:
        """Whether the wrapped problem's forward solve at ``x0`` reaches ``t_end``."""
        return bool(getattr(self._inner, "baseline_reached_tend", True))

    def __init__(self, inner: Any, lift: Callable):
        self._inner = inner
        self._lift = lift

    def residual(self, u: Any):
        """Regularizer rows at the lifted full vector."""
        return self._inner.residual(self._lift(u))

    def penalty(self, u: Any):
        """Scalar penalty at the lifted full vector."""
        return self._inner.penalty(self._lift(u))


def _make_view(
    inner: Any,
    a: np.ndarray,
    coord_index: Optional[int],
    t: float,
    x_warm: np.ndarray,
    reduced_bounds: Any,
) -> _PinnedView:
    """Build the reduced view pinning ``a @ x = t``, warm-started at ``x_warm``."""
    n = a.shape[0]
    if coord_index is not None and a[coord_index] == 1.0 and np.count_nonzero(a) == 1:
        i = coord_index
        pin = jnp.asarray(float(t))

        def lift(u):
            return jnp.insert(jnp.asarray(u), i, pin)

        basis_matrix = np.delete(np.eye(n), i, axis=1)
        u0 = np.delete(np.asarray(x_warm, dtype=float), i)
    else:
        a_hat = a / np.linalg.norm(a)
        # Orthonormal basis of the hyperplane a @ x = t.
        basis_matrix = np.linalg.svd(a_hat[None, :])[2][1:].T
        x_p = jnp.asarray(a_hat * (t / np.linalg.norm(a)))
        N = jnp.asarray(basis_matrix)

        def lift(u):
            return x_p + N @ jnp.asarray(u)

        u0 = basis_matrix.T @ np.asarray(x_warm, dtype=float)
    return _PinnedView(inner, lift, basis_matrix, u0, reduced_bounds)


# ---------------------------------------------------------------------------
# Curvature-based step sizing
# ---------------------------------------------------------------------------


def _profile_sd(problem: Any, x_hat: np.ndarray, a: np.ndarray) -> float:
    """The Gauss-Newton standard deviation of ``a @ x`` at the fit.

    Returns ``inf`` when ``a`` has weight on directions the data curvature
    does not determine (the profile of such a target is expected to be open).
    """
    rj = getattr(problem, "stacked_rj_fn", None)
    x = jnp.asarray(x_hat)
    if rj is not None:
        _, jac = rj(x)
    else:
        seg_fns = problem.segment_fns

        def _r(p):
            return jnp.concatenate([jnp.reshape(fn(p), (-1,)) for fn in seg_fns])

        jac = jax.jacfwd(_r)(x)
    hess = np.asarray(jac.T @ jac)
    hess = 0.5 * (hess + hess.T)
    w, v = np.linalg.eigh(hess)
    w_max = float(w[-1]) if w.size else 0.0
    if w_max <= 0.0:
        return np.inf
    tol = w_max * 1e-12
    comps = v.T @ a
    null_mass = float(np.sum(comps[w <= tol] ** 2))
    if null_mass > 1e-12 * float(a @ a):
        return np.inf
    keep = w > tol
    return float(np.sqrt(np.sum(comps[keep] ** 2 / w[keep])))


# ---------------------------------------------------------------------------
# Uncertified refits
# ---------------------------------------------------------------------------

#: Retries granted per walk direction when a refit does not certify. Each one
#: halves the step and doubles that refit's optimizer budget.
_MAX_WING_RETRIES = 2


def _with_budget(optimizer: Any, factor: int) -> Any:
    """``optimizer`` with its per-refit iteration budget multiplied by ``factor``.

    Optimizers that do not expose a ``max_steps`` field are returned unchanged,
    so a retry of theirs differs from the first attempt only in its step.
    """
    if factor == 1 or optimizer is None or not dataclasses.is_dataclass(optimizer):
        return optimizer
    if not any(field.name == "max_steps" for field in dataclasses.fields(optimizer)):
        return optimizer
    return dataclasses.replace(optimizer, max_steps=int(factor) * int(optimizer.max_steps))


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _profile_engine(
    problem_or_experiment: Any,
    target: Any,
    *,
    fit: Any = None,
    optimizer: Any = None,
    parameters: Any = None,
    levels: tuple = (0.95,),
    bounds: Optional[tuple] = None,
    max_steps: int = 24,
    multistart: int = 0,
    multistart_scale: float = 1.0,
) -> ProfileResult:
    """Profile one scalar target of a fitted problem or experiment.

    Walks the target away from its fitted value in both directions with
    warm-started, curvature-adapted steps, re-minimizing every other
    parameter through the standard ``fit`` engine at each grid value (the
    pinned scalar is eliminated exactly in a reduced coordinate — no penalty
    terms). The walk stops on each side when the objective rise crosses the
    largest requested level's likelihood-ratio threshold, when it reaches
    ``bounds``, or after ``max_steps`` refits; sides that never crossed are
    reported open.

    The first step out is ``sd / 3``, with ``sd`` the Gauss-Newton (Laplace)
    standard-deviation estimate of the target; where that curvature is
    unavailable the step falls back to a quarter of the distance to the walk
    limit on that side, and failing that to ``0.25 * max(|t_fit|, 1)``. A
    finite limit also caps the first step at a quarter of the distance to it.
    After each certified point the step is rescaled by
    ``sqrt(target_rise / observed_rise)`` clipped to ``[0.5, 2]``, where
    ``target_rise`` is a quarter of the threshold, and is doubled outright
    when the observed rise is negligible; the walk stops once the rise passes
    ``1.1`` times the threshold, so the crossing is bracketed rather than
    interpolated off the end. Because the first step is deliberately short,
    an exactly Gaussian curve needs four refits to close a side and a curved
    profile needs more: ``max_steps=3`` cannot close a side even in the
    Gaussian case.

    ``optimizer.max_steps`` is the budget of a single refit, not of the walk.
    A refit that does not converge within it — or that returns a non-finite
    objective — is not evidence: it never joins the curve, and it is retried
    from the last certified point with half the step and twice that budget,
    at most twice per direction. When those retries are spent the direction
    ends as ``"nonconvergence"`` or ``"solver_failure"`` in
    [wing_end][diff_soup.inference.statistics.profile.ProfileResult],
    and every attempt stays in ``refits`` with ``converged=False`` for
    diagnosis.

    Args:
        problem_or_experiment: An inference problem, or an
            [Experiment][diff_soup.inference.statistics.experiment.Channel] with
            response data attached (lowered through ``as_inference_problem``).
        target: A flat-theta scalar name, a direction vector (the profiled
            scalar is ``vᵀx``), or an
            [At][diff_soup.inference.statistics.profile.At] function-value target.
        fit: An existing fit result for this problem (its ``params`` and
            ``cost`` seed the walk); ``None`` runs the base fit here.
        optimizer: The engine for every (re)fit; ``None`` uses the default
            Levenberg-Marquardt configuration.
        parameters: Parameter groups for the experiment lowering; only valid
            together with an experiment input.
        levels: Confidence levels the walk must be able to answer; the
            largest sets the stopping threshold.
        bounds: Optional ``(lo, hi)`` limits for the profiled scalar itself.
        max_steps: Refit budget per direction: how many certified grid points
            the walk may add on each side. Retries of uncertified refits are
            extra, at most two per direction.
        multistart: Extra jittered nuisance starts per grid point. Their
            objective values are recorded per point (``basin_nlls``); the
            profile takes the best basin and reports disagreement, never an
            average.
        multistart_scale: Relative scale of the start jitter.

    Returns:
        A [ProfileResult][diff_soup.inference.statistics.profile.ProfileResult].

    Raises:
        ValueError: If the target cannot be resolved, the levels or bounds
            are malformed, the problem is a decision-variable multishoot
            (window states are not profiled), or box bounds are declared with
            a non-coordinate target (the reduced refits would not respect
            them).
    """
    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    from diff_soup.inference.fitting._optsolve import optsolve as run_fit

    if isinstance(problem_or_experiment, _CompiledExperiment):
        problem = problem_or_experiment.as_inference_problem(parameters=parameters)
        lowered = True
    else:
        if parameters is not None:
            raise ValueError(
                "profile 'parameters' only applies when an Experiment is given; "
                "a problem already owns its parameter groups."
            )
        problem = problem_or_experiment
        lowered = False

    if getattr(problem, "is_decision_variable", False):
        raise ValueError(
            "profile does not support decision-variable multishoot problems: "
            "the decision vector mixes parameters with window states. Profile a "
            "single-shoot or observed-window problem of the same problem."
        )

    levels = tuple(float(lv) for lv in levels)
    if not levels:
        raise ValueError("profile needs at least one confidence level.")
    for lv in levels:
        _check_level(lv)
    if multistart < 0:
        raise ValueError(f"multistart must be a non-negative count; got {multistart}.")

    a, label, coord_index = _resolve_target(problem, target)
    lane_optimizer = isinstance(optimizer, TrustRegion)
    if lane_optimizer and coord_index is None:
        raise ValueError(
            "TrustRegion profile refits require one scalar coordinate; use a scalar handle "
            "or LevenbergMarquardt for a general linear target"
        )
    if lane_optimizer:
        pinned_optimizer = dataclasses.replace(
            optimizer, fixed_indices=tuple(sorted(set(optimizer.fixed_indices) | {coord_index}))
        )

    # Box bounds on the decision vector reduce exactly only when the pinned
    # scalar is one coordinate (the nuisance box is the box minus that row).
    full_bounds = resolve_decision_bounds(problem)
    reduced_bounds = None
    coord_box = None
    if full_bounds is not None:
        if coord_index is None:
            raise ValueError(
                "profile cannot honor the problem's box bounds for a "
                "direction or multi-coefficient target: the reduced nuisance "
                "coordinate has no box representation of them. Profile a named "
                "coordinate, or rebuild the problem without bounds."
            )
        lb, ub = (np.asarray(full_bounds[0]), np.asarray(full_bounds[1]))
        reduced_bounds = (
            jnp.asarray(np.delete(lb, coord_index)),
            jnp.asarray(np.delete(ub, coord_index)),
        )
        coord_box = (float(lb[coord_index]), float(ub[coord_index]))

    # --- base fit ---------------------------------------------------------
    if fit is None:
        base = run_fit(problem, optimizer, diagnostics=False)
        base_source = "computed"
    else:
        base = fit
        base_source = "provided"
    x_hat = np.asarray(getattr(base, "params", None), dtype=float)
    if x_hat.ndim != 1 or x_hat.shape[0] != a.shape[0]:
        raise ValueError(
            "profile 'fit' must carry a flat 'params' vector matching the "
            f"problem's {a.shape[0]} flat-theta scalar(s)."
        )
    nll_hat = float(base.cost)
    t_hat = float(a @ x_hat)

    # --- walk limits and initial step --------------------------------------
    lo_b, hi_b = -np.inf, np.inf
    if bounds is not None:
        lo_b, hi_b = (float(bounds[0]), float(bounds[1]))
        if not lo_b < hi_b:
            raise ValueError(f"profile bounds need lo < hi; got {bounds!r}.")
    if coord_box is not None:
        lo_b, hi_b = max(lo_b, coord_box[0]), min(hi_b, coord_box[1])
    if not lo_b <= t_hat <= hi_b:
        raise ValueError(
            f"the fitted target value {t_hat:g} lies outside the profile bounds "
            f"({lo_b:g}, {hi_b:g}); widen the bounds to include the fit."
        )

    sd = _profile_sd(problem, x_hat, a)
    thr_max = float(chi2.ppf(max(levels), 1) / 2.0)
    target_inc = thr_max / 4.0
    stop_delta = 1.10 * thr_max

    rng = np.random.default_rng(0)
    reg = getattr(problem, "regularization", None)
    has_reg = reg is not None and not isinstance(reg, NoReg)

    def refit_at(t: float, x_warm: np.ndarray, budget_factor: int = 1) -> tuple:
        view = _make_view(problem, a, coord_index, t, x_warm, reduced_bounds)
        u_warm = np.asarray(view.x0)
        starts = [u_warm]
        for _ in range(multistart):
            jitter = rng.normal(size=u_warm.shape)
            starts.append(u_warm + multistart_scale * (1.0 + np.abs(u_warm)) * jitter)
        engine = _with_budget(pinned_optimizer if lane_optimizer else optimizer, budget_factor)
        basins = []
        for u_start in starts:
            view.x0 = np.asarray(u_start, dtype=float)
            if lane_optimizer:
                start = view._lift(jnp.asarray(u_start))
                res = run_fit(problem.with_start(start), engine, diagnostics=False)
            else:
                res = run_fit(view, engine, diagnostics=False)
            basins.append(res)
        costs = [float(r.cost) for r in basins]
        best = basins[int(np.argmin(costs))]
        x_new = (
            np.asarray(best.params)
            if lane_optimizer
            else np.asarray(view._lift(jnp.asarray(best.params)))
        )
        record = ProfileRefit(
            t=float(t),
            nll=float(best.cost),
            n_steps=int(basins[0].n_steps),
            converged=bool(best.converged),
            basin_nlls=tuple(costs),
        )
        return record, x_new

    points = [(t_hat, nll_hat)]
    refits: list = []
    wing_end: dict = {}
    retries: dict = {"lower": 0, "upper": 0}
    for side, sign, limit in (("lower", -1.0, lo_b), ("upper", +1.0, hi_b)):
        span = abs(limit - t_hat)
        if np.isfinite(sd) and sd > 0.0:
            h = sd / 3.0
        elif np.isfinite(span) and span > 0.0:
            h = span / 4.0
        else:
            h = 0.25 * max(abs(t_hat), 1.0)
        if np.isfinite(span):
            if span == 0.0:
                wing_end[side] = "bounds"
                continue
            h = min(h, span / 4.0)

        t_prev, nll_prev, x_prev = t_hat, nll_hat, x_hat
        end = "max_steps"
        taken = 0
        attempt = 0
        while taken < max_steps:
            t_next = t_prev + sign * h
            hit_bound = False
            if sign < 0 and t_next <= limit:
                t_next, hit_bound = limit, True
            if sign > 0 and t_next >= limit:
                t_next, hit_bound = limit, True
            record, x_next = refit_at(t_next, x_prev, 2**attempt)
            refits.append(record)
            if not (record.converged and np.isfinite(record.nll)):
                # A terminal nuisance objective is only an upper bound on the
                # true profile. Using it as a threshold crossing can close an
                # interval spuriously. Retain the refit record for diagnosis,
                # but do not add the uncertified point to the profile curve —
                # and retry the step from the last certified point, halved and
                # with twice the per-refit budget, before ending the direction
                # on one uncertified refit.
                reason = "nonconvergence" if np.isfinite(record.nll) else "solver_failure"
                if retries[side] >= _MAX_WING_RETRIES:
                    end = reason
                    break
                retries[side] += 1
                attempt += 1
                h *= 0.5
                continue
            taken += 1
            attempt = 0
            points.append((t_next, record.nll))
            if record.nll - nll_hat > stop_delta:
                end = "crossed"
                break
            if hit_bound:
                end = "bounds"
                break
            rise = record.nll - nll_prev
            if rise <= max(1e-3 * target_inc, 1e-12):
                h *= 2.0
            else:
                h *= float(np.clip(np.sqrt(target_inc / rise), 0.5, 2.0))
            t_prev, nll_prev, x_prev = t_next, record.nll, x_next
        wing_end[side] = end

    points.sort(key=lambda p: p[0])
    grid = np.asarray([p[0] for p in points])
    nll = np.asarray([p[1] for p in points])
    reference = float(min(nll_hat, float(np.min(nll))))
    undercut = nll_hat - reference

    open_dirs = tuple(
        side
        for side in ("lower", "upper")
        if wing_end[side] != "crossed" and (nll[0 if side == "lower" else -1] - reference) < thr_max
    )

    provenance = {
        "target": label,
        "direction": a,
        "coordinate": "flat_theta",
        "assumptions": {
            "statistic": "likelihood ratio (Wilks asymptotics)",
            "df": 1,
            "calibration": "chi-squared, fixed noise scale",
        },
        "wing_end": dict(wing_end),
        "retries": dict(retries),
        "multistart": (
            {"count": int(multistart), "scale": float(multistart_scale), "seed": 0}
            if multistart
            else None
        ),
        "objective_includes_regularizer": bool(has_reg),
        "base_fit": base_source,
        "lowered_from_experiment": lowered,
        "profile_sd_estimate": float(sd) if np.isfinite(sd) else np.inf,
        "undercut": float(undercut),
    }
    return ProfileResult(
        target=label,
        grid=grid,
        nll=nll,
        reference_nll=reference,
        levels=levels,
        refits=tuple(refits),
        open_directions=open_dirs,
        wing_end=dict(wing_end),
        provenance=provenance,
    )


def _validate_fit(problem: EstimationProblem, fit: OptResult) -> None:
    """Require a converged fit of this problem's parameter layout and objective."""
    if fit.parameter_schema is None:
        raise ValueError(
            "profile requires an OptResult with handle-based parameter schema "
            "provenance; fit the supplied InferenceProblem through diff_soup.inference.optsolve(...)."
        )

    _require_problem_schema(problem, fit.parameter_schema, where="profile")

    expected_objective = type(problem.objective).__name__
    if type(fit.objective).__name__ != expected_objective:
        raise ValueError(
            f"profile: fit objective is {type(fit.objective).__name__!r}, but the problem "
            f"objective is {expected_objective!r}."
        )
    if int(fit.n_failed_segments) != 0:
        raise ValueError(
            "profile refuses a fit with failed forward-solve segments; failed "
            f"indices are {tuple(fit.failed_segment_indices)}."
        )
    if not fit.converged:
        raise ValueError(
            f"profile requires a converged fit; the supplied fit stopped "
            f"with status {getattr(fit.status, 'value', fit.status)!r}."
        )

    params = np.asarray(fit.params, dtype=float)
    expected_shape = (int(problem.n_params),)
    if params.shape != expected_shape:
        raise ValueError(
            f"profile: fitted parameter vector has shape {params.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(params)) or not np.isfinite(float(fit.cost)):
        raise ValueError("profile: fitted parameters and cost must be finite.")


def _resolve_occurrence(problem: EstimationProblem, handle: Any, owner: Any) -> tuple:
    """Resolve one scalar handle occurrence to ``(entry, occurrence)``."""
    if not hasattr(handle, "identity") or not hasattr(handle, "name"):
        raise TypeError(
            "profile handle must be the scalar handle returned by spec.param(...); "
            "strings, flat names, and direction arrays are not parameter identities."
        )
    kind = getattr(handle, "kind", None)
    if kind != "scalar":
        raise TypeError(
            f"profile accepts one scalar spec.param handle; {handle.name!r} is "
            f"{kind!r}. Array components and function values need an explicit "
            "component/value target API and are intentionally not guessed."
        )
    try:
        entry = problem.parameter_schema.entry(handle)
    except KeyError as exc:
        raise ValueError(
            f"profile handle {handle.name!r} does not belong to this Study's "
            "fitted parameter schema."
        ) from exc
    if int(entry.size) != 1:
        raise TypeError(
            f"profile handle {handle.name!r} owns {entry.size} flat components; "
            "selecting one implicitly would be ambiguous."
        )

    is_shared = any(
        candidate.identity == entry.handle.identity for candidate in problem.study.parameters.shared
    )
    occurrences = tuple(entry.occurrences)
    if is_shared:
        if owner is not None:
            raise ValueError(
                f"profile handle {entry.handle.name!r} is shared and therefore has "
                f"no owner; remove owner={owner!r}."
            )
        if len(occurrences) != 1 or occurrences[0].owner is not None:
            raise ValueError(
                f"profile: malformed shared schema occurrence for {entry.handle.name!r}."
            )
        occurrence = occurrences[0]
    else:
        if owner is None:
            available = tuple(occurrence.owner for occurrence in occurrences)
            raise ValueError(
                f"profile handle {entry.handle.name!r} varies by owner; pass one "
                f"of owner={available}."
            )
        matches = tuple(occurrence for occurrence in occurrences if occurrence.owner == owner)
        if len(matches) != 1:
            available = tuple(occurrence.owner for occurrence in occurrences)
            raise ValueError(
                f"profile handle {entry.handle.name!r} has no unique occurrence "
                f"owned by {owner!r}; available owners are {available}."
            )
        occurrence = matches[0]

    if len(occurrence.columns) != 1:
        raise ValueError(
            f"profile: scalar handle {entry.handle.name!r} occurrence owns columns "
            f"{occurrence.columns}; expected exactly one."
        )
    return entry, occurrence


def _profile_facade(
    problem: EstimationProblem,
    fit: OptResult,
    handle: Any,
    *,
    owner: Any = None,
    optimizer: LevenbergMarquardt | TrustRegion | None = None,
    levels: tuple = (0.95,),
    bounds: tuple | None = None,
    max_steps: int = 24,
    multistart: int = 0,
    multistart_scale: float = 1.0,
) -> ProfileResult:
    """Profile of an ``InferenceProblem`` handle (the estimation-problem route of ``profile``).

    ``handle`` is resolved through the fit's handle-based parameter schema, never
    through lowering-generated flat names. Shared handles require no ``owner``;
    cluster-varying handles require the exact owner ID. At every grid value the
    selected physical coordinate is held fixed exactly and every other fitted
    coordinate is reoptimized with the selected optimizer. Recipe bounds and an
    optional narrower target ``bounds=(lo, hi)`` are both honored.

    For an MLE this is a profile likelihood. For a MAP objective it is a
    penalized posterior profile; the curve and refits are valid, while nominal
    Wilks levels are diagnostic rather than calibrated frequentist coverage.

    ``max_steps`` is the number of certified grid points each direction may
    add; ``optimizer.max_steps`` is the budget of one refit. The walk starts
    from a curvature-sized step and adapts it toward a rise of a quarter of
    the threshold per point, so closing a side costs four refits for an
    exactly Gaussian curve and more for a curved one — leave ``max_steps`` at
    its default unless the walk is being cut deliberately. An uncertified
    refit is retried, up to twice per direction, with half the step and twice
    the per-refit budget; a direction that still cannot certify ends as
    ``"nonconvergence"`` or ``"solver_failure"`` in
    [wing_end][diff_soup.inference.statistics.profile.ProfileResult],
    which distinguishes an empty or short wing from a genuinely open one.
    """
    if not isinstance(fit, OptResult):
        raise TypeError("profile fit must be an optimization OptResult.")
    _validate_fit(problem, fit)
    if getattr(problem.diagnostic_problem, "is_decision_variable", False):
        n_theta = sum(
            len(occurrence.columns)
            for entry in problem.parameter_schema.entries
            for occurrence in entry.occurrences
        )
        raise ValueError(
            "profile refits every other coordinate at each point of the curve, and the "
            "window starts of a multiple-shooting fit are not parameters; refit with single "
            f"shooting from fit.params[:{n_theta}] and profile that fit."
        )
    entry, occurrence = _resolve_occurrence(problem, handle, owner)

    if optimizer is None:
        optimizer = LevenbergMarquardt()
    if not isinstance(optimizer, (LevenbergMarquardt, TrustRegion)):
        raise TypeError(
            "profile nuisance refits require LevenbergMarquardt, or TrustRegion on a "
            "lowered study; these optimizers hold the selected coordinate exactly fixed."
        )

    column = int(occurrence.columns[0])
    direction = np.zeros(int(problem.n_params), dtype=float)
    direction[column] = 1.0
    result = _profile_engine(
        problem,
        direction,
        fit=fit,
        optimizer=optimizer,
        levels=levels,
        bounds=bounds,
        max_steps=max_steps,
        multistart=multistart,
        multistart_scale=multistart_scale,
    )

    resolved = resolve_decision_bounds(problem)
    recipe_bounds = None
    active_bound_coordinates = ()
    if resolved is not None:
        lower = np.asarray(resolved[0], dtype=float)
        upper = np.asarray(resolved[1], dtype=float)
        recipe_bounds = (
            float(lower[column]),
            float(upper[column]),
        )
        fitted = np.asarray(fit.params, dtype=float)
        coordinates = _schema_coordinates(problem.parameter_schema, fitted.size)
        active_bound_coordinates = tuple(
            coordinates[index].name
            for index, value in enumerate(fitted)
            if (np.isfinite(lower[index]) and np.isclose(value, lower[index]))
            or (np.isfinite(upper[index]) and np.isclose(value, upper[index]))
        )
    label = str(entry.handle.name)
    if occurrence.owner is not None:
        label = f"{label}@{occurrence.owner}"
    warnings_out = []
    if isinstance(problem.objective, MAP):
        warnings_out.append(
            "This is a penalized posterior profile. Its curve describes the MAP "
            "objective, but nominal Wilks levels are not calibrated frequentist coverage."
        )
    if active_bound_coordinates:
        warnings_out.append(
            "The baseline fit has coordinate(s) on a finite bound at flat "
            f"coordinate(s) {active_bound_coordinates}. Ordinary chi-squared Wilks "
            "calibration can fail at a boundary; interpret profile shape and open "
            "directions rather than nominal coverage."
        )
    provenance = dict(result.provenance)
    provenance.update(
        {
            "coordinate": "Recipe scalar handle",
            "handle_target": {
                "name": entry.handle.name,
                "identity": tuple(entry.handle.identity),
                "owner": occurrence.owner,
                "column": column,
                "schema_fingerprint": problem.parameter_schema.fingerprint,
                "study_fingerprint": problem.study.fingerprint,
                "objective": type(problem.objective).__name__,
                "recipe_bounds": recipe_bounds,
                "requested_target_bounds": bounds,
            },
            "nuisance_optimizer": type(optimizer).__name__,
            "warnings": tuple(warnings_out),
        }
    )
    return dataclasses.replace(result, target=label, provenance=provenance)


def profile(
    problem: Any,
    fit: OptResult,
    handle: Any,
    *,
    owner=None,
    optimizer=None,
    levels=(0.95,),
    bounds=None,
    max_steps=24,
    multistart=0,
    multistart_scale=1.0,
):
    """Compute the profile likelihood of one scalar parameter, refitting every other parameter at each value.

    Each side of the fit is walked until the profile passes the threshold of the largest level,
    a bound, or `max_steps`. A side that stays below the threshold is an open direction.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] or
            [`LoweredStudy`][diff_soup.inference.LoweredStudy] whose measurements all declare a
            likelihood.
        fit: The [`OptResult`][diff_soup.inference.OptResult] of `ds.inference.optsolve` on
            `problem`.
        handle: A scalar handle from `spec.param`.
        owner: The member whose value is walked, required for a handle that varies by member
            and refused for a shared one.
        optimizer: The optimizer of the refits,
            [`LevenbergMarquardt`][diff_soup.inference.LevenbergMarquardt], or
            [`TrustRegion`][diff_soup.inference.TrustRegion] on a lowered study, or `None` for
            `LevenbergMarquardt()`.
        levels: The confidence levels whose likelihood-ratio thresholds the result uses.
        bounds: An optional range `(lo, hi)` for the walked value, which must contain the fit.
        max_steps: The most accepted steps on each side.
        multistart: The number of extra jittered starts per refit, of which the best is kept.
        multistart_scale: The size of the jitter, relative to `1 + |value|` of each coordinate.

    Returns:
        A `ProfileResult`, whose `interval(level)` is the profile interval.
    """
    if not isinstance(problem, EstimationProblem):
        raise TypeError("profile requires an InferenceProblem.")
    require_likelihoods(problem, "profile")
    return _profile_facade(
        problem,
        fit,
        handle,
        owner=owner,
        optimizer=optimizer,
        levels=levels,
        bounds=bounds,
        max_steps=max_steps,
        multistart=multistart,
        multistart_scale=multistart_scale,
    )
