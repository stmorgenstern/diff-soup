"""sample.py — normalized log-posterior targets and whitening for sampling.

The sampling surface starts from one object: a [LogDensity][], the
normalized log posterior of a data-bearing
[Experiment][diff_soup.inference.statistics.measurement.Experiment] under a proper
[Prior][diff_soup.inference.statistics.priors.Prior]. Build it with
[log_posterior][]:

* the likelihood term is assembled from each scored response channel's own
  normalized ``log_prob`` — any `Likelihood` family works here,
  including `CorrelatedGaussian` channels that have no
  least-squares lowering;
* with a [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]
  the target lives in the inference coordinate ``z``: the physically
  declared prior is pulled back exactly, ``log |dx/dz|`` included;
* gradients are **forward-assembled** by default (``jax.jacfwd`` — the mode
  the solver suite's composable forward-sensitivity runs support).
  ``gradient="reverse"`` is admitted only after a build-time check that
  reverse mode reproduces the forward gradient at the initial point, and
  ``gradient=None`` declares a gradient-free target.

Every [LogDensity][] can audit its own gradient:
[LogDensity.verify_gradient][] compares the stored gradient against
Richardson-extrapolated central differences and records a
[GradientCheck][]; gradient-based samplers gate on
[LogDensity.gradient_verified][].

[whitening][] builds the affine [WhiteningTransform][] that maps a
unit-Gaussian coordinate ``w`` to the parameter coordinate of a proper
Gaussian source — a [LaplacePosterior][] (``z = mode + L⁻ᵀ w`` against
the stored precision Cholesky factor) or a
[GaussianPrior][diff_soup.inference.statistics.priors.GaussianPrior]. Improper sources
are refused: whitening needs a finite SPD covariance.

[sample][] draws from a [LogDensity][] with `blackjax
<https://github.com/blackjax-devs/blackjax>`_: a gradient-free Gaussian random walk
(``kernel="rmh"``, the default — any jittable target works, solver-backed
ones included) or NUTS (``kernel="nuts"``, admitted only after the target's
gradient audit passes). Passing ``whitening=`` runs the chains in the
whitened coordinate as a pure sampler preconditioner. The result is a frozen
[Draws][] container: positions in the target's own coordinate,
per-transition diagnostics, a map to the physical coordinate, and
posterior-predictive replication.

[sbc][] closes the loop with simulation-based calibration: draw
parameters from the prior, simulate data through the experiment's own
likelihoods, sample each synthetic posterior, and test the rank statistics
of the prior draws for uniformity. A calibrated pipeline — model, prior,
and sampler together — produces uniform ranks; anything else is reported
with per-parameter chi-square evidence in an [SbcReport][].
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.inference.data.deferred import DeferredValues
from diff_soup.inference.solvers.tangent import deferred_value_and_grad
from diff_soup.inference.statistics.experiment import _channel_likelihood_view
from diff_soup.inference.statistics.laplace import LaplacePosterior
from diff_soup.inference.statistics.parameters import Transform
from diff_soup.inference.statistics.priors import GaussianPrior, Prior
from diff_soup.inference.statistics.report import IdentifiedSubspace

__all__ = [
    "Draws",
    "GradientCheck",
    "LogDensity",
    "SbcReport",
    "WhiteningTransform",
    "log_posterior",
    "sample",
    "sbc",
    "whitening",
]

GRADIENT_MODES = ("forward", "reverse", None)

# Relative agreement demanded between reverse- and forward-mode gradients at
# the initial point before a reverse-mode target is admitted.
_REVERSE_GATE_RTOL = 1e-6


@dataclasses.dataclass(frozen=True)
class GradientCheck:
    """The recorded outcome of one finite-difference gradient audit.

    The pass criterion is the whole-vector relative disagreement
    ``norm_rel_error`` — the measure a gradient consumer actually feels
    (proposals and optimizer steps take the gradient as a vector, so a small
    deviation on a component that is tiny next to the dominant ones cannot
    change behavior). The per-component ``max_rel_error`` is kept as a
    diagnostic: it exposes which directions the finite-difference probe and
    the stored gradient disagree on, even when that disagreement is too
    small in vector terms to matter.

    Attributes:
        at: The point the gradient was checked at.
        gradient: The stored (AD) gradient at ``at``.
        fd_gradient: The Richardson-extrapolated central-difference gradient.
        fd_spread: Per-component disagreement between the two central-
            difference step levels — the probe's own resolving power; gaps
            inside it cannot be attributed to the gradient.
        norm_rel_error: ``‖gradient − fd‖ / max(‖gradient‖, ‖fd‖)`` — the
            quantity ``passed`` is judged on.
        max_rel_error: Largest per-component relative disagreement
            (diagnostic only).
        rtol: The tolerance the check was run against.
        passed: Whether the vector-level disagreement, less the probe's
            noise allowance, is within ``rtol``.
    """

    at: Array
    gradient: Array
    fd_gradient: Array
    fd_spread: Array
    norm_rel_error: float
    max_rel_error: float
    rtol: float
    passed: bool


@dataclasses.dataclass(frozen=True)
class LogDensity:
    """A normalized log-density target with its coordinates and gradient.

    Calling the object evaluates the scalar log density;
    [value_and_grad][] additionally returns the gradient when one was
    built. The target is frozen — [verify_gradient][] records its audit
    result in place, and everything else is read-only.

    Attributes:
        coordinate: The coordinate the density lives in — ``"x"`` (physical)
            or ``"z"`` (inference).
        names: Per-scalar parameter names, in flat-vector order.
        init: The natural starting point (the experiment's declared values,
            mapped into ``coordinate``).
        gradient: How the gradient was built — ``"forward"`` (``jax.jacfwd``),
            ``"reverse"`` (``jax.grad``, admitted by the build-time agreement
            gate), or ``None`` (gradient-free target).
        gradient_check: The most recent [GradientCheck][], or ``None``
            when [verify_gradient][] has not been run.
    """

    _coordinate: str
    names: tuple
    init: Array
    gradient: Optional[str]
    _fn: Callable
    _grad: Optional[Callable]
    gradient_check: Optional[GradientCheck] = None
    # For a "z"-coordinate target, the elementwise map z -> x (populated by
    # log_posterior when built with a parameterization) so downstream results
    # can be reported in the physical coordinate.
    _to_physical: Optional[Callable] = None
    _parameter_schema: Any = None
    _problem: Any = None
    _fit: Any = None
    # ``z -> (value, gradient)`` from one pass of the gradient's own mode (see
    # _build_value_and_gradient); ``None`` falls back to two evaluations.
    _value_and_grad: Optional[Callable] = None

    def __call__(self, z: Array) -> Array:
        """Return the normalized log density at ``z``."""
        return self._fn(jnp.asarray(z))

    def value_and_grad(self, z: Array) -> tuple:
        """Return ``(log density, gradient)`` at ``z``.

        Both come from one pass of the gradient's mode: the forward push that
        assembles a ``"forward"`` gradient, or the forward sweep of a
        ``"reverse"`` one, also yields the value, so a solver-backed density
        is integrated once rather than once for the value and again for the
        gradient.

        Raises:
            ValueError: If the target was built with ``gradient=None``;
                rebuild with ``gradient="forward"`` to get a differentiable
                target.
        """
        if self._grad is None:
            raise ValueError(
                "LogDensity.value_and_grad: this target was built with "
                "gradient=None, so no gradient exists. Rebuild it with "
                "posterior(..., gradient='forward') for a forward-assembled "
                "gradient."
            )
        z = jnp.asarray(z)
        if self._value_and_grad is not None:
            return self._value_and_grad(z)
        return self._fn(z), self._grad(z)

    def _physical(self, position: Array) -> Array:
        """Map one position from the target coordinate to physical ``x``.

        A target already expressed in ``x`` returns the input unchanged. A
        transformed ``z`` target must carry the map installed by its builder;
        this is the single-position analogue of [Draws.to_physical][] and
        makes initialization/provenance checks possible before sampling.
        """
        position = jnp.asarray(position)
        if self._coordinate == "x":
            return position
        if self._to_physical is None:
            raise ValueError(
                "LogDensity.to_physical: this target lives in coordinate 'z' "
                "but carries no z -> x map."
            )
        return self._to_physical(position)

    @property
    def gradient_verified(self) -> bool:
        """Whether a recorded gradient audit exists and passed."""
        return self.gradient_check is not None and self.gradient_check.passed

    def verify_gradient(self, at: Any = None, rtol: float = 1e-3) -> GradientCheck:
        """Audit the stored gradient against central finite differences.

        Each component is probed with Richardson-extrapolated central
        differences (two shrinking steps, so smooth truncation error cancels
        to fourth order). The pass criterion is vector-level: the norm of
        the disagreement, less the probe's own noise allowance (the
        disagreement between the two step levels), must be within ``rtol``
        of the gradient norm. That is the measure a gradient consumer
        feels — a percent-level deviation on a component that is tiny next
        to the dominant ones cannot change a proposal or a step, and
        finite differences of a numerically solved density carry solver
        truncation noise that is not gradient error. The per-component
        relative disagreement is kept on the check as a diagnostic. The
        resulting [GradientCheck][] is recorded on the target and
        returned.

        Args:
            at: The point to audit at; defaults to [init][].
            rtol: Largest acceptable vector-relative disagreement.

        Returns:
            The recorded [GradientCheck][].

        Raises:
            ValueError: If the target has no gradient (``gradient=None``).
        """
        if self._grad is None:
            raise ValueError(
                "LogDensity.verify_gradient: this target was built with "
                "gradient=None, so there is no gradient to verify. Rebuild it "
                "with posterior(..., gradient='forward')."
            )
        z0 = np.asarray(self.init if at is None else at, dtype=float)
        grad = np.asarray(self._grad(jnp.asarray(z0)), dtype=float)

        def scalar(z: np.ndarray) -> float:
            return float(self._fn(jnp.asarray(z)))

        fd = np.zeros_like(z0)
        spread = np.zeros_like(z0)
        for j in range(z0.size):
            h = 1e-4 * max(1.0, abs(float(z0[j])))

            def central(step: float, j: int = j) -> float:
                zp = z0.copy()
                zp[j] += step
                zm = z0.copy()
                zm[j] -= step
                return (scalar(zp) - scalar(zm)) / (2.0 * step)

            coarse = central(h)
            fine = central(0.5 * h)
            fd[j] = (4.0 * fine - coarse) / 3.0
            spread[j] = abs(fine - coarse)

        scale = max(
            float(np.max(np.abs(grad), initial=0.0)), float(np.max(np.abs(fd), initial=0.0))
        )
        denom = np.maximum(np.maximum(np.abs(grad), np.abs(fd)), max(1e-8 * scale, 1e-300))
        max_rel_error = float(np.max(np.abs(grad - fd) / denom, initial=0.0))

        norm_scale = max(float(np.linalg.norm(grad)), float(np.linalg.norm(fd)), 1e-300)
        norm_rel_error = float(np.linalg.norm(grad - fd)) / norm_scale
        # Gap the probe itself cannot resolve: the Richardson-level spread,
        # judged at the same 50x factor the library's finite-difference
        # contracts use.
        noise_allowance = 50.0 * float(np.linalg.norm(spread)) / norm_scale
        passed = norm_rel_error <= float(rtol) + noise_allowance

        check = GradientCheck(
            at=jnp.asarray(z0),
            gradient=jnp.asarray(grad),
            fd_gradient=jnp.asarray(fd),
            fd_spread=jnp.asarray(spread),
            norm_rel_error=norm_rel_error,
            max_rel_error=max_rel_error,
            rtol=float(rtol),
            passed=bool(passed),
        )
        object.__setattr__(self, "gradient_check", check)
        return check

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        *,
        init: Any,
        names: Any = None,
        _coordinate: str = "x",
        gradient: Optional[str] = "forward",
    ) -> "LogDensity":
        """Wrap a user-supplied log-density callable as a sampling target.

        Use this for targets that are not experiment posteriors — analytic
        densities, hand-assembled models, or custom pipelines — so they get
        the same gradient handling, audit, and sampler surface as
        [log_posterior][] targets.

        Args:
            fn: A pure scalar log-density callable of one flat parameter
                vector, safe under ``jax.jit``.
            init: The starting point, shape ``(n,)``.
            names: Optional per-scalar parameter names (defaults to
                positional labels).
            coordinate: The coordinate the density lives in — ``"x"``
                (physical, default) or ``"z"`` (inference).
            gradient: ``"forward"`` (default, ``jax.jacfwd``), ``"reverse"``
                (``jax.grad``, admitted only after it reproduces the forward
                gradient at ``init``), or ``None`` (gradient-free).

        Returns:
            The [LogDensity][] target.

        Raises:
            ValueError: For an unknown ``gradient`` or ``coordinate``, a
                ``names`` length that does not match ``init``, or a
                reverse-mode gradient that fails or disagrees at ``init``.
        """
        if gradient not in GRADIENT_MODES:
            raise ValueError(
                f"LogDensity.from_callable: gradient must be one of "
                f"{GRADIENT_MODES}; got {gradient!r}."
            )
        if _coordinate not in ("x", "z"):
            raise ValueError(
                "LogDensity.from_callable: coordinate must be 'x' (physical) "
                f"or 'z' (inference); got {_coordinate!r}."
            )
        init = jnp.asarray(init)
        n = int(init.shape[0])
        names = tuple(names) if names is not None else tuple(f"theta[{i}]" for i in range(n))
        if len(names) != n:
            raise ValueError(f"LogDensity.from_callable: {len(names)} names for {n} parameters.")
        grad_fn = _build_gradient(fn, init, gradient, caller="LogDensity.from_callable")
        return cls(
            _coordinate=_coordinate,
            names=names,
            init=init,
            gradient=gradient,
            _fn=fn,
            _grad=grad_fn,
            _value_and_grad=_build_value_and_gradient(fn, gradient),
        )


def _kept_data(experiment: Any, scored: tuple) -> dict:
    """Return each scored channel's observed values with masked frames removed."""
    kept = {}
    for c in scored:
        values = experiment.data[c.name]
        if isinstance(values, DeferredValues):
            values = values.materialize()
        if c.mask is not None:
            keep = np.where(np.asarray(c.mask).astype(bool))[0]
            values = jnp.asarray(values)[jnp.asarray(keep)]
        kept[c.name] = values
    return kept


def _build_gradient(
    fn: Callable, init: Array, gradient: Optional[str], *, caller: str
) -> Optional[Callable]:
    """Build the gradient of a scalar target under the declared mode.

    ``"forward"`` returns ``jax.jacfwd(fn)``. ``"reverse"`` returns
    ``jax.grad(fn)`` only after it reproduces the forward gradient at
    ``init`` — a failing or disagreeing reverse pass raises with the reason
    (composable forward-sensitivity solver runs support ``jax.jacfwd``
    only). ``None`` returns ``None`` (a gradient-free target).
    """
    if gradient == "forward":
        return jax.jacfwd(fn)
    if gradient == "reverse":
        forward_grad = np.asarray(jax.jacfwd(fn)(init), dtype=float)
        try:
            reverse_grad = np.asarray(jax.grad(fn)(init), dtype=float)
        except Exception as exc:
            raise ValueError(
                f"{caller}: the reverse-mode gradient failed at the initial "
                f"point ({exc}). Composable forward-sensitivity solver runs "
                "support jax.jacfwd only — build with gradient='forward'."
            ) from exc
        atol = _REVERSE_GATE_RTOL * max(1.0, float(np.max(np.abs(forward_grad), initial=0.0)))
        if not np.allclose(reverse_grad, forward_grad, rtol=_REVERSE_GATE_RTOL, atol=atol):
            raise ValueError(
                f"{caller}: the reverse-mode gradient disagrees with the "
                f"forward-assembled gradient at the initial point (reverse "
                f"{reverse_grad}, forward {forward_grad}). Reverse mode is not "
                "trustworthy for this target — build with gradient='forward'."
            )
        return jax.grad(fn)
    return None


def _build_value_and_gradient(fn: Callable, gradient: Optional[str]) -> Optional[Callable]:
    """Build ``z -> (fn(z), gradient)`` from one derivative pass of the declared mode.

    ``"forward"`` takes the value from the primal of the forward push that
    assembles the gradient (``deferred_value_and_grad``), the push
    ``jax.jacfwd`` makes, so the gradient is the one [_build_gradient][]
    builds and the value is computed once, not once more beside the push.
    ``"reverse"`` is ``jax.value_and_grad``, whose forward sweep already
    carries the value. ``None`` returns ``None``.
    """
    if gradient == "forward":

        def value_and_grad(z: Array) -> tuple:
            value, gradient_at = deferred_value_and_grad(fn, z, ad="forward")
            return value, gradient_at()

        return value_and_grad
    if gradient == "reverse":
        return jax.value_and_grad(fn)
    return None


def log_posterior(
    experiment: Any,
    *,
    prior: Optional[Prior],
    parameterization: Any = None,
    gradient: Optional[str] = "forward",
    parameters: Any = None,
) -> LogDensity:
    """Build the normalized log posterior of a data-bearing experiment.

    The target is ``log p(data | theta) + log p(theta)``: every scored
    response channel contributes its likelihood's own normalized
    ``log_prob`` (so correlated and — as they arrive — non-Gaussian families
    are first-class here), and the prior must be a proper probability model.
    With a ``parameterization`` the target lives in the inference coordinate
    ``z``: the physically declared prior is pulled back exactly, including
    the ``log |dx/dz|`` change-of-variables term, and ``init`` is the
    declared values mapped through the inverse coordinate map.

    Args:
        experiment: The data-bearing
            [Experiment][diff_soup.inference.statistics.measurement.Experiment]
            (attach observations with ``with_data`` first).
        prior: A proper [Prior][diff_soup.inference.statistics.priors.Prior]
            declared on the physical coordinate ``"x"``. Required: without a
            normalized prior the posterior density is not a probability
            model, so there is nothing well-defined to sample.
        parameterization: Optional
            [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]
            mapping the flat inference vector ``z`` to the physical ``x``.
        gradient: ``"forward"`` (default) assembles the gradient with
            ``jax.jacfwd`` — the mode composable forward-sensitivity solver
            runs support; ``"reverse"`` uses ``jax.grad`` and is admitted
            only after it reproduces the forward gradient at ``init``;
            ``None`` builds a gradient-free target.
        parameters: Optional parameter groups selecting the learnable theta
            (as accepted by ``Experiment.predictor``).

    Returns:
        The [LogDensity][] target.

    Raises:
        ValueError: If the prior is missing, improper for this use, or sized
            wrong; if no response channel has attached data; if ``gradient``
            is unknown; or if a reverse-mode gradient fails or disagrees
            with the forward gradient at the initial point.
    """
    from diff_soup.inference.statistics.measurement import require_likelihoods

    require_likelihoods(experiment, "log_posterior")
    if gradient not in GRADIENT_MODES:
        raise ValueError(
            f"log_posterior: gradient must be one of {GRADIENT_MODES}; got {gradient!r}."
        )
    if prior is None:
        raise ValueError(
            "log_posterior needs a proper prior: without a normalized prior "
            "density the posterior is not a probability model, so there is "
            "nothing well-defined to sample or integrate. Declare one (e.g. "
            "GaussianPrior) on the physical coordinate 'x'."
        )
    if getattr(prior, "coordinate", "x") != "x":
        raise ValueError(
            "log_posterior: the prior must be declared on the physical "
            f"coordinate 'x'; got coordinate {prior.coordinate!r}. With a "
            "parameterization the pullback to the inference coordinate is "
            "applied automatically."
        )

    predictor = experiment.predictor(parameters=parameters)
    theta_names = tuple(predictor.theta_names)
    n = len(theta_names)
    prior_size = getattr(prior, "size", None)
    if prior_size is not None and int(prior_size) != n:
        raise ValueError(
            f"log_posterior: the prior covers {int(prior_size)} scalars but the "
            f"experiment has {n} learnable parameters ({theta_names})."
        )

    response = tuple(c for c in experiment.channels if c.role == "response")
    scored = tuple(c for c in response if c.name in experiment.data)
    if not scored:
        raise ValueError(
            "log_posterior needs observed response data; attach it with "
            "Experiment.with_data(<channel_name>=values) first."
        )
    kept = _kept_data(experiment, scored)

    def log_likelihood(x_flat: Array) -> Array:
        preds, _ = predictor(x_flat)
        total = jnp.zeros(())
        for c in scored:
            likelihood, context = _channel_likelihood_view(c, preds[c.name])
            total = total + likelihood.log_prob(
                kept[c.name],
                preds[c.name],
                context=context,
            )
        return total

    x0 = jnp.asarray(predictor.x0)
    to_physical: Optional[Callable] = None
    if parameterization is None:
        coordinate = "x"
        names = theta_names
        init = x0

        def fn(z: Array) -> Array:
            return log_likelihood(z) + prior.log_prob(z)

    else:
        p12n = parameterization
        if p12n.n_parameters != n:
            raise ValueError(
                f"log_posterior: the parameterization covers {p12n.n_parameters} "
                f"scalars but the experiment has {n} learnable parameters "
                f"({theta_names})."
            )
        coordinate = "z"
        names = tuple(p12n.names())
        init = p12n.inverse(x0)
        to_physical = p12n.transform

        def fn(z: Array) -> Array:
            x = p12n.transform(z)
            return log_likelihood(x) + prior.log_prob(x) + jnp.sum(p12n.log_abs_det_jacobian(z))

    grad_fn = _build_gradient(fn, init, gradient, caller="log_posterior")

    return LogDensity(
        _coordinate=coordinate,
        names=names,
        init=jnp.asarray(init),
        gradient=gradient,
        _fn=fn,
        _grad=grad_fn,
        _to_physical=to_physical,
        _value_and_grad=_build_value_and_gradient(fn, gradient),
    )


class WhiteningTransform(Transform):
    """An affine map from a unit-Gaussian coordinate to a parameter coordinate.

    ``forward(w) = center + matrix @ w`` maps white noise ``w ~ N(0, I)`` to
    the source's Gaussian, and ``inverse`` maps back. Unlike the elementwise
    transforms, the Jacobian is a constant full matrix, so
    [log_abs_det_jacobian][] returns the scalar total ``log |det(matrix)|``
    directly.

    Attributes:
        center: The Gaussian's mean/mode, shape ``(n,)``.
        matrix: The coloring matrix ``A`` with ``A Aᵀ`` equal to the source
            covariance, shape ``(n, n)``.
        matrix_inverse: The exact inverse of ``matrix``.
        log_det: The scalar ``log |det(matrix)|``.
        coordinate: Coordinate of the Gaussian source and the target this
            transform may precondition.
    """

    center: Array
    matrix: Array
    matrix_inverse: Array
    log_det: Array
    coordinate: str = eqx.field(static=True)

    def forward(self, z: Array) -> Array:
        """Map the white coordinate ``z`` to the parameter coordinate."""
        return self.center + self.matrix @ jnp.asarray(z)

    def inverse(self, x: Array) -> Array:
        """Map a parameter-coordinate point back to the white coordinate."""
        return self.matrix_inverse @ (jnp.asarray(x) - self.center)

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return the scalar total ``log |det(matrix)|`` (position-independent)."""
        del z
        return self.log_det


def _whitening_from_precision_factor(
    center: Array, lower_factor: Array, *, coordinate: str
) -> WhiteningTransform:
    """Build the whitening map of a Gaussian given its lower precision Cholesky."""
    center = jnp.asarray(center)
    lower = jnp.asarray(lower_factor)
    eye = jnp.eye(center.shape[0], dtype=center.dtype)
    matrix = solve_triangular(lower.T, eye, lower=False)
    return WhiteningTransform(
        center=center,
        matrix=matrix,
        matrix_inverse=lower.T,
        log_det=-jnp.sum(jnp.log(jnp.diag(lower))),
        coordinate=coordinate,
    )


def whitening(source: Any) -> WhiteningTransform:
    """Build the affine whitening map of a proper Gaussian source.

    The returned [WhiteningTransform][] maps ``w ~ N(0, I)`` to the
    source's Gaussian: ``forward(w) = center + A w`` with ``A Aᵀ`` equal to
    the source covariance. Composing a log-density target with it gives a
    unit-curvature coordinate for samplers to work in.

    Args:
        source: A [LaplacePosterior][diff_soup.inference.statistics.laplace.LaplacePosterior]
            (``center`` = mode, ``A = L⁻ᵀ`` from the stored precision
            Cholesky factor) or a proper
            [GaussianPrior][diff_soup.inference.statistics.priors.GaussianPrior]
            (``center`` = mean, ``A`` from the precision's Cholesky factor).

    Returns:
        The [WhiteningTransform][].

    Raises:
        ValueError: If the source is an
            [IdentifiedSubspace][diff_soup.inference.statistics.report.IdentifiedSubspace]
            (an improper source: some directions were left undetermined, so
            no finite covariance exists), or a Gaussian prior whose precision
            is singular or not positive definite.
        TypeError: For any other source type.
    """
    if isinstance(source, LaplacePosterior):
        return _whitening_from_precision_factor(
            source.mode, source.precision_factor, coordinate=source.coordinate
        )
    if isinstance(source, IdentifiedSubspace):
        raise ValueError(
            "whitening: an IdentifiedSubspace is an improper source — it "
            "records the parameter directions an analysis left undetermined, "
            "so no finite (SPD) covariance exists to whiten. Close the "
            "undetermined directions with a proper prior or re-parameterize, "
            "then whiten the resulting proper posterior."
        )
    if isinstance(source, GaussianPrior):
        precision = np.asarray(source.precision(), dtype=float)
        try:
            lower = np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "whitening: the prior precision is singular or not positive "
                "definite, so the prior has no finite SPD covariance to "
                "whiten. Give every parameter a proper (finite, nonzero) "
                "prior scale."
            ) from exc
        return _whitening_from_precision_factor(
            source.mean, jnp.asarray(lower), coordinate=source.coordinate
        )
    raise TypeError(
        f"whitening: unsupported source {type(source).__name__}; supported "
        "sources are LaplacePosterior (posterior curvature at a fitted mode) "
        "and GaussianPrior (a proper Gaussian prior)."
    )


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

#: Per-component random-walk proposal standard deviation, ``2.38 / sqrt(d)``
#: — the classic optimal scaling for a unit-covariance Gaussian target. With
#: ``whitening=`` the chains run in the whitened coordinate where the target
#: really is near unit scale; without it the default assumes a roughly
#: unit-scale target (pass a whitening transform for anything anisotropic).
_RW_OPTIMAL_SCALE = 2.38

#: Standard deviation of the per-chain overdispersed start jitter, applied in
#: the sampling coordinate (the whitened ``w`` when a whitening transform is
#: given, the target's own coordinate otherwise).
_CHAIN_JITTER = 0.5

#: How [sample][] spreads its chains over the target: ``"vmap"`` evaluates
#: every chain in one batched call, ``"sequential"`` runs them one after
#: another with no added batch axis, and ``"devices"`` gives each local device
#: its own chains. See the ``chain_map`` argument of [sample][].
CHAIN_MAPS = ("vmap", "sequential", "devices")


def _with_stored_gradient(logdensity: LogDensity) -> Callable:
    """Route reverse-mode differentiation of a target through its stored gradient.

    The returned callable evaluates the target's density unchanged but carries
    a custom reverse-mode rule whose pullback is ``cotangent * gradient``.
    Gradient consumers that call ``jax.grad``/``jax.value_and_grad``
    internally (the NUTS integrator does) therefore consume exactly the
    gradient the target stores — the one [LogDensity.verify_gradient][]
    audits — instead of re-deriving their own. The forward half of the rule
    evaluates [LogDensity.value_and_grad][], so each value-and-gradient
    request is one derivative pass of the target.
    """
    fn = logdensity._fn

    @jax.custom_vjp
    def wrapped(z: Array) -> Array:
        return fn(z)

    def wrapped_fwd(z: Array) -> tuple:
        return logdensity.value_and_grad(z)

    def wrapped_bwd(grad: Array, cotangent: Array) -> tuple:
        return (cotangent * grad,)

    wrapped.defvjp(wrapped_fwd, wrapped_bwd)
    return wrapped


@dataclasses.dataclass(frozen=True)
class Draws:
    """The MCMC draws of `sample`, in the parameters' own coordinates.

    `draws[handle]` returns the draws of one parameter, or a dictionary by member for a parameter
    that varies by member.

    Attributes:
        values: The draws, shaped `(chains, draws, parameters)`.
        names: The label of each parameter column of `values`.
        info: Sampler statistics per draw, `acceptance_rate`, and for NUTS `is_divergent` and
            `num_integration_steps`.
        chain_map: How the chains ran, `"vmap"`, `"sequential"` or `"devices"`.
    """

    values: Array
    names: tuple
    info: dict
    _parameter_schema: Any = None
    _problem: Any = None
    chain_map: str = "vmap"

    def __getitem__(self, handle):
        """Return parameter samples with leading ``(chains, draws)`` axes."""
        if self._parameter_schema is None:
            raise TypeError("handle lookup requires samples from an InferenceProblem.")
        entry = self._parameter_schema.entry(handle)
        result = {}
        for occurrence in entry.occurrences:
            block = self.values[..., jnp.asarray(occurrence.columns)]
            result[occurrence.owner] = block[..., 0] if entry.size == 1 else block
        return result[None] if tuple(result) == (None,) else result

    @property
    def positions(self):
        """The draws, the same array as `values`."""
        return self.values


def _target_batches_its_solves(problem: Any) -> tuple[bool, bool]:
    """Whether a sampled problem evaluates its solves under a batch of its own.

    A multiple-shooting problem maps one solve over its windows, and a
    multi-experiment problem maps one solve over its children. Returns
    ``(batched, sharded)``: whether such a batch exists, and whether a
    multi-experiment batch is spread over a device mesh. Unknown or foreign
    targets answer ``(False, False)``.
    """
    from diff_soup.inference.fitting._sharding import sharding_active
    from diff_soup.inference.fitting.inference_problem import Multishoot

    batched = sharded = False
    seen: set[int] = set()
    pending = [problem]
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(getattr(node, "trajectory", None), Multishoot):
            batched = True
        children = getattr(node, "problems", None)
        if isinstance(children, tuple) and children:
            if len(children) > 1:
                batched = True
                sharded = sharded or sharding_active(getattr(node, "_placement", None))
            pending.extend(children)
        pending.append(getattr(node, "_inference_problem", None))
    return batched, sharded


def _default_chain_map(logdensity: LogDensity, kernel: str) -> str:
    """The chain placement ``sample`` picks when the caller leaves it out.

    Chains are batched (``"vmap"``) unless the target's own solve batch cannot
    take them. A single-device window or experiment batch absorbs an enclosing
    chain batch into its own lanes wherever it is evaluated or
    forward-differentiated, which covers every RMH chain and NUTS chains
    driven by a forward-mode gradient, so linear backends that hold one
    factorization per batch lane (the native cuDSS factor bank) still see a
    single batch axis. A reverse-mode gradient and a batch sharded over a
    device mesh do not absorb it, so such targets run their chains one after
    another (``"sequential"``).
    """
    batched, sharded = _target_batches_its_solves(logdensity._problem)
    if not batched:
        return "vmap"
    if sharded or (kernel == "nuts" and logdensity.gradient == "reverse"):
        return "sequential"
    return "vmap"


def _refuse_shared_storage_on_devices(problem: Any) -> None:
    """Refuse ``chain_map="devices"`` for experiments whose solvers keep native factor storage.

    ``jax.pmap`` runs the chains of every local device in one execution, as a
    layout's data axis over those devices runs experiments, so the refusal of
    such a placement applies (see
    [refuse_shared_storage][diff_soup.inference.engine._plan.refuse_shared_storage]):
    a backend whose factors are not arrays keeps them in this process's
    storage, which does not tell the devices apart. A target without a fitted
    problem, and one device, are accepted.

    Raises:
        ConfigurationError: If this process has two or more devices and an
            experiment's solver resolves to a backend whose factors are not
            arrays.
    """
    from jax.sharding import Mesh

    from diff_soup.inference.engine._layout import _experiment_problems
    from diff_soup.inference.engine._plan import refuse_shared_storage

    devices = jax.local_devices()
    if problem is None or len(devices) < 2:
        return
    experiments = _experiment_problems(getattr(problem, "_inference_problem", problem))
    try:
        refuse_shared_storage(experiments, Mesh(np.asarray(devices), ("data",)))
    except ConfigurationError as error:
        raise ConfigurationError(
            f"sample(chain_map='devices') runs the chains on the {len(devices)} local devices "
            "in one execution, as a layout's data axis over them runs experiments, so the same "
            f"refusal applies: {error} Or keep the solver and sample with chain_map='vmap' or "
            "'sequential'."
        ) from None


def _run_chains(run_chain: Callable, chain_keys: Array, starts: Array, chain_map: str) -> Any:
    """Run one chain per ``(key, start)`` pair and stack the per-chain results.

    Every mode returns what ``jax.vmap(run_chain)`` returns: the same pytree,
    each leaf carrying a leading chain axis in chain order. ``run_chain`` is
    compiled once and reused across chains.
    """
    chains = int(starts.shape[0])
    if chain_map == "vmap":
        return jax.jit(jax.vmap(run_chain))(chain_keys, starts)
    if chain_map == "sequential":
        one_chain = jax.jit(run_chain)
        collected = [one_chain(chain_keys[c], starts[c]) for c in range(chains)]
        return jax.tree.map(lambda *per_chain: jnp.stack(per_chain), *collected)

    devices = jax.local_device_count()
    if chains % devices:
        raise ValueError(
            f"sample: chain_map='devices' places whole chains on local devices, "
            f"so chains must be a multiple of the {devices} local device(s); got "
            f"chains={chains}. Request a multiple of {devices} chains, or use "
            "chain_map='sequential' to run the chains one after another."
        )
    per_device = chains // devices
    device_keys, device_starts = jax.tree.map(
        lambda leaf: leaf.reshape((devices, per_device, *leaf.shape[1:])),
        (chain_keys, starts),
    )

    def device_chains(keys: Array, positions: Array) -> Any:
        return jax.lax.map(lambda chain: run_chain(*chain), (keys, positions))

    placed = jax.pmap(device_chains)(device_keys, device_starts)
    return jax.tree.map(lambda leaf: leaf.reshape((chains, *leaf.shape[2:])), placed)


def sample(
    problem: Any,
    fit: Any = None,
    *,
    kernel: str = "nuts",
    chains: int = 4,
    draws: int = 1000,
    warmup: int = 500,
    key: Array,
    preconditioner: Any = "laplace",
    curvature: str = "observed",
    chain_map: Optional[str] = None,
    **kernel_options,
) -> Draws:
    """Draw from the posterior of a `MAP` problem with MCMC, with the chains started near the fit.

    Each chain starts from a perturbed copy of the fit, and every draw is in the parameters' own
    coordinates. It is refused in a run of several processes.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] with a `MAP`
            objective, or a target from `posterior_target`, which takes no `fit`.
        fit: The `MAP` fit the chains start from.
        kernel: `"nuts"` for the No-U-Turn sampler, or `"rmh"` for random-walk Metropolis, which
            needs no gradient.
        chains: The number of chains.
        draws: The number of draws kept per chain.
        warmup: The number of steps before the kept draws, at least 1 for NUTS, which adapts
            during them.
        key: The JAX random key that fixes the draws.
        preconditioner: `"laplace"` to scale the proposals by the Laplace approximation at the
            fit, or `None` to run without it.
        curvature: The curvature of that Laplace approximation, `"observed"` or `"expected"`.
        chain_map: How the chains run, `"vmap"` as one batched call, `"sequential"` one after
            another, `"devices"` on the local devices, or `None` to choose per problem.
        **kernel_options: `nuts_target_acceptance`, the acceptance rate NUTS adapts its step
            size toward, by default 0.8.

    Returns:
        The [`Draws`][diff_soup.inference.Draws].
    """
    from diff_soup.inference.statistics.laplace import laplace
    from diff_soup.inference.statistics.posterior import posterior_target

    processes = getattr(problem, "processes", 1)
    if isinstance(processes, int) and processes > 1:
        raise ConfigurationError(
            f"sample draws chains of a lowered study whose lanes run in {processes} processes, "
            "and the sampler evaluates the lanes inside JAX's traced density in one process; "
            "sample in a run of one process, or use inf.bootstrap, which runs over every "
            "process."
        )
    if preconditioner not in (None, "laplace"):
        raise ValueError("sample preconditioner must be 'laplace' or None.")
    if curvature not in ("observed", "expected"):
        raise ValueError("sample curvature must be 'observed' or 'expected'.")
    if preconditioner is None and curvature != "observed":
        raise ValueError(
            "sample curvature applies only with preconditioner='laplace'; "
            "pass preconditioner='laplace' or leave curvature at 'observed'."
        )
    if isinstance(problem, LogDensity):
        if fit is not None:
            raise TypeError("sample(target) takes no fit argument.")
        target = problem
        fitted_problem, fitted_point = target._problem, target._fit
    else:
        if fit is None:
            raise TypeError("sample(problem, fit) requires a fitted MAP result.")
        target = posterior_target(problem, fit)
        fitted_problem, fitted_point = problem, fit
    if fitted_problem is None and curvature != "observed":
        raise ValueError(
            "sample curvature needs a target built from a fitted problem, "
            "sample(problem, fit) or posterior_target(problem, fit); "
            "leave curvature at 'observed' for any other target."
        )
    white = None
    if preconditioner == "laplace" and fitted_problem is not None:
        post = laplace(fitted_problem, fitted_point, curvature=curvature)
        physical_white = whitening(post)
        # dx/dz converts covariance in x into covariance in the sampler's z.
        derivative = jax.jacfwd(target._physical)(target.init)
        matrix = jnp.linalg.solve(derivative, physical_white.matrix)
        inverse = jnp.linalg.solve(matrix, jnp.eye(matrix.shape[0], dtype=matrix.dtype))
        white = WhiteningTransform(
            center=target.init,
            matrix=matrix,
            matrix_inverse=inverse,
            log_det=jnp.linalg.slogdet(matrix)[1],
            coordinate=target._coordinate,
        )
    return _sample_target(
        target,
        key=key,
        kernel=kernel,
        chains=chains,
        num_steps=draws,
        num_warmup=warmup,
        whitening=white,
        chain_map=chain_map,
        **kernel_options,
    )


def _sample_target(
    logdensity: LogDensity,
    *,
    key: Array,
    kernel: str = "rmh",
    chains: int = 4,
    num_steps: int,
    whitening: Optional[WhiteningTransform] = None,
    init: Any = None,
    num_warmup: int = 500,
    nuts_target_acceptance: float = 0.8,
    chain_map: Optional[str] = None,
) -> Draws:
    """Draw posterior samples from a [LogDensity][] with blackjax.

    Two kernels are available. ``"rmh"`` (the default) is a Gaussian
    random-walk Metropolis kernel — it needs nothing beyond a jittable log
    density, so it works on every target, solver-backed ones included. Its
    isotropic proposal uses the per-component standard deviation
    ``2.38 / sqrt(dim)``, the optimal scaling for a unit-covariance Gaussian:
    pass ``whitening=`` so the chains run where the target really has unit
    scale, or leave it out only for targets that are already roughly unit
    scale. ``"nuts"`` additionally needs a trustworthy gradient: it is
    admitted only once the target's gradient audit
    ([LogDensity.verify_gradient][]) has passed — when no audit is stored
    yet it is run here once — and the sampler then consumes the stored
    (audited) gradient itself. NUTS step size and diagonal mass matrix come
    from blackjax window adaptation.

    Warmup: ``num_warmup`` transitions per chain are discarded for
    ``"rmh"`` (burn-in). For ``"nuts"``, ``num_warmup`` is the length of the
    window-adaptation run, executed once from the un-jittered initial point;
    its draws are likewise discarded and every chain then runs ``num_steps``
    transitions with the adapted parameters.

    Randomness is threaded deterministically from ``key``: it is split once
    into (adaptation, jitter, chains) keys; chain ``c`` starts at the initial
    point plus ``0.5 * N(0, I)`` jitter in the sampling coordinate, drawn
    from the ``c``-th split of the jitter key (overdispersed starts), and
    consumes the ``c``-th split of the chains key for its transitions. The
    same ``key`` therefore reproduces ``Draws.positions`` bit for bit, and
    every chain evolves under its own independent key stream. Each chain's
    key and start are fixed before the chains run, so every ``chain_map``
    draws the same chains.

    ``chain_map`` decides how those chains reach the target. ``"vmap"``
    evaluates them as one batched call — the fastest route whenever the target
    tolerates a batch axis. ``"sequential"`` runs them one after another, the
    per-chain program compiled once and reused, adding no axis to the target.
    ``"devices"`` splits the chains evenly across the local devices, each
    device running its own chains in sequence; on two or more devices it
    takes backends whose factors are arrays, as ``sample`` says. Leaving
    ``chain_map`` unset chooses ``"vmap"``, except for a target whose own
    solves are batched (a multiple-shooting trajectory, or a problem over
    several experiments) and either sharded over a device mesh or sampled by
    NUTS with a reverse-mode gradient: those run ``"sequential"``. A linear
    backend that holds one factorization per batch lane, such as the native
    cuDSS factor bank, serves exactly one batch axis; a single-device window
    or experiment batch folds the chain batch into that axis when the target
    is evaluated or forward-differentiated, but a sharded batch or a
    reverse-mode gradient would stack the chain batch on top of it.

    With ``whitening=`` the chains run in the whitened coordinate ``w``,
    targeting ``logdensity(transform.forward(w))`` with the initial point
    mapped through ``transform.inverse``. No Jacobian term is added: the
    whitening map is affine, so the constant ``log |det A|`` cannot change
    the target's shape — the transform is purely a sampler preconditioner.
    Positions are mapped back through ``transform.forward`` before being
    stored, so [Draws.positions][] is always in the target's own
    coordinate.

    Args:
        logdensity: The [LogDensity][] target to sample.
        key: A ``jax.random`` PRNG key.
        kernel: ``"rmh"`` (default, gradient-free) or ``"nuts"``.
        chains: Number of independent chains (default 4).
        num_steps: Post-warmup draws kept per chain.
        whitening: Optional [WhiteningTransform][] (from
            [whitening][]) preconditioning the chains.
        init: Starting point in the target's coordinate; defaults to
            ``logdensity.init``.
        num_warmup: Warmup transitions per the kernel-specific behavior
            above (default 500).
        nuts_target_acceptance: The acceptance probability NUTS adaptation
            tunes its step size toward (default 0.8). Raise it (e.g. 0.95 or
            0.98) when the report shows divergent transitions — curved
            posterior geometry needs smaller steps, paid for with more
            leapfrog evaluations per draw. Ignored by ``"rmh"``.
        chain_map: ``"vmap"``, ``"sequential"``, or ``"devices"`` as described
            above; ``None`` (the default) chooses per target.

    Returns:
        A [Draws][] container with positions of shape
        ``(chains, num_steps, dim)`` in the target's coordinate, and the
        resolved ``chain_map`` on [Draws.chain_map][].

    Raises:
        ValueError: If ``kernel`` or ``chain_map`` is unknown, a count is out
            of range, ``chain_map="devices"`` was asked for a chain count that
            is not a multiple of the local device count, or ``kernel="nuts"``
            was requested for a target whose gradient audit failed or that has
            no gradient at all — ``kernel="rmh"`` needs no gradient.
        TypeError: If ``whitening`` is not a [WhiteningTransform][].
        ConfigurationError: If ``chain_map="devices"`` runs on two or more
            local devices and an experiment of the target's problem solves
            with a backend whose factors are not arrays.
    """
    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    if kernel not in ("rmh", "nuts"):
        raise ValueError(f"sample: kernel must be 'rmh' or 'nuts'; got {kernel!r}.")
    if chain_map is not None and chain_map not in CHAIN_MAPS:
        raise ValueError(
            f"sample: chain_map must be one of {CHAIN_MAPS}, or None to choose "
            f"per target; got {chain_map!r}."
        )
    if chain_map is None:
        chain_map = _default_chain_map(logdensity, kernel)
    if chain_map == "devices":
        _refuse_shared_storage_on_devices(logdensity._problem)
    chains = int(chains)
    num_steps = int(num_steps)
    num_warmup = int(num_warmup)
    if chains < 1:
        raise ValueError(f"sample: chains must be a positive integer; got {chains}.")
    if num_steps < 1:
        raise ValueError(f"sample: num_steps must be a positive integer; got {num_steps}.")
    if num_warmup < 0 or (kernel == "nuts" and num_warmup < 1):
        raise ValueError(
            f"sample: num_warmup must be non-negative (and at least 1 for "
            f"kernel='nuts', where it sizes the adaptation run); got {num_warmup}."
        )
    nuts_target_acceptance = float(nuts_target_acceptance)
    if not 0.0 < nuts_target_acceptance < 1.0:
        raise ValueError(
            f"sample: nuts_target_acceptance must lie strictly between 0 and 1; "
            f"got {nuts_target_acceptance}."
        )
    if whitening is not None and not isinstance(whitening, WhiteningTransform):
        raise TypeError(
            f"sample: whitening must be the WhiteningTransform built by "
            f"whitening(...); got {type(whitening).__name__}."
        )

    if kernel == "nuts":
        if logdensity._grad is None:
            raise ValueError(
                "sample kernel='nuts' requires a target with a gradient; gradient=None supports kernel='rmh'."
            )
        if logdensity.gradient_check is None:
            logdensity.verify_gradient()
        if not logdensity.gradient_verified:
            check = logdensity.gradient_check
            raise ValueError(
                "sample: kernel='nuts' is admitted only after a passing gradient "
                f"audit, but the recorded audit failed (vector relative error "
                f"{check.norm_rel_error:.3e} against tolerance {check.rtol:.1e}). "
                "Fix the target's gradient before using NUTS, or use "
                "kernel='rmh', which needs no gradient."
            )

    import blackjax

    base_fn = logdensity._fn
    if kernel == "nuts":
        base_fn = _with_stored_gradient(logdensity)
    if whitening is None:
        target_fn = base_fn
        start = jnp.asarray(logdensity.init if init is None else init)
    else:
        transform = whitening

        def target_fn(w: Array) -> Array:
            return base_fn(transform.forward(w))

        start = transform.inverse(jnp.asarray(logdensity.init if init is None else init))
    dim = int(start.shape[0])

    key_adapt, key_jitter, key_chains = jax.random.split(key, 3)
    jitter = _CHAIN_JITTER * jax.random.normal(key_jitter, (chains, dim), dtype=start.dtype)
    starts = start[None, :] + jitter
    chain_keys = jax.random.split(key_chains, chains)

    if kernel == "rmh":
        scale = jnp.full((dim,), _RW_OPTIMAL_SCALE / np.sqrt(dim), dtype=start.dtype)
        algorithm = blackjax.additive_step_random_walk(
            target_fn, blackjax.mcmc.random_walk.normal(scale)
        )
        total = num_warmup + num_steps

        def run_chain(rng_key: Array, position: Array) -> tuple:
            state = algorithm.init(position)

            def step(st, k):
                st, step_info = algorithm.step(k, st)
                return st, (st.position, step_info.acceptance_rate)

            _, (pos, acceptance) = jax.lax.scan(step, state, jax.random.split(rng_key, total))
            return pos, acceptance

        pos, acceptance = _run_chains(run_chain, chain_keys, starts, chain_map)
        positions = pos[:, num_warmup:]
        info = {"acceptance_rate": acceptance[:, num_warmup:]}
    else:
        adaptation = blackjax.window_adaptation(
            blackjax.nuts, target_fn, target_acceptance_rate=nuts_target_acceptance
        )
        (_, adapted), _ = adaptation.run(key_adapt, start, num_steps=num_warmup)
        algorithm = blackjax.nuts(target_fn, **adapted)

        def run_chain(rng_key: Array, position: Array) -> tuple:
            state = algorithm.init(position)

            def step(st, k):
                st, step_info = algorithm.step(k, st)
                return st, (
                    st.position,
                    step_info.acceptance_rate,
                    step_info.is_divergent,
                    step_info.num_integration_steps,
                )

            _, collected = jax.lax.scan(step, state, jax.random.split(rng_key, num_steps))
            return collected

        pos, acceptance, divergent, integration_steps = _run_chains(
            run_chain, chain_keys, starts, chain_map
        )
        positions = pos
        info = {
            "acceptance_rate": acceptance,
            "is_divergent": divergent,
            "num_integration_steps": integration_steps,
        }

    if whitening is not None:
        positions = jax.vmap(jax.vmap(whitening.forward))(positions)

    if logdensity._to_physical is not None:
        positions = jax.vmap(jax.vmap(logdensity._to_physical))(positions)

    return Draws(
        values=positions,
        names=tuple(logdensity.names),
        info=info,
        _parameter_schema=logdensity._parameter_schema,
        _problem=logdensity._problem,
        chain_map=chain_map,
    )


# ---------------------------------------------------------------------------
# Simulation-based calibration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SbcReport:
    """The rank-uniformity evidence of one simulation-based-calibration run.

    Each replicate draws parameters from the prior, simulates data through
    the experiment's likelihoods, samples the resulting posterior, and
    records the rank of the prior draw among the kept posterior draws, per
    parameter. On a calibrated pipeline the ranks are uniform on
    ``{0, ..., n_posterior}``; the report bins them and tests uniformity
    with a per-parameter chi-square statistic.

    Attributes:
        names: Per-parameter labels.
        ranks: Rank statistics, shape ``(n_replicates, dim)``, each in
            ``{0, ..., n_posterior}``.
        n_posterior: Kept posterior draws per replicate (after thinning).
        ranks_bins: Number of equal-width rank bins.
        bin_counts: Binned rank counts, shape ``(dim, ranks_bins)``.
        chi2: Per-parameter chi-square statistic against the uniform
            expectation (``ranks_bins - 1`` degrees of freedom).
        p_value: Per-parameter chi-square p-values.
        acceptance_mean: Mean Metropolis acceptance rate over every
            replicate's kept transitions — a low value means the proposal
            scale is off and the ranks are less trustworthy.
        alpha: The p-value threshold the flags were evaluated against.
        flags: Plain-words uniformity failures, one per flagged parameter.
    """

    names: tuple
    ranks: np.ndarray
    n_posterior: int
    ranks_bins: int
    bin_counts: np.ndarray
    chi2: np.ndarray
    p_value: np.ndarray
    acceptance_mean: float
    alpha: float
    flags: tuple

    @property
    def ok(self) -> bool:
        """Whether every parameter's rank histogram passed the uniformity check."""
        return not self.flags

    def summary(self) -> str:
        """Format the per-parameter uniformity evidence as plain text."""
        width = max(9, *(len(str(n)) for n in self.names))
        lines = [
            f"SBC report: {self.ranks.shape[0]} replicates, "
            f"{self.n_posterior} kept posterior draws, {self.ranks_bins} bins",
            f"{'parameter':<{width}} {'chi2':>9} {'p':>9}",
        ]
        for j, name in enumerate(self.names):
            lines.append(f"{str(name):<{width}} {self.chi2[j]:>9.2f} {self.p_value[j]:>9.4f}")
        lines.append(f"mean acceptance: {self.acceptance_mean:.3f}")
        if self.flags:
            lines.append("flags:")
            lines.extend(f"  - {flag}" for flag in self.flags)
        else:
            lines.append("flags: none")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the compact human-readable summary."""
        return self.summary()


def sbc(
    model_builder: Callable,
    *,
    key: Array,
    n_replicates: int,
    ranks_bins: int = 20,
    num_steps: int = 495,
    num_warmup: int = 200,
    thin: int = 5,
    proposal_scale: Any = None,
    alpha: float = 0.005,
) -> SbcReport:
    """Run simulation-based calibration of a model, prior, and sampler.

    For each replicate: draw parameters from the prior, simulate one
    synthetic data set through the experiment's own channel likelihoods,
    sample the synthetic posterior with a Gaussian random-walk Metropolis
    chain warm-started at the prior draw, and record the rank of the prior
    draw among the kept (thinned) posterior draws, per parameter. Ranks are
    computed in the sampling coordinate; elementwise monotone coordinate
    maps preserve them. On a calibrated pipeline the ranks are uniform;
    the report tests each parameter's binned ranks with a chi-square
    statistic and flags departures at ``alpha``.

    The replicate sampler is compiled once and reused with the synthetic
    data as an argument, so solver-backed models pay one compilation for
    the whole run, not one per replicate. blackjax is not needed here.

    Args:
        model_builder: A zero-argument callable returning a mapping with
            keys ``"experiment"`` (the design-stage
            [Experiment][diff_soup.inference.statistics.measurement.Experiment] whose
            response channels define the observation models — no attached
            data needed) and ``"prior"`` (a proper
            [Prior][diff_soup.inference.statistics.priors.Prior] on the physical
            coordinate). Optional keys: ``"parameters"`` (parameter groups),
            ``"parameterization"`` (a
            [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]
            — chains then run in the inference coordinate with the exact
            prior pullback), and ``"generator"`` (an experiment whose
            channels generate the synthetic data instead; scoring still uses
            ``"experiment"``, so a deliberate mismatch measures how the
            calibration check responds to misspecification).
        key: A ``jax.random`` PRNG key; the same key reproduces the ranks
            bit for bit.
        n_replicates: Number of prior-draw/refit replicates.
        ranks_bins: Number of equal-width rank bins for the chi-square
            test. The ``num_steps / thin + 1`` possible rank values must
            divide evenly into the bins.
        num_steps: Post-warmup Metropolis transitions per replicate; must be
            divisible by ``thin``.
        num_warmup: Discarded warmup transitions per replicate.
        thin: Keep every ``thin``-th post-warmup draw for the rank
            statistic (thinning reduces the autocorrelation that distorts
            finite-chain ranks).
        proposal_scale: Per-parameter random-walk proposal standard
            deviations in the sampling coordinate. Default: the prior's
            marginal standard deviations scaled by ``2.38 / sqrt(dim)`` — a
            reasonable start, but posterior-scale values mix better; check
            ``acceptance_mean`` on the report.
        alpha: Chi-square p-value threshold below which a parameter's rank
            histogram is flagged as non-uniform.

    Returns:
        The [SbcReport][].

    Raises:
        ValueError: If the counts are inconsistent (non-positive replicate
            count, ``num_steps`` not divisible by ``thin``, rank outcomes
            that do not divide into ``ranks_bins``), the builder's mapping
            is missing ``"experiment"``/``"prior"``, the prior is improper
            for this use, or the generator does not declare every scoring
            response channel.
    """
    from scipy.stats import chi2 as chi2_dist

    n_replicates = int(n_replicates)
    ranks_bins = int(ranks_bins)
    num_steps = int(num_steps)
    num_warmup = int(num_warmup)
    thin = int(thin)
    if n_replicates < 1:
        raise ValueError(f"sbc: n_replicates must be a positive integer; got {n_replicates}.")
    if thin < 1 or num_steps < thin or num_steps % thin != 0:
        raise ValueError(
            f"sbc: num_steps ({num_steps}) must be a positive multiple of thin ({thin})."
        )
    if num_warmup < 0:
        raise ValueError(f"sbc: num_warmup must be non-negative; got {num_warmup}.")
    kept = num_steps // thin
    outcomes = kept + 1
    if ranks_bins < 2 or outcomes % ranks_bins != 0:
        raise ValueError(
            f"sbc: the rank statistic takes {outcomes} possible values "
            f"(num_steps / thin + 1), which must divide evenly into ranks_bins "
            f"(got {ranks_bins}). Adjust num_steps, thin, or ranks_bins — e.g. "
            f"num_steps = (k * ranks_bins - 1) * thin for an integer k."
        )

    bundle = model_builder()
    if not hasattr(bundle, "get") or "experiment" not in bundle or "prior" not in bundle:
        raise ValueError(
            "sbc: model_builder() must return a mapping with 'experiment' (the "
            "design-stage experiment) and 'prior' (a proper prior on the "
            "physical coordinate); optional keys are 'parameters', "
            "'parameterization', and 'generator'."
        )
    experiment = bundle["experiment"]
    prior = bundle["prior"]
    parameters = bundle.get("parameters")
    parameterization = bundle.get("parameterization")
    generator = bundle.get("generator", experiment)
    if getattr(prior, "coordinate", "x") != "x":
        raise ValueError(
            "sbc: the prior must be declared on the physical coordinate 'x'; "
            f"got coordinate {prior.coordinate!r}."
        )

    predictor = experiment.predictor(parameters=parameters)
    names = tuple(predictor.theta_names)
    dim = len(names)
    prior_size = getattr(prior, "size", None)
    if prior_size is not None and int(prior_size) != dim:
        raise ValueError(
            f"sbc: the prior covers {int(prior_size)} scalars but the experiment "
            f"has {dim} learnable parameters ({names})."
        )
    response = tuple(c for c in experiment.channels if c.role == "response")
    if not response:
        raise ValueError("sbc: the experiment declares no response channel to simulate.")
    gen_by_name = {c.name: c for c in generator.channels if c.role == "response"}
    missing = sorted(set(c.name for c in response) - set(gen_by_name))
    if missing:
        raise ValueError(
            f"sbc: the generator experiment does not declare response channel(s) "
            f"{missing}; it must generate data for every scoring channel."
        )
    gen_predictor = (
        predictor if generator is experiment else generator.predictor(parameters=parameters)
    )

    if parameterization is not None and parameterization.n_parameters != dim:
        raise ValueError(
            f"sbc: the parameterization covers {parameterization.n_parameters} "
            f"scalars but the experiment has {dim} learnable parameters ({names})."
        )
    if parameterization is not None:
        names = tuple(parameterization.names())

    if proposal_scale is None:
        precision = np.asarray(prior.precision(), dtype=float)
        lower = np.linalg.cholesky(precision)
        linv = np.linalg.solve(lower, np.eye(dim))
        marginal_sd = np.sqrt(np.sum(linv**2, axis=0))
        proposal_scale = _RW_OPTIMAL_SCALE / np.sqrt(dim) * marginal_sd
    scale = jnp.asarray(np.broadcast_to(np.asarray(proposal_scale, dtype=float), (dim,)))

    def target(z: Array, data: tuple) -> Array:
        x = z if parameterization is None else parameterization.transform(z)
        preds, _ = predictor(x)
        total = jnp.zeros(())
        for c, observed in zip(response, data):
            likelihood, context = _channel_likelihood_view(c, preds[c.name])
            total = total + likelihood.log_prob(
                observed,
                preds[c.name],
                context=context,
            )
        total = total + prior.log_prob(x)
        if parameterization is not None:
            total = total + jnp.sum(parameterization.log_abs_det_jacobian(z))
        return total

    total_steps = num_warmup + num_steps

    @jax.jit
    def run_replicate(z0: Array, data: tuple, chain_key: Array) -> tuple:
        def step(carry, k):
            z, logp = carry
            k_prop, k_accept = jax.random.split(k)
            proposal = z + scale * jax.random.normal(k_prop, z.shape, dtype=z.dtype)
            logp_prop = target(proposal, data)
            accept = jnp.log(jax.random.uniform(k_accept)) < (logp_prop - logp)
            z_new = jnp.where(accept, proposal, z)
            logp_new = jnp.where(accept, logp_prop, logp)
            return (z_new, logp_new), (z_new, accept)

        keys = jax.random.split(chain_key, total_steps)
        _, (positions, accepted) = jax.lax.scan(step, (z0, target(z0, data)), keys)
        kept_positions = positions[num_warmup:][thin - 1 :: thin]
        rank = jnp.sum(kept_positions < z0[None, :], axis=0)
        return rank, jnp.mean(accepted[num_warmup:])

    replicate_keys = jax.random.split(key, n_replicates)
    ranks = np.empty((n_replicates, dim), dtype=int)
    acceptance = np.empty(n_replicates)
    for r in range(n_replicates):
        k_prior, k_sim, k_chain = jax.random.split(replicate_keys[r], 3)
        theta = prior.sample(k_prior)
        gen_preds, _ = gen_predictor(theta)
        sim_keys = jax.random.split(k_sim, len(response))
        generated = []
        for c, k_c in zip(response, sim_keys):
            generator_channel = gen_by_name[c.name]
            likelihood, context = _channel_likelihood_view(generator_channel, gen_preds[c.name])
            generated.append(likelihood.sample(k_c, gen_preds[c.name], context=context))
        data = tuple(generated)
        z0 = theta if parameterization is None else parameterization.inverse(theta)
        rank, acc = run_replicate(jnp.asarray(z0), data, k_chain)
        ranks[r] = np.asarray(rank)
        acceptance[r] = float(acc)

    bin_width = outcomes // ranks_bins
    expected = n_replicates / ranks_bins
    bin_counts = np.zeros((dim, ranks_bins), dtype=int)
    chi2 = np.empty(dim)
    p_value = np.empty(dim)
    flags: list = []
    for j in range(dim):
        counts = np.bincount(ranks[:, j] // bin_width, minlength=ranks_bins)
        bin_counts[j] = counts
        chi2[j] = float(np.sum((counts - expected) ** 2 / expected))
        p_value[j] = float(chi2_dist.sf(chi2[j], ranks_bins - 1))
        if p_value[j] < alpha:
            flags.append(
                f"parameter {names[j]!r}: rank histogram departs from uniformity "
                f"(chi-square p = {p_value[j]:.2e} < {alpha:g}) — the model, "
                "prior, and sampler are not jointly calibrated"
            )

    return SbcReport(
        names=names,
        ranks=ranks,
        n_posterior=kept,
        ranks_bins=ranks_bins,
        bin_counts=bin_counts,
        chi2=chi2,
        p_value=p_value,
        acceptance_mean=float(np.mean(acceptance)),
        alpha=float(alpha),
        flags=tuple(flags),
    )
