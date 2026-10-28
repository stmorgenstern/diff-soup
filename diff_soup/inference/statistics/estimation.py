"""MLE/MAP estimation problems lowered onto the optimization engines."""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._precision import cast_floating_leaves
from diff_soup.inference.fitting._layout import resolve_point
from diff_soup.inference.fitting.inference_problem import _MemoryCheckedHessian
from diff_soup.inference.fitting.regularization import NoReg
from diff_soup.inference.fitting.result import OptResult
from diff_soup.inference.solvers.tangent import value_and_jacfwd
from diff_soup.inference.statistics.likelihoods import (
    CorrelatedGaussian,
    Gaussian,
    LogNormal,
    MultiplicativeGaussian,
    Poisson,
    StudentT,
)
from diff_soup.inference.statistics.measurement import (
    measurements_without_likelihood,
    require_likelihoods,
)
from diff_soup.inference.statistics.priors import compile_prior
from diff_soup.inference.statistics.study import Study
from diff_soup.utils.ad_guards import reverse_linearization_observed


@dataclasses.dataclass(frozen=True)
class MLE:
    """The maximum-likelihood objective, which minimizes the negative log likelihood of the data.

    Pass it as `objective=` of an [`InferenceProblem`][diff_soup.inference.InferenceProblem],
    where it is the default.
    """


@dataclasses.dataclass(frozen=True)
class MAP:
    """The maximum-a-posteriori objective, which minimizes the negative log likelihood of the data plus the negative log prior.

    Pass it as `objective=` of an [`InferenceProblem`][diff_soup.inference.InferenceProblem].
    Every fitted handle needs a prior, and every measurement a likelihood.

    Args:
        state_prior: A [`Gaussian`][diff_soup.inference.Gaussian] with one positive scale, the
            prior on the window starts of a multiple-shooting fit, or `None`.
    """

    state_prior: Any = None

    def __post_init__(self) -> None:
        """Reject a state prior that is not a Gaussian with a positive scale."""
        if self.state_prior is None:
            return
        from diff_soup.inference.statistics.likelihoods import Gaussian

        if not isinstance(self.state_prior, Gaussian):
            raise TypeError(
                "MAP state_prior must be a Gaussian(sigma) or None; "
                f"got {type(self.state_prior).__name__}."
            )
        scale = np.asarray(self.state_prior.scale)
        if scale.size != 1 or not float(scale.reshape(())) > 0.0:
            raise ValueError(
                "MAP state_prior needs one positive scale for every window state; "
                f"got {self.state_prior.scale!r}."
            )


def _data_residual(problem: Any):
    """Select the existing cached residual closure for one leaf or joint."""
    batched = getattr(problem, "batched_r_fn", None)
    if batched is not None:
        return batched
    stacked = getattr(problem, "stacked_r_fn", None)
    if stacked is not None:
        return stacked
    segment_fns = tuple(problem.segment_fns)
    return jax.jit(lambda x: jnp.concatenate(tuple(fn(x) for fn in segment_fns)))


def _measurement_log_normalizer(study: Study, holders: tuple | None = None) -> float:
    """Return each supported family's saturated-model log likelihood.

    ``holders`` names, per experiment, the processes holding its observed data
    under a multi-process mesh: each holder contributes its share of that
    experiment's constant and the shares are summed across processes, so a
    measurement whose values are deferred is read only where it was loaded.
    """
    total = 0.0
    process = jax.process_index()
    for index, experiment in enumerate(study.experiments):
        share = 1.0
        if holders is not None:
            if process not in holders[index]:
                continue
            share = 1.0 / len(holders[index])
        for measurement in experiment.measurements:
            likelihood = measurement.likelihood
            supported = (
                Gaussian,
                CorrelatedGaussian,
                StudentT,
                Poisson,
                LogNormal,
                MultiplicativeGaussian,
            )
            if not isinstance(likelihood, supported):
                raise ValueError(
                    "InferenceProblem accepts Gaussian, CorrelatedGaussian, StudentT, Poisson, "
                    "LogNormal, and MultiplicativeGaussian likelihoods; "
                    f"measurement {measurement.name!r} uses {type(likelihood).__name__}."
                )
            values = np.asarray(experiment.data[measurement.name])
            if measurement.mask is not None:
                keep = np.asarray(measurement.mask, dtype=bool)
                values = values[keep]
                likelihood, _ = likelihood.for_frames(
                    jnp.asarray(np.flatnonzero(keep)),
                    full_prediction_shape=tuple(np.asarray(measurement.values).shape),
                    context=None,
                )
            saturated = values
            if isinstance(likelihood, LogNormal):
                saturated = values * np.exp(0.5 * np.asarray(likelihood.scale) ** 2)
            elif isinstance(likelihood, MultiplicativeGaussian):
                saturated = 2 * values / (1 + np.sqrt(1 + 4 * np.asarray(likelihood.scale) ** 2))
            elif isinstance(likelihood, Poisson):
                saturated = values / np.asarray(likelihood.exposure)
            total += share * float(likelihood.log_prob(jnp.asarray(values), jnp.asarray(saturated)))
    if holders is not None and jax.process_count() > 1:
        from diff_soup._jax_compat import multihost_utils

        total = float(np.sum(multihost_utils.process_allgather(np.asarray([total]))))
    return total


def _compile_prior_blocks(schema: Any, objective: Any) -> tuple:
    """Compile each handle occurrence exactly once for a physical MAP."""
    if isinstance(objective, MLE):
        return ()
    blocks = []
    missing = []
    for entry in schema.entries:
        declaration = getattr(entry.handle, "prior", None)
        if declaration is None:
            missing.append(entry.handle.name)
            continue
        prior = compile_prior(declaration, entry.handle, entry.size)
        for occurrence in entry.occurrences:
            blocks.append((jnp.asarray(occurrence.columns, dtype=jnp.int32), prior))
    if missing:
        raise ValueError(
            f"MAP requires a proper prior on every fitted handle; missing priors for "
            f"{sorted(missing)}. Declare priors on spec.param/spec.function or use MLE()."
        )
    return tuple(blocks)


def _resolve_handle_regularization(regularization: Any, schema: Any) -> Any:
    """Resolve a ``Tikhonov(on=handle)`` to the handle's column blocks; pass others through."""
    handle = getattr(regularization, "on", None)
    if handle is None:
        return regularization
    try:
        entry = schema.entry(handle)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Tikhonov(on=handle) names a handle this problem does not fit; the fitted "
            f"handles are {[getattr(h, 'name', h) for h in schema.handles]}."
        ) from exc
    return regularization.on_columns(
        tuple(tuple(int(c) for c in occurrence.columns) for occurrence in entry.occurrences)
    )


@dataclasses.dataclass(frozen=True)
class _StructuredPriorRegularizer:
    """Expose independent MAP prior rows with their exact column ownership."""

    blocks: tuple
    extra: Any = dataclasses.field(default_factory=NoReg)

    def residual(self, x: Any) -> Any:
        """Concatenate the same physical-prior rows used by dense LM."""
        x = jnp.asarray(x)
        parts = (self.extra.residual(x),) + tuple(
            prior.residual(x[columns]) for columns, prior in self.blocks
        )
        return jnp.concatenate(parts)

    def penalty(self, x: Any) -> Any:
        """Return half the squared norm of all prior residual rows."""
        return self.extra.penalty(x) + sum(
            0.5 * jnp.sum(prior.residual(x[columns]) ** 2) for columns, prior in self.blocks
        )

    def block_rows(self, x: Any) -> tuple:
        """Return column-local ``(columns, residual, jacobian)`` triples of every prior row."""
        x = jnp.asarray(x)
        result = []
        if self.extra.residual(x).size:
            provider = getattr(self.extra, "block_rows", None)
            if not callable(provider):
                raise ValueError(
                    f"regularization {type(self.extra).__name__} has no block_rows(x); a fit "
                    "over the joint problem takes a regularization that defines "
                    "block_rows(x) returning (columns, residual, jacobian) triples."
                )
            result.extend(provider(x))
        for columns, prior in self.blocks:
            values = x[columns]
            residual = prior.residual(values)
            jacobian = jax.jacfwd(prior.residual)(values)
            result.append((tuple(int(item) for item in np.asarray(columns)), residual, jacobian))
        return tuple(result)


@dataclasses.dataclass(frozen=True, init=False)
class EstimationProblem:
    """A Study paired with MLE or physical-space MAP, consumable by ``optsolve``.

    Exact likelihood residuals and declared priors share the same optimization
    coordinate, parameter schema, and normalized density for later analyses.
    """

    study: Study
    objective: MLE | MAP
    parameter_schema: Any
    _inference_problem: Any
    _x0: np.ndarray
    _data_r: Any
    _data_residual_jacobian: Any
    _observed_information: Any
    _prior_precision: Any
    _residual: Any
    _residual_jacobian: Any
    _prior_blocks: tuple
    _log_normalizer: float | None
    _loss_fn: Any
    _loss_grad_fns: dict
    _loss_grad_hessian_fns: dict
    regularization: NoReg
    trial_predicate: Any

    def __init__(
        self,
        study: Study,
        objective: MLE | MAP,
        *,
        start: Any = None,
        layout: Any = None,
        regularization: Any = None,
        trial_predicate: Any = None,
    ):
        """Compile the declaration, parameter schema, prior rows, and warm start.

        Args:
            study: The declared :class:`~diff_soup.inference.Study`.
            objective: :class:`MLE` or physical-space :class:`MAP`.
            regularization: Optional parameter regularization. A
                ``Tikhonov(lambda_=..., on=handle)`` is resolved here to the
                handle's columns, once per owner when the handle varies.
            trial_predicate: Optional callable ``x -> bool`` (JAX-traceable)
                evaluated on every optimizer trial point before any residual or
                PDE solve; a ``False`` rejects the trial without evaluating it.
            start: Optional warm start: a flat vector, a mapping from every fitted
                handle to its value, or an
                :class:`~diff_soup.inference.fitting.result.OptResult` of the same
                parameter layout.
            layout: Optional device placement for the Study's child batch,
                forwarded to :meth:`Study.compile_inference_problem`.
        """
        if not isinstance(study, Study):
            raise TypeError("EstimationProblem first argument must be a Study.")
        if not isinstance(objective, (MLE, MAP)):
            raise TypeError("EstimationProblem objective must be MLE() or MAP().")
        if isinstance(objective, MAP):
            require_likelihoods(study, "MAP")
        likelihood_free = bool(measurements_without_likelihood(study))
        problem = study.compile_inference_problem(
            layout=layout, state_prior=getattr(objective, "state_prior", None)
        )
        holders = getattr(problem, "data_holders", None) if jax.process_count() > 1 else None
        schema = study.parameter_schema(problem)
        regularization = _resolve_handle_regularization(regularization, schema)
        if trial_predicate is not None and not callable(trial_predicate):
            raise TypeError("InferenceProblem trial_predicate must be callable or None.")
        data_r = _data_residual(problem)
        # The data problem's decision vector carries the problem's precision;
        # the prior rows run at the same dtype so the objective stays in it.
        dtype = np.asarray(problem.x0).dtype
        prior_blocks = tuple(
            (columns, cast_floating_leaves(prior, dtype))
            for columns, prior in _compile_prior_blocks(schema, objective)
        )

        def residual(x):
            parts = [data_r(x)]
            parts.extend(prior.residual(x[columns]) for columns, prior in prior_blocks)
            return jnp.concatenate(parts)

        residual = jax.jit(residual)
        residual_jacobian = jax.jit(lambda x: value_and_jacfwd(residual, x))
        data_residual_jacobian = jax.jit(lambda x: value_and_jacfwd(data_r, x))
        data_loss_grad_hessian = problem.loss_grad_hessian_fn("auto")

        # Memory-checked once here, so the data Hessian's own check skips itself and the
        # Hessian compiles once.
        observed_information = _MemoryCheckedHessian(
            lambda x: data_loss_grad_hessian(x)[2], tile="full"
        )

        def negative_log_prior(x):
            total = jnp.zeros((), dtype=x.dtype)
            for columns, prior in prior_blocks:
                total = total - prior.log_prob(x[columns])
            return total

        prior_precision = jax.jit(jax.jacfwd(jax.jacfwd(negative_log_prior)))

        x0 = np.asarray(problem.x0)
        if start is not None:
            view = types.SimpleNamespace(parameter_schema=schema, x0=x0)
            x0 = np.asarray(resolve_point(view, start, where="EstimationProblem start"))
            x0 = x0.astype(dtype)
            if x0.shape != np.asarray(problem.x0).shape:
                raise ValueError(
                    f"EstimationProblem start has shape {x0.shape}; expected "
                    f"{np.asarray(problem.x0).shape}."
                )

        object.__setattr__(self, "study", study)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "parameter_schema", schema)
        object.__setattr__(self, "_inference_problem", problem)
        object.__setattr__(self, "_x0", x0)
        object.__setattr__(self, "_data_r", data_r)
        object.__setattr__(self, "_data_residual_jacobian", data_residual_jacobian)
        object.__setattr__(self, "_observed_information", observed_information)
        object.__setattr__(self, "_prior_precision", prior_precision)
        object.__setattr__(self, "_residual", residual)
        object.__setattr__(self, "_residual_jacobian", residual_jacobian)
        object.__setattr__(self, "_prior_blocks", prior_blocks)
        object.__setattr__(
            self,
            "_log_normalizer",
            None if likelihood_free else _measurement_log_normalizer(study, holders),
        )
        object.__setattr__(self, "_loss_fn", None)
        object.__setattr__(self, "_loss_grad_fns", {})
        object.__setattr__(self, "_loss_grad_hessian_fns", {})
        object.__setattr__(
            self, "regularization", regularization if regularization is not None else NoReg()
        )
        object.__setattr__(self, "trial_predicate", trial_predicate)

    def trial_is_valid(self, x: Any):
        """Whether ``x`` passes the declared trial predicate and the data problem's own checks."""
        point = jnp.asarray(x)
        valid = jnp.asarray(True)
        if self.trial_predicate is not None:
            valid = valid & jnp.asarray(self.trial_predicate(point), dtype=bool)
        inner = getattr(self._inference_problem, "trial_is_valid", None)
        if inner is not None:
            valid = valid & jnp.asarray(inner(point), dtype=bool)
        return valid

    def _prior_objective(self, x: Any) -> Any:
        """Return the MAP prior penalty represented by the compiled prior rows."""
        x = jnp.asarray(x)
        total = jnp.zeros((), dtype=x.dtype)
        for columns, prior in self._prior_blocks:
            rows = prior.residual(x[columns])
            total = total + 0.5 * jnp.sum(rows * rows)
        return total

    @property
    def x0(self) -> np.ndarray:
        """Initial physical parameter vector, optionally supplied by a prior fit."""
        return self._x0

    @property
    def n_params(self) -> int:
        """Number of scalar decision variables."""
        return int(self._x0.size)

    @property
    def n_segments(self) -> int:
        """Underlying PDE residual-block count for evaluation accounting."""
        return int(getattr(self._inference_problem, "n_segments", 1))

    @property
    def segment_fns(self) -> tuple:
        """One exact data-plus-prior residual consumed by both native optimizers."""
        return (self._residual,)

    @property
    def stacked_r_fn(self):
        """Cached combined residual closure."""
        return self._residual

    @property
    def stacked_rj_fn(self):
        """Cached combined residual/Jacobian closure."""
        return self._residual_jacobian

    @property
    def batched_r_fn(self):
        """Combined prior rows make the public residual a single stacked block."""
        return None

    @property
    def batched_rj_fn(self):
        """Combined prior rows make the public Jacobian a single stacked block."""
        return None

    @property
    def loss_fn(self):
        """Cached data-plus-prior scalar objective."""
        if self._loss_fn is None:
            data_loss = self._inference_problem.loss_fn

            @jax.jit
            def loss(x):
                return data_loss(x) + self._prior_objective(x)

            object.__setattr__(self, "_loss_fn", loss)
        return self._loss_fn

    def loss_and_grad_fn(self, ad: str = "reverse"):
        """Return the cached data-plus-prior objective and gradient seam."""
        cached = self._loss_grad_fns.get(ad)
        if cached is None:
            data_loss_and_grad = self._inference_problem.loss_and_grad_fn(ad)

            @jax.jit
            def loss_and_grad(x):
                loss, gradient = data_loss_and_grad(x)
                prior_loss, prior_gradient = jax.value_and_grad(self._prior_objective)(x)
                return loss + prior_loss, gradient + prior_gradient

            cached = loss_and_grad
            self._loss_grad_fns[ad] = cached
        return cached

    @property
    def supported_compositions(self) -> frozenset[str]:
        """Differentiation modes supported by the compiled inference problem."""
        return self._inference_problem.supported_compositions

    def resolved_hessian_composition(self, composition: str = "auto") -> str:
        """Resolve a true-Hessian composition on the compiled inference problem."""
        return self._inference_problem.resolved_hessian_composition(composition)

    def loss_grad_hessian_fn(self, composition: str = "auto"):
        """Return the cached data-plus-prior loss, gradient, and Hessian seam."""
        resolved = self.resolved_hessian_composition(composition)
        cached = self._loss_grad_hessian_fns.get(resolved)
        if cached is None:
            data_loss_grad_hessian = self._inference_problem.loss_grad_hessian_fn(resolved)

            @jax.jit
            def loss_grad_hessian(x):
                loss, gradient, hessian = data_loss_grad_hessian(x)
                prior_loss, prior_gradient = jax.value_and_grad(self._prior_objective)(x)
                prior_hessian = jax.hessian(self._prior_objective)(x)
                return (
                    loss + prior_loss,
                    gradient + prior_gradient,
                    hessian + prior_hessian,
                )

            # One memory check, here, compiles the Hessian once: the data Hessian's own
            # check skips itself inside this one.
            cached = _MemoryCheckedHessian(loss_grad_hessian, tile="full")
            self._loss_grad_hessian_fns[resolved] = cached
        return cached

    @property
    def unflatten_fn(self):
        """Underlying physical parameter reconstruction, when one exists."""
        return getattr(self._inference_problem, "unflatten_fn", None)

    @property
    def prob_static(self):
        """Underlying frozen model partition, when one exists."""
        return getattr(self._inference_problem, "prob_static", None)

    @property
    def theta_names(self) -> tuple:
        """Ordered scalar labels used by optimizer diagnostics."""
        return tuple(getattr(self._inference_problem, "theta_names", ()))

    @property
    def param_group_names(self) -> tuple:
        """Parameter-group labels aligned with the flat schema."""
        return tuple(getattr(self._inference_problem, "param_group_names", ()))

    @property
    def param_group_cols(self) -> tuple:
        """Flat columns owned by each parameter group."""
        return tuple(getattr(self._inference_problem, "param_group_cols", ()))

    @property
    def bounds(self):
        """Explicit bounds of the underlying problem, if any."""
        return getattr(self._inference_problem, "bounds", None)

    @property
    def _declared_bounds(self):
        """Recipe-declared leaf bounds used when no explicit Bounds object exists."""
        return getattr(self._inference_problem, "_declared_bounds", None)

    @property
    def is_sharded(self) -> bool:
        """Whether the underlying Study lowering uses verified device sharding."""
        return bool(getattr(self._inference_problem, "is_sharded", False))

    @property
    def diagnostic_problem(self) -> Any:
        """Data-only problem for practical-identifiability diagnostics."""
        return self._inference_problem

    def joint_view(self) -> Any:
        """The joint problem with its level structure, warm-started at this problem's start.

        For a study of several experiments this is the underlying
        [JointInferenceProblem][diff_soup.inference.fitting.joint.JointInferenceProblem]
        at this problem's start. Under MLE that is the data problem; under
        MAP, or with a regularization, it also carries the independent prior
        rows with their global/owner-local column declarations, so a fit on
        the levels adds their Gauss--Newton statistics exactly once; a trial
        predicate travels with it. A study of one experiment has no joint
        problem: its view is a copy of this problem whose regularizer carries
        the prior rows with their columns, and whose ``n_residuals`` counts
        the data rows.
        """
        structured = isinstance(self.objective, MAP) or not isinstance(self.regularization, NoReg)
        with_start = getattr(self._inference_problem, "with_start", None)
        if not callable(with_start):
            import copy

            view = copy.copy(self)
            object.__setattr__(view, "n_residuals", int(self._inference_problem.residual_rows))
            if structured:
                regularizer = _StructuredPriorRegularizer(self._prior_blocks, self.regularization)
                object.__setattr__(view, "regularization", regularizer)
            return view
        problem = with_start(self._x0)
        if structured:
            regularizer = _StructuredPriorRegularizer(self._prior_blocks, self.regularization)
            problem = problem.with_regularization(regularizer)
        if self.trial_predicate is not None:
            problem = problem.with_trial_predicate(self.trial_predicate)
        return problem

    def numeric_problem(self, optimizer: Any) -> Any:
        """The [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem] ``optimizer`` solves.

        Its residual is the whitened data rows followed by the MAP prior
        rows; its value and gradient, exact Hessian and Gauss-Newton
        statistics are this problem's (``loss_fn``,
        ``loss_and_grad_fn(optimizer.ad)``,
        ``loss_grad_hessian_fn(optimizer.composition)``), evaluated on first
        use. A problem sharded across a device mesh gives the Gauss-Newton
        statistics reduced across the mesh. The bounds are resolved once;
        ``regularization`` is this problem's, or, for a solver that requires
        ``"batched_lanes"``, the prior rows with their columns.
        """
        return lower_estimation(self, self, optimizer)

    def count_failed(self, x_flat: Any) -> tuple[int, tuple[int, ...]]:
        """Delegate sentinel failure detection to the underlying PDE problem."""
        count = getattr(self._inference_problem, "count_failed", None)
        return (0, ()) if not callable(count) else count(x_flat)

    @property
    def child_results(self):
        """Expose joint child reconstruction only for a multi-experiment Study."""
        value = getattr(self._inference_problem, "child_results", None)
        if not callable(value):
            raise AttributeError("child_results")
        return value

    @property
    def shared_values(self):
        """Expose joint shared reconstruction only for a multi-experiment Study."""
        value = getattr(self._inference_problem, "shared_values", None)
        if not callable(value):
            raise AttributeError("shared_values")
        return value

    def log_likelihood(self, x: Any) -> Any:
        """Normalized response log likelihood at physical parameters.

        Differentiable in both modes. Evaluation and forward-mode derivatives
        (``jax.jvp``, ``jax.jacfwd``) run the forward-mode solve seam; a
        reverse-mode derivative (``jax.grad``, ``jax.vjp``) taken of this call
        runs the reverse-capable seam, whose solves use the solver's
        configured reverse adjoint (``CheckpointAdjoint`` unless the solver
        declares another). The choice is read from the differentiation visible
        on ``x`` when the call is traced, so take the reverse derivative of a
        function that calls this method, not of a ``jax.jit`` wrapper around
        it.
        """
        if self._log_normalizer is None:
            require_likelihoods(self.study, "log_likelihood")
        x = jnp.asarray(x)
        if reverse_linearization_observed(x):
            reverse_loss = getattr(self._inference_problem, "loss_fn", None)
            if reverse_loss is not None:
                return -reverse_loss(x) + self._log_normalizer
        residual = self._data_r(x)
        return -0.5 * jnp.sum(residual**2) + self._log_normalizer

    def data_residual_jacobian(self, x: Any) -> tuple[Any, Any]:
        """Return whitened data residuals and their cached forward Jacobian."""
        return self._data_residual_jacobian(jnp.asarray(x))

    def observed_information(self, x: Any) -> Any:
        """Return the observed NLL Hessian from the inference-problem seam."""
        return self._observed_information(jnp.asarray(x))

    def prior_precision(self, x: Any) -> Any:
        """Return local physical-coordinate prior curvature (zero for MLE)."""
        return self._prior_precision(jnp.asarray(x))

    def log_prior(self, x: Any) -> Any:
        """Normalized physical-coordinate prior log density for a MAP problem."""
        x = jnp.asarray(x, dtype=self._x0.dtype)
        total = jnp.zeros((), dtype=x.dtype)
        for columns, prior in self._prior_blocks:
            total = total + prior.log_prob(x[columns])
        return total

    def finalize_opt_result(self, result: OptResult, *, optimizer: Any) -> OptResult:
        """Attach handle-based schema and normalized statistical provenance."""
        del optimizer
        kind = type(self.objective).__name__
        log_prior = None
        if isinstance(self.objective, MAP):
            log_prior = float(self.log_prior(result.params))
        statistical = self._log_normalizer is not None
        info = dict(result.info)
        info["uq"] = {
            "study_fingerprint": self.study.fingerprint,
            "objective": kind,
            "coordinate": "physical",
            "likelihood": (
                "normalized measurement likelihood"
                if statistical
                else "weighted least squares; no measurement declares a noise model"
            ),
        }
        return dataclasses.replace(
            result,
            parameter_schema=self.parameter_schema,
            objective=self.objective,
            log_likelihood=float(self.log_likelihood(result.params)) if statistical else None,
            log_prior=log_prior,
            info=info,
        )


def _reduced_cost(problem: EstimationProblem) -> Any:
    """``x -> `` the reduced cost plus the MAP prior rows, NaN where a child failed."""
    evaluate_cost = problem._inference_problem.evaluate_cost_fn

    def value(x):
        x = jnp.asarray(x)
        cost, _rows, failed = evaluate_cost(x)
        return jnp.where(failed == 0, cost + problem._prior_objective(x), jnp.nan)

    return value


def _reduced_gauss_newton(problem: EstimationProblem) -> Any:
    """``x -> (cost, J^T r, J^T J, failed)`` reduced across the mesh, with the MAP prior rows."""
    evaluate = problem._inference_problem.evaluate_lsq_fn
    blocks = problem._prior_blocks

    def prior_rows(x):
        return jnp.concatenate(
            [jnp.zeros((0,), dtype=x.dtype)] + [prior.residual(x[c]) for c, prior in blocks]
        )

    def gauss_newton(x):
        x = jnp.asarray(x)
        cost, gradient, matrix, _rows, failed = evaluate(x)
        if blocks:
            rows = prior_rows(x)
            jacobian = jax.jacfwd(prior_rows)(x)
            cost = cost + 0.5 * jnp.sum(rows * rows)
            gradient = gradient + jacobian.T @ rows
            matrix = matrix + jacobian.T @ jacobian
        return cost, gradient, matrix, failed

    return gauss_newton


def lower_estimation(problem: Any, evaluator: Any, optimizer: Any, *, lanes: Any = None) -> Any:
    """The [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem] of an estimation problem.

    Args:
        problem: The [EstimationProblem][diff_soup.inference.statistics.estimation.EstimationProblem]
            whose residual (data rows, then MAP prior rows), bounds,
            regularization and reconstruction the result carries.
        evaluator: The problem whose ``loss_fn``, ``loss_and_grad_fn``,
            ``loss_grad_hessian_fn`` and ``gauss_newton_fn`` evaluate the
            objective: ``problem`` itself, or a lowered study's copy for one
            fit.
        optimizer: The solver; its ``requires``, ``ad`` and ``composition``
            choose the derivatives. For a solver that requires
            ``"batched_lanes"`` the regularization carries the prior rows with
            their columns.
        lanes: The lowered study that evaluates the objective on its lanes,
            or ``None``.
    """
    from diff_soup.inference.fitting.inference_problem import lower_residual_problem

    gauss_newton = value = None
    if lanes is not None:
        gauss_newton = getattr(evaluator, "gauss_newton_fn", None)
    elif problem.is_sharded and getattr(problem._inference_problem, "evaluate_lsq_fn", None):
        gauss_newton = _reduced_gauss_newton(problem)
        if getattr(problem._inference_problem, "evaluate_cost_fn", None) is not None:
            value = _reduced_cost(problem)
    regularization = problem.regularization
    if "batched_lanes" in frozenset(getattr(optimizer, "requires", ())) and (
        isinstance(problem.objective, MAP) or not isinstance(regularization, NoReg)
    ):
        regularization = _StructuredPriorRegularizer(problem._prior_blocks, regularization)
    return lower_residual_problem(
        problem,
        optimizer,
        evaluator=evaluator,
        lanes=lanes,
        gauss_newton=gauss_newton,
        value=value,
        regularization=regularization,
        joint_view=lambda: problem.joint_view(),
    )


__all__ = ["MLE", "MAP", "EstimationProblem"]
