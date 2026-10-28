"""_optsolve.py — the ``optsolve`` optimization composer.

``optsolve(problem, optimizer=NewtonTrustRegion())`` is the single entry
point for parameter estimation:

* ``problem`` is an ``InferenceProblem`` (an experiment or study paired with
  an MLE or MAP objective), a lowered study (``ds.lower(study)``), or an
  [OptimizationProblem][diff_soup.inference.OptimizationProblem]; the start
  point is the problem's, there is no positional ``x0`` argument.
* ``optimizer`` defaults to ``NewtonTrustRegion()``, so the minimal call is
  ``optsolve(problem)``.
* The declared problem lowers to the optimization problem the optimizer
  solves; an optimizer whose ``requires`` the problem's ``structure`` does
  not meet is refused before any evaluation.

``optsolve`` surfaces sentinel-stamped segment failures host-side (setting
``OptResult.status = SOLVER_FAILED`` with failure indices) and forwards
callbacks to the optimizer, which fires their ``on_start`` / ``on_step`` /
``on_end`` lifecycle during the solve.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

from diff_soup._distributed import assert_decision_value, local_value
from diff_soup._errors import ConfigurationError
from diff_soup._logging import log as logger
from diff_soup._warnings import warn_user
from diff_soup.inference._abc import RESUME
from diff_soup.inference.fitting._protocols import StatisticalInferenceProblem
from diff_soup.inference.fitting.result import OptResult, TerminationStatus

#: The built-in solvers and the structure each requires, in the order a refusal lists them.
_BUILT_IN = (
    ("LevenbergMarquardt", frozenset({"residual_jacobian"})),
    ('NewtonTrustRegion(hessian="true")', frozenset({"value_gradient", "hessian"})),
    ('NewtonTrustRegion(hessian="gauss-newton")', frozenset({"value_gradient", "gauss_newton"})),
    ("BFGS", frozenset({"value_gradient"})),
    ("GradientDescent", frozenset({"value_gradient"})),
    ("Optax", frozenset({"value_gradient"})),
    ("TrustRegion", frozenset({"batched_lanes"})),
)


def _numeric_problem(problem: Any, optimizer: Any) -> Any:
    """The optimization problem ``optimizer`` solves for the declared ``problem``.

    A statistical problem (an ``InferenceProblem`` or a lowered study) lowers
    itself; an object with a ``structure`` is an optimization problem already;
    a residual problem (one with ``segment_fns``) lowers to a
    ``LeastSquaresProblem``.
    """
    if isinstance(problem, StatisticalInferenceProblem):
        return problem.numeric_problem(optimizer)
    if getattr(problem, "structure", None) is not None:
        return problem
    if getattr(problem, "segment_fns", None) is not None:
        from diff_soup.inference.fitting.inference_problem import lower_residual_problem

        return lower_residual_problem(problem, optimizer)
    raise TypeError(
        f"optsolve takes an InferenceProblem, a lowered study (ds.lower(study)) or an "
        f"OptimizationProblem; got {type(problem).__name__}."
    )


def _refuse_unmet(numeric: Any, optimizer: Any) -> None:
    """Refuse ``optimizer`` when ``numeric`` lacks structure the optimizer requires."""
    structure = frozenset(numeric.structure)
    missing = (frozenset(getattr(optimizer, "requires", ())) - {RESUME}) - structure
    if not missing:
        return
    suited = [name for name, needed in _BUILT_IN if needed <= structure]
    raise ConfigurationError(
        f"{type(optimizer).__name__} requires the problem structure {sorted(missing)}, which "
        f"this problem does not provide: it provides {sorted(structure)}. Solvers whose "
        f"requirements it meets: {', '.join(suited) if suited else 'none of the built-in solvers'}."
    )


def _solve(optimizer: Any, numeric: Any, callbacks: tuple, state: Any = None):
    """``optimizer.solve`` on ``numeric`` after the structure check; ``state=`` when it resumes."""
    _refuse_unmet(numeric, optimizer)
    if RESUME in frozenset(getattr(optimizer, "requires", ())):
        return optimizer.solve(numeric, callbacks=callbacks, state=state)
    return optimizer.solve(numeric, callbacks=callbacks)


def _surface_failures(result: OptResult, problem: Any) -> OptResult:
    """Re-detect sentinel-penalized segment solve failures and report them.

    Each segment whose inner PDE solve hit ``max_steps`` is stamped with a sentinel
    residual so the optimizer can continue, but this hides the failure from the
    final result.  This helper reads the data residual the optimizer recorded at
    the converged params, or re-evaluates it host-side through ``count_failed``
    when none is recorded or its block structure is not readable from it; if
    any segment failed, it warns once and tags the result ``SOLVER_FAILED`` with
    the count and indices.
    """
    # A fit that reports ``failure_sentinel_absent`` evaluated every final
    # residual while building its accepted statistics and found no sentinel.
    # Re-running the complete joint residual here would add a PDE solve and,
    # for a heterogeneous sharded joint, compile a new serial all-child graph
    # solely to rediscover the same fact.
    if bool(result.info.get("failure_sentinel_absent", False)):
        return result
    count_failed = getattr(problem, "count_failed", None)
    if count_failed is None:
        return result
    recorded = _recorded_failures(result, problem)
    n_failed, failed_indices = count_failed(result.params) if recorded is None else recorded
    if assert_decision_value(n_failed == 0, label="failure audit"):
        return result
    n_seg = int(local_value(getattr(problem, "n_segments", len(failed_indices))))
    msg = (
        f"optsolve: {n_failed} of {n_seg} segment PDE solve(s) did not reach t_end "
        f"(segments {failed_indices}); their residuals were penalized with a "
        f"sentinel. The fit may be unreliable — raise the solver's max_steps or "
        f"loosen its tolerances."
    )
    warn_user(msg)
    logger.warning(msg)
    return dataclasses.replace(
        result,
        status=TerminationStatus.SOLVER_FAILED,
        message=TerminationStatus.SOLVER_FAILED.value,
        n_failed_segments=n_failed,
        failed_segment_indices=failed_indices,
    )


def _data_row_count(problem: Any) -> int | None:
    """Host-static row count of a data problem's flat residual, or ``None``."""
    for name in ("residual_rows", "n_residuals"):
        rows = getattr(problem, name, None)
        if rows is not None:
            return int(rows)
    return None


def _attach_final_data_rows(result: OptResult, problem: Any) -> OptResult:
    """Record the data rows of the final residual and Jacobian an optimizer holds.

    An optimizer that ends holding the residual and Jacobian of its returned
    point records them as ``info["final_residual"]`` / ``info["final_jacobian"]``.
    Their rows are the data residual of the problem's ``diagnostic_problem``
    (the problem itself when it has none), then the MAP prior rows in prior
    block order, then the regularization rows. The leading data rows are
    recorded as ``info["final_data_residual"]`` / ``info["final_data_jacobian"]``,
    the whitened data residual and Jacobian at ``result.params``.
    """
    residual = result.info.get("final_residual")
    jacobian = result.info.get("final_jacobian")
    if residual is None or jacobian is None:
        return result
    data_problem = getattr(problem, "diagnostic_problem", problem)
    rows = _data_row_count(data_problem)
    n_params = int(np.asarray(result.params).size)
    if (
        rows is None
        or residual.ndim != 1
        or rows > residual.shape[0]
        or jacobian.shape != (residual.shape[0], n_params)
    ):
        return result
    return dataclasses.replace(
        result,
        info={
            **dict(result.info),
            "final_data_residual": residual[:rows],
            "final_data_jacobian": jacobian[:rows],
        },
    )


def _recorded_failures(result: OptResult, problem: Any) -> tuple[int, tuple[int, ...]] | None:
    """Sentinel failures read from the recorded final data residual, or ``None``.

    The residual blocks are those ``count_failed`` evaluates: one block per
    child of a joint problem (``residual_rows_per_child``), one equal block per
    window of a batched (observed-transition multishoot) residual, and one
    block for a residual of a single closure. ``None`` (re-evaluate with
    ``count_failed``) when no residual is recorded or its blocks are separate
    closures whose row counts the problem does not expose.
    """
    from diff_soup.inference.fitting._forward_seam import _FAILED_SENTINEL

    residual = result.info.get("final_data_residual")
    if residual is None:
        return None
    data_problem = getattr(problem, "diagnostic_problem", problem)
    per_child = getattr(data_problem, "residual_rows_per_child", None)
    n_seg = int(getattr(data_problem, "n_segments", 1) or 1)
    if per_child:
        sizes = tuple(int(rows) for rows in per_child)
    elif getattr(data_problem, "batched_r_fn", None) is not None:
        if residual.size % n_seg:
            return None
        sizes = (residual.size // n_seg,) * n_seg
    elif len(tuple(getattr(data_problem, "segment_fns", ()))) == 1:
        sizes = (residual.size,)
    else:
        return None
    if sum(sizes) != residual.size:
        return None
    bounds = np.cumsum((0,) + sizes)
    failed = tuple(
        index
        for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:]))
        if stop > start and np.any(np.abs(residual[start:stop]) >= _FAILED_SENTINEL)
    )
    return len(failed), failed


def _maybe_freeze_then_relax(
    problem: Any,
    engine: Any,
    *,
    callbacks: tuple = (),
) -> tuple[Any, dict]:
    """Run a theta-only warm-start stage for a decision-variable Multishoot.

    When the problem declares ``theta_only_steps > 0`` (the freeze-then-relax
    strategy), a stage-1 LM run with that budget optimizes over the reduced θ̃-only
    problem (window states pinned at their manifold-init values); the recovered θ̃
    warm-starts the full decision-variable problem for the main solve.  A no-op
    for every other problem type.

    Args:
        problem: The inference problem.
        engine: The resolved numeric-core engine (to clone the budget for stage 1).
        callbacks: Solver callbacks forwarded to the stage-1 solve.

    Returns:
        ``(problem, metadata)``. The problem is possibly
        θ-warm-started; metadata records the otherwise separate stage-1 cost,
        status, and wall time for trustworthy phase accounting.
    """
    steps = int(local_value(getattr(problem, "theta_only_steps", 0) or 0))
    if steps <= 0 or not getattr(problem, "is_decision_variable", False):
        return problem, {}
    replacements = {"max_steps": steps}
    theta_only = problem.theta_only_problem()
    fixed_indices = getattr(engine, "fixed_indices", None)
    if fixed_indices is not None:
        replacements["fixed_indices"] = tuple(
            index
            for index in map(int, fixed_indices)
            if index < int(local_value(np.asarray(theta_only.x0).size))
        )
    stage1_engine = dataclasses.replace(engine, **replacements)
    for callback in callbacks:
        bind = getattr(callback, "bind", None)
        if callable(bind):
            bind(theta_only)
    stage1 = _solve(stage1_engine, _numeric_problem(theta_only, stage1_engine), callbacks)
    reduced_theta = np.asarray(stage1.params)
    full_theta = np.concatenate([reduced_theta, local_value(problem.x0)[len(reduced_theta) :]])
    logger.info(
        "optsolve: freeze-then-relax stage 1 (theta-only, %d steps) cost %.3e -> warm "
        "start the full decision-variable solve.",
        steps,
        float(stage1.cost),
    )
    return problem.with_warm_theta(full_theta), {
        "restricted_initialization_complete": True,
        "restricted_initialization_wall_time_s": float(stage1.wall_time),
        "restricted_initialization_steps": int(stage1.n_steps),
        "restricted_initialization_accepted": int(stage1.n_accepted),
        "restricted_initialization_cost": float(stage1.cost),
        "restricted_initialization_status": str(stage1.status),
    }


def _attach_joint_children(result: OptResult, problem: Any) -> OptResult:
    """Attach each joint child's fitted parameter reconstruction.

    Structural joints use positional experiment keys. The public problem's
    result finalizer replaces them with the declared experiment IDs and adds
    per-experiment handle lookup.
    """
    child_results = getattr(problem, "child_results", None)
    if not callable(child_results):
        return result
    return dataclasses.replace(
        result,
        experiments=dict(enumerate(child_results(result.params))),
    )


def _attach_parameter_schema(result: OptResult, problem: Any) -> OptResult:
    """Attach Recipe handle coordinates supplied by a structural problem."""
    schema = getattr(problem, "parameter_schema", None)
    if result.parameter_schema is not None or schema is None:
        return result
    return dataclasses.replace(result, parameter_schema=schema)


def _attach_diagnostics(result: OptResult, problem: Any) -> OptResult:
    """Attach [OptDiagnostics][diff_soup.inference.fitting.diagnostics.OptDiagnostics].

    Computed at the converged params from the data residual and Jacobian the
    optimizer recorded there (``info["final_data_residual"]`` /
    ``info["final_data_jacobian"]``); an optimizer that records none costs one
    extra ``jacfwd`` and one residual evaluation of the residual closures at
    the solution.  Fires
    [IdentifiabilityWarning][diff_soup.inference.fitting.diagnostics.IdentifiabilityWarning]
    when
    the θ-block is rank-deficient, and records the warning text plus the
    window-state warm-start handle on the result.
    """
    from diff_soup.inference.fitting.diagnostics import compute_diagnostics

    diagnostic_problem = getattr(problem, "diagnostic_problem", problem)
    diag = compute_diagnostics(
        diagnostic_problem,
        result.params,
        residual=result.info.get("final_data_residual"),
        jacobian=result.info.get("final_data_jacobian"),
    )
    diag = dataclasses.replace(
        diag,
        ad_mode=result.info.get("ad_mode"),
        hessian=result.info.get("hessian"),
        hessian_composition=result.info.get("hessian_composition"),
        subproblem=result.info.get("subproblem"),
        linear_solver=result.info.get("linear_solver"),
    )
    window_states = getattr(diagnostic_problem, "window_states_of", None)
    ws = window_states(result.params) if callable(window_states) else None
    return dataclasses.replace(result, diagnostics=diag, window_states=ws)


def _recorded_fingerprint(result: OptResult) -> Any:
    """The parameter schema fingerprint an earlier result records, or ``None``."""
    recorded = result.info.get("schema_fingerprint") or (
        None if result.parameter_schema is None else result.parameter_schema.fingerprint
    )
    state = result.state
    if not recorded and state is not None and "fingerprint" in state:
        recorded = str(np.asarray(state["fingerprint"]))
    return recorded


def _foreign(problem: Any, recorded: Any) -> bool:
    """Whether a recorded fingerprint names another problem than ``problem``."""
    schema = getattr(problem, "parameter_schema", None)
    return bool(
        schema is not None
        and recorded
        and not str(recorded).startswith("structural:")
        and str(recorded) != str(schema.fingerprint)
    )


def _checkpoint_source(problem: Any, directory: Any) -> Any:
    """The newest attempt file of ``directory`` as a result to resume from, or ``None``."""
    from diff_soup.inference.fitting.callbacks import _attempt_files

    files = _attempt_files(directory)
    if not files:
        return None
    newest = files[-1]
    result = OptResult.load(newest)
    recorded = _recorded_fingerprint(result)
    if _foreign(problem, recorded):
        raise ValueError(
            f"optsolve(checkpoint={str(directory)!r}) holds {newest.name} of another problem: "
            f"its parameter schema fingerprint is {str(recorded)!r}, this problem's is "
            f"{str(problem.parameter_schema.fingerprint)!r}; give a directory of this problem "
            "or an empty one."
        )
    return result


def _resume_from(problem: Any, resume: Any) -> tuple[Any, Any]:
    """``problem`` warm-started at the point of an earlier result, and that result's state."""
    if isinstance(resume, OptResult):
        result = resume
    else:
        from diff_soup.inference.fitting.callbacks import checkpoint_files

        if os.path.isdir(str(resume)):
            files = checkpoint_files(resume)
            if not files:
                raise FileNotFoundError(
                    f"optsolve(resume={str(resume)!r}) names a directory without a "
                    "CheckpointCallback result (attempt-NNNNNN.npz or step-NNNNNN.npz) to "
                    "resume from."
                )
            resume = files[-1]
        result = OptResult.load(resume)
    schema = getattr(problem, "parameter_schema", None)
    recorded = _recorded_fingerprint(result)
    if _foreign(problem, recorded):
        raise ValueError(
            "optsolve(resume=...) needs a result of the same problem: the result's "
            f"parameter schema fingerprint is {recorded!r}, this problem's is "
            f"{schema.fingerprint!r}."
        )
    with_start = getattr(problem, "with_start", None)
    if not callable(with_start):
        raise TypeError(
            f"optsolve(resume=...) needs a problem with with_start; {type(problem).__name__} "
            "has none."
        )
    return with_start(result), result.state


def _default_optimizer(problem: Any) -> Any:
    """``NewtonTrustRegion()``, or its Gauss-Newton form when the fit includes window states.

    A multiple-shooting fit whose window starts are decision variables has
    hundreds of them, and the exact Hessian of that many costs memory
    quadratic in their number; the Gauss-Newton matrix does not.
    """
    from diff_soup.inference.solvers.newton_trust_region import NewtonTrustRegion

    data_problem = getattr(problem, "diagnostic_problem", problem)
    if bool(getattr(data_problem, "fits_window_states", False)):
        return NewtonTrustRegion(hessian="gauss-newton")
    return NewtonTrustRegion()


def optsolve(
    problem: Any,
    optimizer: Any = None,
    *,
    callbacks: tuple = (),
    diagnostics: bool = True,
    run_restricted_initialization: bool = True,
    restricted_callbacks: tuple = (),
    resume: Any = None,
    checkpoint: Any = None,
) -> OptResult:
    """Fit a problem with an optimizer and return the result.

    `problem` is an `InferenceProblem`, a study lowered with `ds.lower(study)`, or an
    `OptimizationProblem`. When a time-stepping solve at the solution fails, the result has
    status `SOLVER_FAILED` and a warning is raised.

    Args:
        problem: The problem to fit.
        optimizer: The optimizer, `ds.inference.NewtonTrustRegion()` when `None`, or `NewtonTrustRegion(hessian="gauss-newton")` for a multiple-shooting fit of window states.
        callbacks: [Callback][diff_soup.inference.Callback] objects called during the fit.
        diagnostics: If `True`, attach the fit diagnostics to the result.
        run_restricted_initialization: If `True` and the trajectory sets `warm_start_steps`, first fit the parameters alone for that many steps.
        restricted_callbacks: Callbacks for that first fit.
        resume: An `OptResult`, a saved result file, or a checkpoint directory to continue from.
        checkpoint: A directory the fit is saved to as it runs, and resumed from when it already holds a checkpoint.

    Returns:
        The [OptResult][diff_soup.inference.OptResult] of the fit.

    Raises:
        ConfigurationError: If the optimizer requires a structure the problem does not provide.
        ValueError: If `resume` or `checkpoint` holds a fit of a different problem.
    """
    from diff_soup.utils.precision import warn_if_float32_default

    warn_if_float32_default()

    if optimizer is None:
        optimizer = _default_optimizer(problem)

    callbacks = tuple(callbacks)
    if checkpoint is not None:
        from diff_soup.inference.fitting.callbacks import CheckpointCallback

        if resume is None:
            resume = _checkpoint_source(problem, checkpoint)
        callbacks = callbacks + (CheckpointCallback(checkpoint),)
    state = None
    if resume is not None:
        problem, state = _resume_from(problem, resume)
    declared_problem = problem
    statistical_problem = (
        declared_problem if isinstance(declared_problem, StatisticalInferenceProblem) else None
    )

    if run_restricted_initialization:
        problem, stage_info = _maybe_freeze_then_relax(
            problem,
            optimizer,
            callbacks=tuple(restricted_callbacks),
        )
    else:
        stage_info = {}
    # The declared problem keeps the metadata of the fit (schema, failure audit,
    # children); the optimizer solves the optimization problem it lowers to.
    numeric = _numeric_problem(problem, optimizer)
    for callback in callbacks:
        bind = getattr(callback, "bind", None)
        if callable(bind):
            bind(problem)
    result = _solve(optimizer, numeric, callbacks, state)
    result = dataclasses.replace(
        result,
        info={
            **dict(result.info),
            **stage_info,
            "main_wall_time_s": float(result.wall_time),
        },
    )
    result = _attach_parameter_schema(result, problem)
    result = _attach_final_data_rows(result, problem)
    result = _surface_failures(result, problem)
    result = _attach_joint_children(result, problem)
    if diagnostics:
        result = _attach_diagnostics(result, declared_problem)

    if statistical_problem is not None:
        result = statistical_problem.finalize_opt_result(result, optimizer=optimizer)

    return result
