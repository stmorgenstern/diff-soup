"""diagnostics.py — OptDiagnostics and identifiability warnings.

After a fit completes, [OptDiagnostics][] makes the *conditioning* of the
parameter estimate visible.  A poor fit on a PDE inversion usually means the
problem is under-determined, not that the optimizer failed; this module surfaces
that evidence on every [OptResult][diff_soup.inference.fitting.result.OptResult].

Fields on [OptDiagnostics][]:

* ``theta_singular_values`` — singular values of the θ-block of the residual
  Jacobian ``J`` at the solution.
* ``jacobian_cond`` — ``σ_max / σ_min`` of that θ-block (``inf`` on a zero column).
* ``window_obs_rank`` — per-window rank of ``∂(obs rows)/∂s̃_w`` (decision-variable
  multishoot mode only; ``None`` for θ-only or observed-IC problems).
* ``continuity_defect_wrms`` — raw continuity defect RMS (decision-variable mode
  only; ``None`` otherwise).
* ``per_series_cost`` — cost split across data series (detects one observable
  swamping another even when the total cost looks fine).
* ``warnings`` — human-readable notes carried on the result.

**Warning policy.**

* θ rank-deficiency (``cond > θ_cond_warn`` or a near-zero column) raises a
  [IdentifiabilityWarning][] naming the offending parameter — the actionable
  signal that the data under-determine the chosen parameters.
* Window under-determination (decision-variable mode) is recorded as an INFO note
  and logged at INFO.  It is always true for population/distribution observables
  (permutation-invariant null space) and must not be raised as a warning-as-error.

Computed host-side after the solve from the data residual and Jacobian the
optimizer recorded at the converged params (Levenberg-Marquardt ends holding
both), so the diagnostics add no forward solve.  Without recorded arrays, one
extra ``jacfwd`` and one residual evaluation of the residual closures at the
converged params supply them.  The Jacobian inherits the residual's compute
dtype; no hard fp64 casts.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._logging import log as logger
from diff_soup._warnings import warn_user

# Default θ-conditioning threshold above which the identifiability warning fires.
# A condition number of 1e8 in fp64 means ~8 digits of precision are lost — the
# parameter combination is effectively unobservable from the data.
_THETA_COND_WARN = 1.0e8
# A column whose norm is below this fraction of the largest column norm is treated
# as a (near-)zero column — that θ leaf does not move the residual at all.
_ZERO_COL_REL = 1.0e-12


class IdentifiabilityWarning(UserWarning):
    """A fitted parameter (combination) is poorly identified by the data.

    Raised when the θ-block of the residual Jacobian at the solution is
    rank-deficient (large condition number or a zero column).  The message names
    the offending parameter(s).  A poor fit signaled this way is an
    under-determination issue, not an optimizer failure — the remedy is to
    add observations, tighten the measurement, or reduce the learnable set.
    """


@dataclasses.dataclass(frozen=True)
class OptDiagnostics:
    """The quality report of a fit, at `result.diagnostics`, from the residual Jacobian at the solution.

    A condition number above `1e8`, or a parameter that does not move the residual, also
    issues an `IdentifiabilityWarning`.

    Attributes:
        theta_singular_values: The singular values of the Jacobian with respect to the model parameters.
        jacobian_cond: The ratio of the largest to the smallest of them, `inf` when the smallest is zero.
        window_obs_rank: Per fitted window start, the rank of the window's observations with respect to it.
        continuity_defect_wrms: Per window boundary, the root mean square of the continuity mismatch, in state units.
        continuity_unrepresented: Per window boundary, the weighted continuity mismatch a reduced `StateBasis` cannot represent.
        per_series_cost: The cost of each measurement stream, or of each window under `Windows.from_data`.
        series_names: The stream names, parallel to `per_series_cost`, or `None`.
        param_group_names: The names of the parameter groups.
        param_group_cond: The condition number of each parameter group, parallel to `param_group_names`.
        ad_mode: The derivative mode of the gradient.
        hessian: The curvature the optimizer used, `"true"` or `"gauss-newton"`.
        hessian_composition: The autodiff composition of the exact Hessian.
        subproblem: How the optimizer solved its step subproblem.
        linear_solver: The factorization the optimizer used.
        warnings: The identifiability and window notes, as text.
    """

    theta_singular_values: Optional[np.ndarray] = None
    jacobian_cond: float = float("nan")
    window_obs_rank: Optional[tuple] = None
    continuity_defect_wrms: Optional[np.ndarray] = None
    continuity_unrepresented: Optional[np.ndarray] = None
    per_series_cost: tuple = ()
    series_names: Optional[tuple] = None
    param_group_names: tuple = ()
    param_group_cond: tuple = ()
    ad_mode: Optional[str] = None
    hessian: Optional[str] = None
    hessian_composition: Optional[str] = None
    subproblem: Optional[str] = None
    linear_solver: Optional[str] = None
    warnings: tuple = ()


def _theta_jacobian(
    problem: Any, x_flat: np.ndarray, jacobian: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Stack the residual Jacobian ``∂r/∂x`` at ``x_flat`` (host numpy).

    A ``jacobian`` recorded at ``x_flat`` is returned as it is.  Otherwise the
    Jacobian is evaluated in forward mode, because the problem already resolved the forward engine
    to a forward-capable adjoint (the same derivative path the LM core uses).  When
    the problem exposes a batched Jacobian (observed-transition multishoot), that
    single ``jacfwd``-over-the-map path is used so the diagnostic cost is independent
    of the window count; otherwise each segment residual's ``jacfwd`` is concatenated.

    A sharded joint reduces ``J^T J`` across the device mesh and never forms the
    global Jacobian, so this returns a small ``(n_params, n_params)`` surrogate with
    the same Gram (``sqrt(Λ) Vᵀ`` from the reduced ``J^T J``).  Column norms, singular
    values, and per-group conditioning are all determined by ``J^T J``, so every
    diagnostic computed from this surrogate is identical to one from the full Jacobian,
    without gathering it.

    Returns:
        The ``(n_rows, n_params)`` Jacobian as host numpy, or ``None`` if the
        problem exposes no residual closures.
    """
    if jacobian is not None:
        return np.asarray(jacobian)
    x = jnp.asarray(x_flat)
    if getattr(problem, "is_sharded", False):
        evaluate_lsq = getattr(problem, "evaluate_lsq_fn", None)
        if evaluate_lsq is not None:
            _cost, _jtr, jtj, _n_active, _any_failed = evaluate_lsq(x)
            gram = np.asarray(jtj)
            eigvals, eigvecs = np.linalg.eigh(gram)
            return np.sqrt(np.clip(eigvals, 0.0, None))[:, None] * eigvecs.T
    if "residual_jacobian" in (getattr(problem, "structure", None) or ()):
        _r, jac = problem.residual_and_jacobian(x)
        return np.asarray(jac)
    batched_rj = getattr(problem, "batched_rj_fn", None)
    if batched_rj is not None:
        _r, jac = batched_rj(x)
        return np.asarray(jac)
    seg_fns = getattr(problem, "segment_fns", None)
    if not seg_fns:
        return None
    blocks = []
    for fn in seg_fns:
        blocks.append(np.asarray(jax.jacfwd(fn)(x)))
    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def _per_series_cost(
    problem: Any, x_flat: np.ndarray, residual: Optional[np.ndarray] = None
) -> tuple:
    """Return per-stream/segment cost ``0.5·‖r_s‖²`` and the matching names.

    A ``residual`` recorded at ``x_flat`` is split in place of evaluating the
    residual; residual blocks that are separate closures are still evaluated.

    When the problem exposes named-stream row slices (single-shoot, where one
    residual carries every stream), the single residual is evaluated once and
    split by stream so each stream's cost is labeled by name.  Otherwise the cost
    is taken per residual block (multishoot), with no stream names.

    Returns ``(costs, names)`` where ``names`` is ``None`` for the per-block case.

    A sharded joint computes its cost as a mesh sum and never gathers the global
    residual; materialising per-series rows here (``seg_fns``/``batched_r``) would
    undo that no-gather property after the optimization path preserved it.  On a
    sharded layout the split is skipped — the conditioning diagnostics above use the
    reduced Gram, so the whole diagnostic stays no-gather under default ``fit``.
    """
    if getattr(problem, "is_sharded", False):
        return (), None
    seg_fns = getattr(problem, "segment_fns", None)
    if not seg_fns:
        return (), None
    x = jnp.asarray(x_flat)
    slices = getattr(problem, "stream_residual_slices", ())
    names = getattr(problem, "stream_names", ())
    recorded = None if residual is None else np.asarray(residual)
    if slices and len(seg_fns) == 1:
        r = np.asarray(seg_fns[0](x)) if recorded is None else recorded
        costs = [0.5 * float(np.sum(r[start:stop] ** 2)) for (start, stop) in slices]
        return tuple(costs), (tuple(names) if names else None)
    # Observed-transition multishoot: one batched residual, split per window.
    batched_r = getattr(problem, "batched_r_fn", None)
    n_seg = int(getattr(problem, "n_segments", 1) or 1)
    if not slices and batched_r is not None and n_seg > 1:
        r = np.asarray(batched_r(x)) if recorded is None else recorded
        if r.size % n_seg == 0:
            blocks = r.reshape((n_seg, -1))
            return tuple(0.5 * float(np.sum(blocks[i] ** 2)) for i in range(n_seg)), None
    if recorded is not None and len(seg_fns) == 1:
        return (0.5 * float(np.sum(recorded**2)),), None
    costs = [0.5 * float(np.sum(np.asarray(fn(x)) ** 2)) for fn in seg_fns]
    return tuple(costs), None


def _per_group_cond(jac: Optional[np.ndarray], problem: Any) -> tuple:
    """Return per-parameter-group θ-block condition numbers and their names.

    For each named parameter group, slices the group's columns out of the θ-block
    Jacobian and reports ``σ_max / σ_min`` (``inf`` on a rank-deficient block).
    """
    names = tuple(getattr(problem, "param_group_names", ()) or ())
    cols = tuple(getattr(problem, "param_group_cols", ()) or ())
    if jac is None or not names or len(names) != len(cols):
        return (), ()
    conds: list = []
    for group_cols in cols:
        idx = np.asarray(group_cols, dtype=int)
        if idx.size == 0:
            conds.append(float("nan"))
            continue
        sv = np.linalg.svd(jac[:, idx], compute_uv=False)
        smin = float(sv[-1]) if sv.size else 0.0
        smax = float(sv[0]) if sv.size else 0.0
        conds.append(float("inf") if smin <= 0.0 else smax / smin)
    return names, tuple(conds)


def _theta_names(problem: Any) -> Optional[tuple]:
    """Best-effort θ-leaf names for the warning message (or ``None``)."""
    return getattr(problem, "theta_names", None)


def compute_diagnostics(
    problem: Any,
    x_flat: np.ndarray,
    *,
    residual: Optional[np.ndarray] = None,
    jacobian: Optional[np.ndarray] = None,
    theta_cond_warn: float = _THETA_COND_WARN,
    emit_warning: bool = True,
) -> OptDiagnostics:
    """Build [OptDiagnostics][] at the converged params.

    Computes the θ-block SVD, condition number, and per-series cost, then applies
    the warning policy: raises [IdentifiabilityWarning][] for a rank-deficient
    θ-block (naming the offending parameter), and records an INFO note for window
    under-determination in decision-variable mode.

    Args:
        problem: The inference problem (exposes ``segment_fns`` / ``x0`` and,
            in decision-variable mode, ``window_diagnostics``).
        x_flat: The converged flat parameter vector.
        residual: The flat residual of ``problem`` at ``x_flat``, when the
            optimizer recorded it; the per-series cost splits it instead of
            evaluating the residual.
        jacobian: The residual Jacobian of ``problem`` at ``x_flat``, when the
            optimizer recorded it; the conditioning diagnostics use it instead
            of a ``jacfwd``.
        theta_cond_warn: Condition-number threshold for the identifiability
            warning (default ``1e8``).
        emit_warning: Whether to emit the real warning (set ``False`` to
            collect the note without raising under ``-W error``).

    Returns:
        A frozen [OptDiagnostics][].
    """
    notes: list[str] = []

    # --- θ-block SVD + condition number ----------------------------------
    # The θ block is the leading learnable-parameter columns of the decision vector.
    # In decision-variable multishoot the vector also carries window-state blocks
    # ``[θ | s_0 | s_1 | ...]``; those are NOT θ — their conditioning is the window
    # rank diagnostic's job (an INFO note), not the θ identifiability warning's.  Use
    # the problem's θ-name count to size the block, falling back to the full
    # vector for single-shoot / observed-IC (where the vector is θ alone).
    sv = None
    cond = float("nan")
    theta_names = getattr(problem, "theta_names", None)
    n_theta = len(theta_names) if theta_names is not None else int(np.asarray(x_flat).size)
    jac = _theta_jacobian(problem, x_flat, jacobian)
    jac_theta_full = None
    if jac is not None and jac.size and n_theta > 0:
        jac_theta_full = jac[:, :n_theta]
    if jac is not None and jac.size and n_theta > 0:
        jac_theta = jac[:, :n_theta]
        sv_arr = np.linalg.svd(jac_theta, compute_uv=False)
        sv = sv_arr
        smin = float(sv_arr[-1]) if sv_arr.size else 0.0
        smax = float(sv_arr[0]) if sv_arr.size else 0.0
        cond = float("inf") if smin <= 0.0 else smax / smin

        # The weights of a learnable function (a network) are not identifiable
        # one by one, so the rank tests read only the other columns.
        function_cols = set(getattr(problem, "function_columns", ()) or ())
        tested = np.asarray([i for i in range(n_theta) if i not in function_cols], dtype=int)
        if function_cols and tested.size == 0:
            notes.append(
                "optsolve: every fitted parameter is a learnable-function weight, so the "
                "identifiability check was skipped."
            )
        elif tested.size:
            jac_tested = jac_theta[:, tested]
            # Zero-column detection (a θ leaf the residual is blind to).
            col_norms = np.linalg.norm(jac_tested, axis=0)
            max_col = float(np.max(col_norms)) if col_norms.size else 0.0
            zero_cols = (
                tested[np.where(col_norms <= _ZERO_COL_REL * max(max_col, 1.0))[0]]
                if max_col > 0.0
                else tested
            )
            if function_cols:
                tested_sv = np.linalg.svd(jac_tested, compute_uv=False)
                tested_min = float(tested_sv[-1]) if tested_sv.size else 0.0
                tested_max = float(tested_sv[0]) if tested_sv.size else 0.0
                tested_cond = float("inf") if tested_min <= 0.0 else tested_max / tested_min
            else:
                tested_cond = cond
        else:
            zero_cols = np.zeros(0, dtype=int)
            tested_cond = cond

        if tested.size and (zero_cols.size > 0 or tested_cond > theta_cond_warn):
            names = _theta_names(problem)
            if zero_cols.size > 0:
                bad = (
                    tuple(names[i] for i in zero_cols)
                    if names is not None and len(names) == n_theta
                    else tuple(int(i) for i in zero_cols)
                )
                detail = f"parameter(s) {bad} do not move the residual (zero Jacobian column)"
            else:
                block = "theta-block" if not function_cols else "non-function theta-block"
                detail = f"{block} condition number {tested_cond:.2e} exceeds {theta_cond_warn:.0e}"
            msg = (
                f"optsolve: the learnable-parameter Jacobian is rank deficient: {detail}. "
                "The observations must determine every fitted parameter direction; "
                "add independent observations or reduce the learnable set."
            )
            notes.append(msg)
            if emit_warning:
                warn_user(msg, IdentifiabilityWarning)
            logger.warning(msg)

    # --- per-stream cost split (by name when available) ------------------
    per_series, series_names = _per_series_cost(problem, x_flat, residual)

    # --- per-parameter-group conditioning --------------------------------
    group_names, group_cond = _per_group_cond(jac_theta_full, problem)

    # --- window diagnostics (decision-variable mode only) ----------------
    window_obs_rank = None
    continuity_wrms = None
    continuity_unrepresented = None
    win = getattr(problem, "window_diagnostics", None)
    if callable(win):
        window_obs_rank, continuity_wrms, continuity_unrepresented, win_notes = win(x_flat)
        notes.extend(win_notes)

    return OptDiagnostics(
        theta_singular_values=None if sv is None else np.asarray(sv),
        jacobian_cond=cond,
        window_obs_rank=window_obs_rank,
        continuity_defect_wrms=continuity_wrms,
        continuity_unrepresented=continuity_unrepresented,
        per_series_cost=per_series,
        series_names=series_names,
        param_group_names=group_names,
        param_group_cond=group_cond,
        warnings=tuple(notes),
    )
