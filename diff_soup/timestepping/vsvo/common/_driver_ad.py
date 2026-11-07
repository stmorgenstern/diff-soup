"""Derivatives of the driver's loop: the forward loop's tangent and the composable loop's rules."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._jax_compat import Primitive as _Primitive
from diff_soup._jax_compat import ad as _ad
from diff_soup._jax_compat import batching as _batching
from diff_soup._jax_compat import mlir as _mlir
from diff_soup._jax_compat import partial_eval as _partial_eval
from diff_soup.timestepping.vsvo.common._driver_finalize import _finalize
from diff_soup.timestepping.vsvo.common._driver_loop import _loop_fns
from diff_soup.timestepping.vsvo.common._driver_state import _loop_init

from .checkpointed_loop import checkpointed_while_loop

_COMPOSABLE_REVERSE_ERROR = (
    "Reverse-mode differentiation was applied directly to a solve running the "
    "composable forward-sensitivity mode (ForwardAdjoint(composable=True)). "
    "This mode computes forward sensitivities (jax.jvp / jax.jacfwd) and "
    "supports differentiating functions of those sensitivities, for example "
    "jax.jacrev(jax.jacfwd(solve)), or jax.grad of a scalar criterion of "
    "jax.jacfwd(solve). For a reverse-mode gradient of the solution itself, "
    "solve with adjoint=CheckpointAdjoint() instead."
)


_COMPOSABLE_NESTED_FORWARD_ERROR = (
    "A second forward-mode differentiation was applied to a solve running the "
    "composable forward-sensitivity mode (ForwardAdjoint(composable=True)). "
    "This mode supports one forward differentiation of the solve plus "
    "reverse-mode differentiation of the resulting sensitivities. For "
    "forward-over-forward compositions, solve with the default "
    "ForwardAdjoint() instead."
)


# The reverse-linearization check below recognises the partial-evaluation
# tracer through the public ``jax.interpreters`` module. If a JAX release
# stops exporting it, the check degrades gracefully: bare reverse mode on the
# composable atom then surfaces jax's own trace-time error instead of the
# public message above.
_JAXPR_TRACER = getattr(_partial_eval, "JaxprTracer", None)


def _is_reverse_linearize_tangent(leaf) -> bool:
    """Whether ``leaf`` is an unknown partial-eval tangent (reverse linearization).

    Reverse-mode differentiation linearizes a function with symbolic (unknown)
    tangent inputs before transposing; a concrete forward differentiation
    carries known tangent values. Batch wrappers are unwrapped so the check
    also catches ``vmap``-of-``grad`` compositions.
    """
    if _JAXPR_TRACER is None:
        return False
    depth = 0
    while leaf is not None and depth < 8:
        if isinstance(leaf, _JAXPR_TRACER):
            return not leaf.pval.is_known()
        leaf = getattr(leaf, "val", None)
        depth += 1
    return False


_FORWARD_ONLY_REVERSE_ERROR = (
    "Reverse-mode differentiation (jax.grad, jax.vjp, jax.jacrev) was applied to a "
    "solve running the forward-only adjoint, which is the default and supports "
    "jax.jvp and jax.jacfwd only. For a reverse-mode gradient, solve with "
    "adjoint=ds.CheckpointAdjoint(). For reverse mode over forward sensitivities, "
    "use adjoint=ds.ForwardAdjoint(composable=True)."
)


# An identity on the forward-only loop's input tangents. Forward mode passes
# through it unchanged. Reverse mode partially evaluates it with unknown
# tangents before it reaches the loop (top level and inside jit alike), and
# its rule raises the public message there instead of JAX's raw while-loop
# error. The transpose rule is a second line of defence.
_forward_only_tangent_p = _Primitive("diffsoup_forward_only_tangent")


_forward_only_tangent_p.def_impl(lambda x: x)


_forward_only_tangent_p.def_abstract_eval(lambda aval: aval)


_mlir.register_lowering(
    _forward_only_tangent_p, _mlir.lower_fun(lambda x: x, multiple_results=False)
)


_batching.primitive_batchers[_forward_only_tangent_p] = lambda args, dims: (
    _forward_only_tangent_p.bind(args[0]),
    dims[0],
)


def _forward_only_tangent_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    out = _forward_only_tangent_p.bind(x)
    return out, (dx if type(dx) is _ad.Zero else _forward_only_tangent_p.bind(dx))


def _forward_only_tangent_partial_eval(trace, tracer):
    if not tracer.pval.is_known():
        raise ValueError(_FORWARD_ONLY_REVERSE_ERROR)
    return trace.default_process_primitive(_forward_only_tangent_p, (tracer,), {})


def _forward_only_tangent_transpose(cotangent, x):
    raise ValueError(_FORWARD_ONLY_REVERSE_ERROR)


_ad.primitive_jvps[_forward_only_tangent_p] = _forward_only_tangent_jvp


_ad.primitive_transposes[_forward_only_tangent_p] = _forward_only_tangent_transpose


_partial_eval.custom_partial_eval_rules[_forward_only_tangent_p] = (
    _forward_only_tangent_partial_eval
)


def _forward_loop_impl(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The forward-only step loop: the plain early-exit ``jax.lax.while_loop``."""
    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )
    return jax.lax.while_loop(cond_fun, body_fun, init_state)


@eqx.filter_custom_jvp
def _forward_loop(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The forward-only step loop, with a forward rule that refuses reverse mode.

    The primal and the forward rule run the same plain loop, so forward-mode
    results are unchanged. The rule marks the input tangents so that reverse
    mode stops with a message naming the reverse adjoint.
    """
    return _forward_loop_impl(
        engine,
        problem,
        cfg,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    )


@_forward_loop.def_jvp
def _forward_loop_jvp(primals, tangents):
    tangents = jax.tree_util.tree_map(
        lambda t: None if t is None else _forward_only_tangent_p.bind(t),
        tangents,
        is_leaf=lambda t: t is None,
    )
    final, d_final = eqx.filter_jvp(_forward_loop_impl, primals, tangents)
    return final, _canon_tangent(final, d_final)


def _canon_tangent(primal, tangent):
    """Normalize a tangent tree: arrays at every inexact leaf, ``None`` elsewhere.

    A checkpointed loop carry must keep one fixed structure across
    iterations, while ``filter_jvp`` prunes symbolically zero tangents to
    ``None``; materialising those to zeros (and forcing ``None`` at
    non-inexact leaves) restores the fixed structure.
    """

    def one(p, t):
        if eqx.is_inexact_array(p):
            return jnp.zeros_like(p) if t is None else t
        return None

    return jax.tree_util.tree_map(one, primal, tangent, is_leaf=lambda x: x is None)


def _composable_solve(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    ts_grid,
    ts_output,
    t0,
    t1,
    x0,
    theta,
    control0,
    atol_rows=None,
    dt0_value=None,
    history_in=None,
):
    """The composable forward-sensitivity solve.

    Three differentiable segments: the loop-state construction and the
    solution packaging are ordinary compiled functions (differentiable in
    both modes through their implicit-function-theorem atoms), while the
    step loop itself is `_composable_loop` — the atom whose forward
    rule makes ``jax.jacrev(jax.jacfwd(solve))`` compose. The undifferentiated
    solve runs the same early-exit primal loop as the default forward mode.
    """
    init_state = _composable_init(
        engine,
        problem,
        cfg,
        has_affect,
        control_folds_theta,
        ts_grid,
        ts_output,
        t0,
        t1,
        x0,
        theta,
        control0,
        atol_rows,
        dt0_value,
        history_in,
    )
    final = _composable_loop(
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    )
    return _composable_finalize(
        engine, problem, cfg, adjoint, has_affect, control_folds_theta, ts_grid, init_state, final
    )


_composable_init = eqx.filter_jit(_loop_init)


@eqx.filter_jit
def _composable_loop_primal(
    engine, problem, cfg, has_affect, control_folds_theta, init_state, ts_grid, t1, theta, control0
):
    """The composable atom's primal: the early-exit forward step loop."""
    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )
    return jax.lax.while_loop(cond_fun, body_fun, init_state)


_composable_finalize = eqx.filter_jit(_finalize)


@eqx.filter_custom_jvp
def _composable_loop(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    init_state,
    ts_grid,
    t1,
    theta,
    control0,
):
    """The composable forward-sensitivity step-loop atom.

    The primal is the early-exit ``jax.lax.while_loop`` — full adaptivity,
    O(1) memory, the same loop an undifferentiated forward solve runs. The
    attached forward rule (`_composable_loop_jvp`) re-integrates the
    trajectory together with its directional tangent inside a checkpointed
    loop, so the sensitivities it produces support reverse-mode
    differentiation on top.
    """
    del adjoint  # read by the differentiation rule, not by the primal
    return _composable_loop_primal(
        engine,
        problem,
        cfg,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    )


@_composable_loop.def_jvp
def _composable_loop_jvp(primals, tangents):
    """Forward rule: integrate the primal and its tangent together, checkpointed.

    Rejects, with public errors, the applications outside this mode's
    contract: reverse-mode differentiation of the solve itself, tangents for
    arrays embedded in the problem/solver objects, and per-step tracing under
    differentiation.
    """
    (
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        init_state,
        ts_grid,
        t1,
        theta,
        control0,
    ) = primals
    (d_engine, d_problem, d_cfg, d_adjoint, _d_ha, _d_cf, *loop_tangents) = tangents

    for leaf in jax.tree_util.tree_leaves(tuple(loop_tangents)):
        if _is_reverse_linearize_tangent(leaf):
            raise ValueError(_COMPOSABLE_REVERSE_ERROR)
    if jax.tree_util.tree_leaves((d_engine, d_problem, d_cfg, d_adjoint)):
        raise ValueError(
            "The composable forward-sensitivity mode differentiates the solve "
            "with respect to its array inputs (the time span, the initial "
            "state, theta, the save grid, and the event control state); a "
            "tangent arrived for an array embedded in the problem or solver "
            "object instead. Route differentiated parameters through theta."
        )
    if cfg.trace:
        raise ValueError(
            "The per-step trace is not recorded while differentiating through "
            "the composable forward-sensitivity mode. Run the traced solve "
            "undifferentiated, or differentiate with trace=False."
        )

    final, _joint_final, d_final = _composable_loop_jvp_core(
        engine,
        problem,
        cfg,
        adjoint,
        has_affect,
        control_folds_theta,
        (init_state, ts_grid, t1, theta, control0),
        tuple(loop_tangents),
    )
    # The joint loop's primal half is discarded: the primal output must reach
    # the caller through exactly one differentiable path (the primal-only
    # loop below), or an outer reverse pass double-counts the first-order
    # Hessian block of a criterion that reads both the solution and its
    # sensitivities. It is still returned from the core so tests can check
    # that both loops realized the same adaptive schedule.
    return final, d_final


@eqx.filter_jit
def _composable_loop_jvp_core(
    engine,
    problem,
    cfg,
    adjoint,
    has_affect,
    control_folds_theta,
    loop_primals,
    loop_tangents,
):
    """Trace-once body of the composable forward rule.

    Integrates the augmented ``(state, tangent)`` pair step-by-step inside a
    checkpointed while loop whose supported differentiation mode is exactly
    the reverse transposition that ``jacrev``-of-``jacfwd`` needs. The primal
    trajectory is recomputed alongside its tangent (no values are
    gradient-stopped); the adaptive controls remain integer/boolean state
    with no tangent, and the early-exit condition reads only the primal half
    of the carry.

    Returns ``(final, joint_final, d_final)``: the primal output, the joint
    loop's (discarded) primal half, and the tangent output. The primal output
    comes from a *separate* primal-only checkpointed loop rather than from
    the joint loop: when a custom-JVP rule returns both primal and tangent
    from one checkpointed loop carry, an outer
    reverse-mode linearization applies the primal output's cotangent twice
    (the equinox checkpointed loop's saved-residual replay retains a second
    reverse path to the joint carry's primal half), which double-counts the
    Gauss-Newton block of any criterion that reads the solution together
    with its sensitivities. Splitting the loops leaves exactly one reverse
    path to each output. The extra primal integration runs only when this
    rule fires (i.e. under differentiation) and has unbatched operands under
    an outer ``vmap``, so it runs once for all tangent seeds. Both loops are
    separately compiled programs of the same step math; the joint primal is
    returned so tests can gate that they realize the same adaptive schedule.
    """
    init_state, ts_grid, t1, theta, control0 = loop_primals
    d_init, d_ts, d_t1, d_theta, d_control0 = _canon_tangent(loop_primals, loop_tangents)

    cond_fun, body_fun = _loop_fns(
        engine, problem, cfg, has_affect, control_folds_theta, ts_grid, t1, theta, control0
    )

    def aug_body(carry):
        state, dstate = carry

        def step(state_in, ts_grid_in, t1_in, theta_in, control0_in):
            _, body_fun = _loop_fns(
                engine,
                problem,
                cfg,
                has_affect,
                control_folds_theta,
                ts_grid_in,
                t1_in,
                theta_in,
                control0_in,
            )
            return body_fun(state_in)

        new, d_new = eqx.filter_jvp(
            step,
            (state, ts_grid, t1, theta, control0),
            (dstate, d_ts, d_t1, d_theta, d_control0),
        )
        return new, _canon_tangent(new, d_new)

    def aug_cond(carry):
        return cond_fun(carry[0])

    checkpoints = getattr(adjoint, "checkpoints", None)
    checkpoints = int(checkpoints) if checkpoints is not None else None
    try:
        joint_final, d_final = checkpointed_while_loop(
            aug_cond,
            aug_body,
            (init_state, d_init),
            max_steps=int(cfg.max_steps),
            checkpoints=checkpoints,
        )
    except TypeError as exc:
        if "custom_vjp" in str(exc):
            raise ValueError(_COMPOSABLE_NESTED_FORWARD_ERROR) from exc
        raise
    final = checkpointed_while_loop(
        cond_fun,
        body_fun,
        init_state,
        max_steps=int(cfg.max_steps),
        checkpoints=checkpoints,
    )
    return final, joint_final, d_final
