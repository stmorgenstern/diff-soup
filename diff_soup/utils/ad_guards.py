"""Trace-time recognition of the differentiation applied over a solve.

The implicit-root derivative boundaries of a solve (the accepted step root,
the consistent initialization, the startup slope) each bind one forward-mode
``custom_jvp`` entry, whose tangent solve JAX transposes for reverse mode and
differentiates again for ``jax.hessian``; they need no guard.
The helpers here serve the call sites that do depend on the differentiation
reaching their traced inputs. :func:`reject_forward_over_reverse` raises the
public statement when a forward push reaches an explicit reverse-mode
(``custom_vjp``) entry, which has no forward rule.
:func:`reverse_linearization_observed` reports whether a reverse-mode
linearization reaches the inputs. Both are trace-time only: they read tracer
types and change nothing in the compiled program.
"""

from __future__ import annotations

import jax

from diff_soup._jax_compat import ad as _ad
from diff_soup._jax_compat import partial_eval as _partial_eval

# The checks below recognise the JVP and partial-evaluation tracers through
# the public ``jax.interpreters`` modules. If a JAX release stops exporting
# them, they degrade gracefully: nothing is observed, and forward
# differentiation over a reverse-mode entry then surfaces jax's own
# trace-time error instead of the public message.
_JVP_TRACER = getattr(_ad, "JVPTracer", None)
_JAXPR_TRACER = getattr(_partial_eval, "JaxprTracer", None)


def _is_linearize_tracer(leaf) -> bool:
    """Whether ``leaf`` is the tracer of a reverse-mode linearization.

    JAX linearizes directly: reverse mode wraps a value in a linearize
    tracer that carries the primal and a staged tangent, and that class is
    not exported, so the check is structural: a tracer with ``primal`` and
    ``tangent`` that is not the forward-mode JVP tracer. (Older releases
    expressed the same linearization as a JVP tracer with a symbolic
    partial-evaluation tangent, handled in ``_jvp_tangent_is_known``.)
    """
    return (
        isinstance(leaf, jax.core.Tracer)
        and not (_JVP_TRACER is not None and isinstance(leaf, _JVP_TRACER))
        and hasattr(leaf, "primal")
        and hasattr(leaf, "tangent")
    )


_FORWARD_OVER_REVERSE_ERROR = (
    "Forward-mode differentiation (jax.jvp / jax.jacfwd, including the "
    "outer forward pass of jax.hessian) was applied over {entry}, a "
    "reverse-mode derivative entry with an explicit transpose factorization. "
    "That entry supports first-order reverse-mode gradients only: its "
    "transpose path cannot be forward-differentiated. Differentiate it with "
    "jax.grad / jax.vjp alone, or bind its forward-mode counterpart, which "
    "jax.hessian composes over."
)


def _jvp_tangent_is_known(tangent) -> bool:
    """Whether a JVP tangent carries known values (a concrete forward push).

    Reverse-mode linearization also runs a JVP trace, but with *symbolic*
    (unknown) partial-eval tangents that the transpose consumes; a
    ``jax.jacfwd`` / ``jax.jvp`` push carries concrete tangent values (or
    batch wrappers of them). Unknown wrapper types count as known so an
    internals drift fails loud rather than letting the composition proceed
    into a trace error.
    """
    depth = 0
    while depth < 8:
        if not isinstance(tangent, jax.core.Tracer):
            return tangent is not None
        if _JAXPR_TRACER is not None and isinstance(tangent, _JAXPR_TRACER):
            return tangent.pval.is_known()
        inner = getattr(tangent, "val", None)
        if inner is None:
            return True
        tangent = inner
        depth += 1
    return True


def _differentiation_tracers(values):
    """Yield ``(kind, beneath_reverse)`` for each differentiation tracer in ``values``.

    ``kind`` is ``"forward"`` for a JVP tracer carrying known tangent values
    (a ``jax.jvp`` / ``jax.jacfwd`` push, or a batch wrapper of one) and
    ``"reverse"`` for a reverse-mode linearization (JAX's linearize tracer,
    or a JVP tracer whose tangent is symbolic); ``beneath_reverse`` tells
    whether a reverse linearization sits above the tracer. Batch wrappers are
    unwrapped so ``vmap`` compositions are seen through, and the walk is
    bounded so an unexpected tracer nest cannot loop.
    """
    stack = [(leaf, False) for leaf in jax.tree_util.tree_leaves(values)]
    steps = 0
    while stack and steps < 256:
        leaf, beneath_reverse = stack.pop()
        steps += 1
        if not isinstance(leaf, jax.core.Tracer):
            continue
        if _is_linearize_tracer(leaf):
            # The reverse linearization itself; an outer forward push, if
            # any, sits beneath its primal.
            yield "reverse", beneath_reverse
            stack.append((leaf.primal, True))
            continue
        if isinstance(leaf, _JVP_TRACER):
            if _jvp_tangent_is_known(leaf.tangent):
                yield "forward", beneath_reverse
                stack.append((leaf.primal, beneath_reverse))
            else:
                # A symbolic tangent is a reverse linearization expressed as
                # a JVP trace; an outer forward push, if any, sits beneath
                # its primal.
                yield "reverse", beneath_reverse
                stack.append((leaf.primal, True))
            continue
        inner = getattr(leaf, "val", None)
        if inner is not None:
            stack.append((inner, beneath_reverse))


def reverse_linearization_observed(values) -> bool:
    """Whether a reverse-mode linearization reaches any traced leaf of ``values``.

    ``True`` under ``jax.grad``, ``jax.vjp``, ``jax.jacrev`` and the inner pass
    of ``jax.hessian``, whether or not a forward push is also visible; ``False``
    for an undifferentiated or purely forward-differentiated call, and inside a
    ``jax.jit`` whose outer transformation the trace cannot see. It reads
    tracer types at trace time only.
    """
    if _JVP_TRACER is None:
        return False
    return any(kind == "reverse" for kind, _ in _differentiation_tracers(values))


def reject_forward_over_reverse(
    values,
    entry: str,
    *,
    message: str | None = None,
) -> None:
    """Raise the public error when a forward push reaches a reverse binding.

    Called with traced inputs at the sites that bind an explicit reverse-mode
    (``custom_vjp``) derivative entry.
    A forward differentiation composed over a reverse binding (the
    ``jax.hessian`` shape) shows up there as JVP tracers with concrete
    tangent values; rejecting at the binding states the contract instead of
    failing deep inside the transpose machinery. First-order reverse traces
    carry only the linearization the transpose consumes, and the check is
    trace-time only, so it adds nothing to the compiled program.

    ``entry`` names the reverse-mode entry in the default text; ``message``
    replaces that text for callers whose restriction has a different cause.
    """
    if _JVP_TRACER is None:
        return
    for kind, _ in _differentiation_tracers(values):
        if kind == "forward":
            raise ValueError(
                _FORWARD_OVER_REVERSE_ERROR.format(entry=entry) if message is None else message
            )


__all__ = ["reject_forward_over_reverse"]
