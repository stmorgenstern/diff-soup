"""Canonical differentiation-strategy objects shared across solver layers.

An adjoint object is static solver configuration. It selects the nonlinear
implicit-function-theorem direction and, for a time integration, the loop or
backend differentiation mechanism. Compatibility modules in ``nonlinear`` and
``timestepping`` re-export these classes; the definitions live only here so
``isinstance`` checks and extension predicates see one class identity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Optional

import equinox as eqx
from jaxtyping import PyTree

__all__ = [
    "AbstractAdjoint",
    "ImplicitAdjoint",
    "ForwardAdjoint",
    "CheckpointAdjoint",
    "BacksolveAdjoint",
]


class AbstractAdjoint(eqx.Module):
    """Base class of the strategies that decide how derivatives flow through a solve.

    A strategy passed to `JNDF`, `JBDF`, or `JIDA` must report `is_forward_only` or `uses_online_checkpoints` as `True`, or the solve raises `TypeError`.

    Attributes:
        ift_mode: Direction of the implicit-function derivative, `"vjp"` for reverse or `"jvp"` for forward, or `None` when the strategy has none.
    """

    ift_mode: ClassVar[Optional[str]] = None

    def compositions(self, lowered_problem: Any) -> frozenset[str]:
        """Return the derivative compositions this strategy supports on a lowered problem.

        The names are `"forward"`, `"forward-over-forward"`, `"reverse"`, and `"forward-over-reverse"`. The base class supports none.

        Args:
            lowered_problem: The lowered problem the strategy will differentiate.

        Returns:
            A frozenset of composition names.
        """
        del lowered_problem
        return frozenset()

    @property
    def uses_online_checkpoints(self) -> bool:
        """Whether a time loop uses online checkpointed replay."""
        return False

    @property
    def is_forward_only(self) -> bool:
        """Whether differentiation runs forward through the time loop."""
        return False

    @property
    def enables_reverse_replay(self) -> bool:
        """Whether a time loop must allocate reverse-replay state."""
        return not self.is_forward_only

    def apply(
        self,
        primal_fn: Callable[..., PyTree],
        rewrite_fn: Optional[Callable[..., Any]],
        inputs: PyTree,
        tags: frozenset,
    ) -> PyTree:
        """Run `primal_fn` on `inputs` under this strategy.

        The base class calls `primal_fn(inputs)`. `ImplicitAdjoint` and `ForwardAdjoint` also pass `ift_mode`, `linear_solver`, and `tags` as keywords.

        Args:
            primal_fn: Function computing the result from `inputs`.
            rewrite_fn: Ignored by the built-in strategies.
            inputs: Pytree handed to `primal_fn`.
            tags: Frozenset of tags that the implicit strategies pass on.

        Returns:
            The value `primal_fn` returns.
        """
        del rewrite_fn, tags
        return primal_fn(inputs)


class _IFTAdjointMixin(AbstractAdjoint):
    """Shared nonlinear IFT routing for forward and reverse strategies."""

    def apply(
        self,
        primal_fn: Callable[..., PyTree],
        rewrite_fn: Optional[Callable[..., Any]],
        inputs: PyTree,
        tags: frozenset,
    ) -> PyTree:
        """Run the IFT primal selected by this class's ``ift_mode``.

        Args:
            primal_fn: Nonlinear runner accepting ``ift_mode``,
                ``linear_solver``, and ``tags`` keyword arguments.
            rewrite_fn: Residual rewrite owned by ``primal_fn``.
            inputs: Differentiable nonlinear inputs.
            tags: Operator structure tags forwarded to the linear solve.

        Returns:
            The output pytree from the IFT-ruled primal.
        """
        del rewrite_fn
        return primal_fn(
            inputs,
            ift_mode=type(self).ift_mode,
            linear_solver=getattr(self, "linear_solver", None),
            tags=tags,
        )


class ImplicitAdjoint(_IFTAdjointMixin):
    """Reverse-mode differentiation at a converged root, by the implicit function theorem.

    It is the default `adjoint` of the nonlinear solvers, and `DiffraxStepper` maps it to `diffrax.ImplicitAdjoint`. `JNDF`, `JBDF`, and `JIDA` refuse it.

    Args:
        linear_solver: Linear solver passed on to the solve, ignored by the built-in solvers.
    """

    ift_mode: ClassVar[str] = "vjp"
    linear_solver: Optional[Any] = None

    def compositions(self, lowered_problem: Any) -> frozenset[str]:
        """Declare first-order reverse differentiation for an implicit adjoint."""
        del lowered_problem
        return frozenset({"reverse"})


class ForwardAdjoint(_IFTAdjointMixin):
    """Forward-mode differentiation through a solve, the default of `JNDF`, `JBDF`, and `JIDA`.

    It supports `jax.jvp`, `jax.jacfwd`, and forward over forward. Reverse mode needs [CheckpointAdjoint][diff_soup.CheckpointAdjoint].

    Args:
        linear_solver: Linear solver passed on to the solve, ignored by the built-in solvers.
        composable: If `True`, reverse mode over the forward sensitivities works, as in `jax.jacrev(jax.jacfwd(f))`, and forward over forward is refused.
        checkpoints: Number of states stored for reverse mode over the composable forward mode, chosen from `max_steps` when `None`.
    """

    ift_mode: ClassVar[str] = "jvp"
    linear_solver: Optional[Any] = None
    composable: bool = False
    checkpoints: Optional[int] = None

    @property
    def is_forward_only(self) -> bool:
        """Return ``True`` to select forward differentiation."""
        return True

    def compositions(self, lowered_problem: Any) -> frozenset[str]:
        """Declare forward modes supported by this forward sensitivity rule.

        The composable rule supports reverse-over-forward differentiation of
        sensitivities, but that composition is not an optimizer Hessian mode.
        Its direct reverse and second forward derivatives remain unsupported.
        """
        del lowered_problem
        if self.composable:
            return frozenset({"forward"})
        return frozenset({"forward", "forward-over-forward"})


class CheckpointAdjoint(AbstractAdjoint):
    """Reverse-mode differentiation through a solve by checkpointed replay.

    Pass `adjoint=CheckpointAdjoint()` to use `jax.grad` or `jax.vjp` on a solve. It cannot be combined with `SaveAt(steps=...)`.

    Args:
        checkpoints: Number of states the forward pass stores for the backward replay, chosen from `max_steps` when `None`.
    """

    checkpoints: Optional[int] = None

    @property
    def uses_online_checkpoints(self) -> bool:
        """Return ``True`` to select the checkpointed replay loop."""
        return True

    def compositions(self, lowered_problem: Any) -> frozenset[str]:
        """Declare reverse modes supported by checkpointed replay.

        A precision policy that explicitly changes the reverse sweep binds an
        explicit transpose rule. That rule provides first-order gradients but
        cannot itself be differentiated by the outer forward pass.
        """
        precision = getattr(lowered_problem, "precision", None)
        has_reverse_policy = (
            getattr(precision, "adjoint_compute_dtype", None) is not None
            or getattr(precision, "reverse_sensitivity_solve_mode", None) is not None
        )
        modes = {"reverse"}
        if not has_reverse_policy:
            modes.add("forward-over-reverse")
        return frozenset(modes)


class BacksolveAdjoint(AbstractAdjoint):
    """Continuous ODE-of-the-adjoint strategy for ``DiffraxStepper``.

    Native engines reject this strategy because they do not expose
    the continuous adjoint system that Diffrax integrates.
    """

    def compositions(self, lowered_problem: Any) -> frozenset[str]:
        """Declare first-order reverse differentiation for Diffrax backsolve."""
        del lowered_problem
        return frozenset({"reverse"})
