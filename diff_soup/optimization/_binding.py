"""How a numeric problem's `(x, args)` closures are bound, wrapped and derived.

A closure of an [OptimizationProblem][diff_soup.optimization.OptimizationProblem] takes
`args` last. Its properties hand out the closure bound to the problem's own `args`; a
lowering whose closures never read `args` wraps them so they accept and drop it; and a
closure derived from others (a value from a value and gradient, a penalty folded in) is a
[Derived][diff_soup.optimization._binding.Derived], equal to another derived by the same
rule from equal inputs. The closures are static fields of the problem, so two lowerings
of one problem are equal and `filter_jit` reuses one compile, with no cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Bound:
    """A closure with its trailing `args` bound: `Bound(fn, args)(*operands)` is `fn(*operands, args)`.

    Attributes the closure carries, such as a tiled Jacobian's `at_tile`, read through. Each
    property read binds afresh, so `jax.jit(problem.value)` read twice traces twice; hold one
    read to reuse its trace.
    """

    __slots__ = ("fn", "args", "__weakref__")

    def __init__(self, fn: Callable, args: Any):
        """Bind `args` as the last argument of `fn`."""
        self.fn = fn
        self.args = args

    def __call__(self, *operands: Any) -> Any:
        """`fn(*operands, args)`."""
        return self.fn(*operands, self.args)

    def __getattr__(self, name: str) -> Any:
        """Read an attribute the closure carries."""
        if name in Bound.__slots__:
            raise AttributeError(name)
        return getattr(self.fn, name)

    def __repr__(self) -> str:
        return f"Bound({self.fn!r})"


class IgnoringArgs:
    """A closure of the operands alone, made to accept and drop a trailing `args`.

    The lowerings whose closures never read `args` (an `InferenceProblem`'s, those
    `from_value` and `from_residual` build) store their closures in this form. Two wrappers
    of one closure are equal. Attributes the closure carries read through.
    """

    __slots__ = ("fn", "__weakref__")

    def __init__(self, fn: Callable):
        """Wrap `fn`, a function of the operands without `args`."""
        self.fn = fn

    def __call__(self, *operands_and_args: Any) -> Any:
        """`fn` of every argument but the last."""
        return self.fn(*operands_and_args[:-1])

    def __getattr__(self, name: str) -> Any:
        """Read an attribute the closure carries."""
        if name in IgnoringArgs.__slots__:
            raise AttributeError(name)
        return getattr(self.fn, name)

    def __eq__(self, other: object) -> bool:
        """Equal when both wrap the same closure."""
        return isinstance(other, IgnoringArgs) and other.fn is self.fn

    def __hash__(self) -> int:
        """The hash of the closure's identity."""
        return hash(id(self.fn))

    def __repr__(self) -> str:
        return f"IgnoringArgs({self.fn!r})"


def ignoring_args(fn: Callable | None) -> Callable | None:
    """`fn` made to accept and drop a trailing `args`; `None` stays `None`."""
    if fn is None or isinstance(fn, IgnoringArgs):
        return fn
    return IgnoringArgs(fn)


class Derived:
    """A closure built by a module-level `rule` from `inputs`: `Derived(rule, *inputs)(*operands)` is `rule(*inputs, *operands)`.

    Two are equal when their rules are the same function and their inputs are equal, so a
    closure derived again from the same closures is the same static field. An input is a
    closure, a `Held` object, or a hashable configuration value, never an array.
    """

    __slots__ = ("rule", "inputs", "__weakref__")

    def __init__(self, rule: Callable, *inputs: Any):
        """Derive a closure by `rule` from `inputs`."""
        self.rule = rule
        self.inputs = inputs

    def __call__(self, *operands: Any) -> Any:
        """`rule(*inputs, *operands)`."""
        return self.rule(*self.inputs, *operands)

    def __eq__(self, other: object) -> bool:
        """Equal when derived by the same rule from equal inputs."""
        return (
            isinstance(other, Derived) and other.rule is self.rule and other.inputs == self.inputs
        )

    def __hash__(self) -> int:
        """The hash of the rule and the inputs."""
        return hash((self.rule, self.inputs))

    def __repr__(self) -> str:
        return f"Derived({self.rule.__name__})"


class Held:
    """A value held by identity: hashable, and not a pytree whatever it holds.

    A problem's model, its reconstruction or its regularization can hold arrays; held by
    identity they keep the problem's static data hashable and array-free, and two problems
    built from the same object compare equal.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        """Hold `value` by identity, outside the pytree."""
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)

    def __eq__(self, other: object) -> bool:
        """Equal when both hold the same object."""
        return isinstance(other, Held) and other.value is self.value

    def __hash__(self) -> int:
        """The held object's identity."""
        return id(self.value)


def hold(value: Any) -> Held:
    """`value` in a `Held`, unless it is one already."""
    return value if isinstance(value, Held) else Held(value)


__all__ = ["Bound", "Derived", "Held", "IgnoringArgs", "hold", "ignoring_args"]
