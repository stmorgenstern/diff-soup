"""How a numeric problem's `(x, args)` closures meet code that calls them with `x` alone.

A closure of an [OptimizationProblem][diff_soup.optimization.OptimizationProblem] takes
`args` last. Its properties hand out the closure bound to the problem's own `args`, and a
lowering whose closures never read `args` wraps them so they accept and drop it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Bound:
    """A closure with its trailing `args` bound: `Bound(fn, args)(*operands)` is `fn(*operands, args)`.

    Two bindings of the same closure to the same `args` object are equal and hash alike, so
    `jax.jit` of a property read twice reuses one trace. Attributes the closure carries,
    such as a tiled Jacobian's `at_tile`, read through.
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

    def __eq__(self, other: object) -> bool:
        """Equal when both bind the same closure to the same `args` object."""
        return isinstance(other, Bound) and other.fn is self.fn and other.args is self.args

    def __hash__(self) -> int:
        """The hash of the closure's and the `args` object's identities."""
        return hash((id(self.fn), id(self.args)))

    def __repr__(self) -> str:
        return f"Bound({self.fn!r})"


class IgnoringArgs:
    """A closure of the operands alone, made to accept and drop a trailing `args`.

    The lowerings whose closures never read `args` (an `InferenceProblem`'s, those
    `from_value` and `from_residual` build) store their closures in this form. Attributes
    the closure carries read through.
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


__all__ = ["Bound", "IgnoringArgs", "ignoring_args"]
