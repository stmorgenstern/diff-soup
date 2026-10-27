# @spec.equation / @spec.equations decorators.
"""@spec.equation / @spec.equations decorators.

Registers an [EquationSpec][diff_soup.problem._spec.specs.EquationSpec]
(single-write) or
[MultiEquationSpec][diff_soup.problem._spec.specs.MultiEquationSpec]
(multi-write) on the parent ``ProblemSpec``
as a side effect; the decorated function is returned unchanged so
callers can still invoke it directly for testing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

from .specs import EquationSpec, MultiEquationSpec


class _EquationDecorator:
    """Bound to ``spec.equation``.  Calling it returns the actual decorator."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def __call__(
        self,
        *,
        writes: str,
        reads: Sequence[str],
        name: Optional[str] = None,
    ):
        def decorator(fn: Callable) -> Callable:
            eq_name = name if name is not None else fn.__name__
            self._spec._equations.append(
                EquationSpec(
                    writes=writes,
                    reads=tuple(reads),
                    fn=fn,
                    name=eq_name,
                )
            )
            return fn

        return decorator


class _MultiEquationDecorator:
    """Bound to ``spec.equations`` (plural).  Returns a decorator.

    The decorated callable must accept a ``q`` accessor and return a
    ``dict`` whose keys cover every name listed in ``writes``.  At
    lower time the IR runs the closure once per residual call and
    distributes its outputs into the assembled residual; this avoids
    the trace-time and HLO-time redundancy of registering multiple
    single-write equations that all call the same per-cell kernel.
    """

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def __call__(
        self,
        *,
        writes: Sequence[str],
        reads: Sequence[str],
        name: Optional[str] = None,
    ):
        if isinstance(writes, str) or len(writes) < 2:
            raise ValueError(
                "spec.equations(...) requires writes= as a tuple/list of >=2 "
                "field names; use spec.equation (singular) for single-write."
            )

        def decorator(fn: Callable) -> Callable:
            eq_name = name if name is not None else fn.__name__
            self._spec._multi_equations.append(
                MultiEquationSpec(
                    writes=tuple(writes),
                    reads=tuple(reads),
                    fn=fn,
                    name=eq_name,
                )
            )
            return fn

        return decorator
