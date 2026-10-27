"""Solve contexts: owners of native linear-solve factor storage.

A backend that holds factors outside the arrays of a program (a host or
device factor slot) needs an owner for that storage and an identity that
separates independent logical solves running in one process.
[SolveContext][diff_soup.linear.SolveContext] is both: a cheap, hashable
object created per logical solve (one per engine instance, one per forward
or adjoint pass, ...) whose integer ``context_id`` the native backends carry
with every factor they build, and whose release frees that storage.

[DEFAULT_CONTEXT][diff_soup.linear._context.DEFAULT_CONTEXT] is the context
the solver engines factor in: their native factor storage lives there for
the life of the process.

This module imports nothing from the rest of ``diff_soup.linear``, so every
module of the package can import it without a cycle.
"""

from __future__ import annotations

import itertools
import threading
from typing import Any, Optional

import jax

from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import dispatch

#: Largest context id: the native backends carry the id as an ``int32``.
_MAX_CONTEXT_ID = 2**31 - 1

_CONTEXT_IDS = itertools.count()
_CONTEXT_LOCK = threading.Lock()


class FactorContractError(ConfigurationError):
    """Raised for a factor built or used in a way its backend does not allow.

    It is a `ConfigurationError`. Examples are a released `SolveContext`, values of the
    wrong shape, and a KLU factor in a function compiled for a GPU.
    """


class SolveContext:
    """The owner of the factors that KLU, SuperLU and cuDSS store outside JAX arrays.

    Pass it as `ctx=` to `factor` or `empty_factor`. Releasing it, or leaving its `with`
    block, frees every factor it owns, and their solves raise from then on.

    Args:
        name: A label shown in its `repr`.

    !!! example

        ```python
        with SolveContext() as ctx:
            f = factor(analysis, values, ctx=ctx)
            x, f = solve(f, b)
        ```
    """

    __slots__ = ("context_id", "name", "_backends", "_released", "_lock")

    def __init__(self, name: Optional[str] = None):
        """Create an open context with a fresh process-unique identity.

        Construction is cheap: no native storage exists until a backend
        factors in the context.

        Args:
            name: Optional label used in messages and ``repr``.

        Raises:
            FactorContractError: If the process has created ``2**31``
                contexts, the most an ``int32`` id can name.
        """
        with _CONTEXT_LOCK:
            context_id = next(_CONTEXT_IDS)
        if context_id > _MAX_CONTEXT_ID:
            raise FactorContractError(
                f"SolveContext ids are int32 and this process has used all {_MAX_CONTEXT_ID + 1} "
                "of them; reuse open contexts instead of creating one per call."
            )
        self.context_id = context_id
        self.name = name
        self._backends: list = []
        self._released = False
        self._lock = threading.Lock()

    def __hash__(self) -> int:
        """Hash by the integer identity."""
        return hash(("diff_soup.SolveContext", self.context_id))

    def __eq__(self, other: object) -> bool:
        """Two contexts are equal only when they have the same ``context_id``."""
        if not isinstance(other, SolveContext):
            return NotImplemented
        return self.context_id == other.context_id

    def __enter__(self) -> SolveContext:
        """Return the context itself."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release the context."""
        self.release()

    def __repr__(self) -> str:
        """Show the id, the label and whether the context is released."""
        label = f", {self.name!r}" if self.name is not None else ""
        state = "released" if self._released else "open"
        return f"SolveContext(id={self.context_id}{label}, {state})"

    def __copy__(self) -> SolveContext:
        """A context is an identity, so a copy is the context itself."""
        return self

    def __deepcopy__(self, memo: dict) -> SolveContext:
        """A context is an identity, so a deep copy is the context itself."""
        return self

    def __reduce__(self) -> tuple:
        """Refuse pickling: a context owns factor storage of this process only.

        Raises:
            TypeError: Always.
        """
        raise TypeError(
            f"{self!r} owns factor storage of this process and cannot be pickled; build "
            "factors that travel between processes without a SolveContext, or factor again "
            "in the receiving process."
        )

    @property
    def released(self) -> bool:
        """Whether [release][diff_soup.linear.SolveContext.release] has run."""
        return self._released

    def check_open(self) -> None:
        """Raise unless the context is still open.

        Raises:
            FactorContractError: If the context has been released.
        """
        if self._released:
            raise FactorContractError(
                f"{self!r} has been released, so its factors no longer exist; build the "
                "factor again in an open SolveContext."
            )

    def register(self, backend: Any) -> None:
        """Record that ``backend`` may store factors under this context.

        Called by [factor][diff_soup.linear.factor]; registering a backend
        twice has no effect.

        Args:
            backend: The configured backend.

        Raises:
            FactorContractError: If the context has been released.
        """
        with self._lock:
            self.check_open()
            if not any(b is backend or b == backend for b in self._backends):
                self._backends.append(backend)

    def release(self) -> None:
        """Free every factor this context owns.

        A context that holds stored factors first waits for all running JAX work in the process.
        A second call does nothing.

        Raises:
            FactorContractError: For the default context of the solvers, which is never released.
        """
        if self is DEFAULT_CONTEXT:
            raise FactorContractError(
                "DEFAULT_CONTEXT holds the solver engines' factor storage for the life of the "
                "process and is never released; create a SolveContext() of your own to release "
                "its factors."
            )
        with self._lock:
            if self._released:
                return
            backends, self._backends = self._backends, []
            self._released = True
        if any(backend.factor_residency != "arrays" for backend in backends):
            wait_for_dispatched_programs()
        for backend in backends:
            backend._release(self)

    def resident_bytes(self) -> int:
        """The bytes of stored factors this context owns.

        Returns:
            The total, `0` for factors held in arrays and after release.
        """
        with self._lock:
            backends = tuple(self._backends)
        return int(sum(int(backend._resident_bytes(self)) for backend in backends))


def wait_for_dispatched_programs() -> None:
    """Block until every program dispatched so far has finished, failed ones included.

    A program of the linear-solver contract carries no JAX effect, so
    ``jax.effects_barrier`` does not wait for it; its results are arrays
    whose readiness marks the end of its calls, so this barrier waits for
    every live array. It then waits for the tokens JAX keeps for programs
    with effects (a host callback, for example), which covers such a program
    whose results were already dropped, and clears them as
    ``jax.effects_barrier`` does after a wait without errors.

    A failed program counts as finished and its error is not raised here: it
    reaches whoever reads that program's results. ``jax.effects_barrier``
    instead raises the error of the first failed token it meets, leaves the
    remaining tokens unwaited and keeps the failed one, so every later
    barrier would raise that error again. An array deleted meanwhile is
    skipped as well.
    """
    for array in jax.live_arrays():
        try:
            array.block_until_ready()
        except Exception:  # another program's failure, or an array deleted meanwhile
            continue
    tokens = dispatch.runtime_tokens
    for token in (*tokens.current_tokens.values(), *tokens.output_runtime_tokens.values()):
        try:
            token.block_until_ready()
        except Exception:  # a failed program has finished; nothing left to wait for
            continue
    tokens.clear()


#: The context the solver engines factor in. Their native factor storage
#: (host slots of KLU and SuperLU, device slots of cuDSS) lives here for the
#: life of the process: it has no release path, and ``release`` refuses it.
DEFAULT_CONTEXT = SolveContext(name="default")


__all__ = [
    "DEFAULT_CONTEXT",
    "FactorContractError",
    "SolveContext",
    "wait_for_dispatched_programs",
]
