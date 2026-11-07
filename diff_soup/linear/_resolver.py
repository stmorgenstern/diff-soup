"""One backend chooser for every consistent-init and engine Jacobian dispatch.

A single [BackendResolver][] maps a requested backend name plus the
problem's block size and device platform to a concrete backend NAME, and
[resolve_backend][] turns a concrete name into the process's one instance of
that backend, so every layer that selects a backend by name shares it.

The policy: ``dense`` is used only for tiny blocks (at or below
[_DENSE_BLOCK_CEIL][], where a dense LU is cheap. Above that a
sparse backend is mandatory — cuDSS on GPU, KLU on CPU — because a dense
``(n, n)`` Jacobian costs ``O(n^2)`` memory and OOMs at FEM scale. When the
mandatory sparse backend is unavailable the resolver RAISES with a build or
override hint instead of silently degrading to that dense cost; pass
``backend="dense"`` explicitly to accept it.

The ceiling is a memory-and-parity policy, not a speed optimum. On CPU the
dense/KLU crossover for banded Jacobians lies near a few hundred rows (every
native KLU call carries a fixed per-call cost, so a dense LU is faster below
that), yet ``auto`` keeps the sparse route from 11 rows so that a CPU
development run exercises the same declared-topology path as a GPU or
production run, and so that the released sparse-by-default behavior holds at
every size. Pass ``backend="dense"`` for a small CPU problem where the
per-call cost dominates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from diff_soup.linear import KLU, CuDSS, Dense, LinearSolver, SuperLU
from diff_soup.linear._availability import cudss_available
from diff_soup.linear.klu_backend import klu_available
from diff_soup.linear.superlu_backend import require_superlu
from diff_soup.utils.placement import array_platform, is_batch_tracer

# Dense is used only at or below this many unknowns. Above it (block_size > 10)
# a sparse backend is mandatory on the ``auto`` path: every real problem is far
# larger, so it always routes sparse; only tiny blocks stay dense. The value is
# deliberately below the measured CPU speed crossover (see the module docstring).
_DENSE_BLOCK_CEIL = 10


@dataclass(frozen=True)
class BackendResolver:
    """Choose a linear-backend NAME from problem size and platform.

    ``backend`` is the requested name (``"auto"``, ``"dense"``, ``"klu"``,
    ``"superlu"``, ``"cudss"``). A concrete name is honored verbatim.
    ``"auto"`` resolves by size and platform: dense at or below
    [_DENSE_BLOCK_CEIL][], otherwise cuDSS on GPU or KLU on CPU. When the
    mandatory sparse backend is unavailable [choose][] raises rather than
    silently returning a dense ``O(n^2)`` solve — pass ``backend="dense"`` to
    opt into the dense cost.

    A batch of independent systems never changes the choice: every backend
    serves one through the factor contract, as its ``vmap_factors`` declares
    (one batched factorization on Dense and cuDSS, lane by lane on KLU).
    """

    backend: str = "auto"

    def choose(self, *, block_size, platform, batched=False, needs_ad_rules=False) -> str:
        """Return the concrete backend name for a block of ``block_size`` rows.

        Args:
            block_size: The side length of the state Jacobian block being solved.
            platform: ``"cpu"`` or ``"gpu"`` — the device the solve runs on.
            batched: Ignored. A batch of independent systems runs on every
                backend through the factor contract (``vmap_factors``).
            needs_ad_rules: Ignored. Every backend differentiates through the
                factor contract's rules.

        Returns:
            One of ``"dense"``, ``"klu"``, ``"superlu"``, ``"cudss"``.

        Raises:
            ValueError: For an unknown requested backend name.
            RuntimeError: When the mandatory sparse backend for a large block is
                not available on this host.
        """
        del batched, needs_ad_rules
        if self.backend in ("dense", "klu", "superlu", "cudss"):
            return self.backend
        if self.backend != "auto":
            raise ValueError(
                f"unknown linear backend {self.backend!r}; expected "
                "'auto', 'cudss', 'klu', 'superlu', or 'dense'."
            )

        if int(block_size) <= _DENSE_BLOCK_CEIL:
            return "dense"
        if platform == "gpu":
            if not cudss_available():
                raise RuntimeError(
                    f"a {block_size}-row block on GPU needs the cuDSS extension, which is "
                    "not available. Install the cuDSS build, or pass backend='dense' to "
                    "accept the dense O(n^2) cost."
                )
            return "cudss"
        if not klu_available():
            raise RuntimeError(
                f"a {block_size}-row block on CPU needs the KLU extension, which is not "
                "built. Build it (cmake -B build && cmake --build build -j under "
                "diff_soup/_internal/klu_ffi/), or pass backend='dense'."
            )
        return "klu"


def is_batched(*sample_arrs) -> bool:
    """Whether any leaf of the given pytrees is a ``vmap`` batch tracer.

    [BackendResolver.choose][diff_soup.linear._resolver.BackendResolver.choose]
    does not depend on the answer.
    """
    return any(
        is_batch_tracer(leaf)
        for sample in sample_arrs
        for leaf in jax.tree_util.tree_leaves(sample)
    )


#: The backend each concrete name resolves to, one instance per process: a stable identity keeps
#: an engine's static configuration cache-stable, so two engines built from the same
#: configuration share one compiled program. Backends hold no mutable state; native handles,
#: factor slots and analyses live in the linear layer's registries.
_SINGLETONS: dict[str, LinearSolver] = {}


def resolve_backend(linear: Any) -> LinearSolver:
    """Resolve a CONCRETE linear-backend name to a process-global instance.

    ``"dense"`` returns the dense backend; ``"klu"``, ``"superlu"`` and
    ``"cudss"`` return their sparse-direct backends at their default settings.
    ``"superlu"`` and ``"cudss"`` raise when their native extension is
    unavailable — neither silently falls back to a dense ``O(n^2)`` solve.
    ``"auto"`` is NOT resolved here: the size- and platform-aware
    [BackendResolver][diff_soup.linear._resolver.BackendResolver] must pick a concrete
    name upstream, so a bare ``"auto"`` (or any unknown name) reaching this
    function is a bug and raises. The same name always returns the same instance.
    A configured [LinearSolver][diff_soup.linear.LinearSolver] instance (for
    example ``KLU(ordering="colamd")`` or ``Krylov(CuDSS())``) is returned as
    it is.

    Args:
        linear: A concrete backend name (``"cudss"``, ``"klu"``, ``"superlu"``,
            ``"dense"``), or a backend instance.

    Returns:
        The backend singleton for ``linear``, or the instance itself.

    Raises:
        RuntimeError: When ``"superlu"`` or ``"cudss"`` is requested but its
            native extension is not available on this host.
        ValueError: When ``linear`` is ``"auto"`` or an unrecognized name.
    """
    if isinstance(linear, LinearSolver):
        return linear
    name = str(linear).lower()
    inst = _SINGLETONS.get(name)
    if inst is not None:
        return inst
    if name == "klu":
        inst = KLU()
    elif name == "superlu":
        require_superlu("linear='superlu'")
        inst = SuperLU()
    elif name == "cudss":
        if CuDSS is None:
            raise RuntimeError(
                "the cuDSS backend was requested but its native extension is not "
                "available on this host; install the cuDSS build, or select 'klu' "
                "(CPU) or 'dense'."
            )
        inst = CuDSS()
    elif name == "dense":
        inst = Dense()
    else:
        raise ValueError(
            f"resolve_backend received the non-concrete name {linear!r}; the size- and "
            "platform-aware BackendResolver must choose a concrete backend ('dense', "
            "'klu', 'superlu', or 'cudss') before this call, so 'auto' must not reach here."
        )
    _SINGLETONS[name] = inst
    return inst


__all__ = ["BackendResolver", "array_platform", "is_batched", "resolve_backend"]
