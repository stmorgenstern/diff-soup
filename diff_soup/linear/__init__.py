"""diff_soup.linear: linear-solver backends and the analyze / factor / solve contract.

Public surface: ``LinearSolver`` (abstract base), ``SparseTopology``, the
backends ``Dense`` / ``KLU`` / ``SuperLU`` / ``CuDSS``, and the typed option
object ``CuDSSOptions``. ``KLU`` wraps SuiteSparse KLU via the ``_native_klu``
extension and ``SuperLU`` wraps SuperLU_MT via ``_native_superlu``; both run
on the host CPU.

The analyze / factor / solve contract is a set of functions over any backend that
implements it: ``analyze`` (host, memoised by content, returns an
``Analysis``), ``factor`` (traced, returns a ``Factor`` pytree), ``solve``
(traced, linear in the right-hand side, with ``transpose``), ``refactor``
(the per-lane gated update), ``empty_factor`` (a factor that holds no
factorization yet, the initial value of a loop carry that ``refactor``
fills), ``invalidate`` (marks a factor's native storage stale),
``after`` (links the order words of factors that name one storage) and
``solve_checked`` (returns a ``SolveReport`` with a ``SolveStatus`` code).
Every operation on a factor returns the factor to use next
(``x, f = solve(f, b)``), and passing it on orders the operations.
``refine`` improves the solution of a reduced-precision factor against
full-precision values and reports the outcome in a ``RefineReport``.
``SolveContext`` owns native factor storage, ``Tolerance`` is the stopping
rule of an inexact solve, and ``FactorContractError`` reports a factor built
or used outside the contract. ``LinearOperator`` describes an operator by its
action (``mv``, ``rmv``, shape, dtype, structure tags), built from assembled
values by ``operator_from_values`` or from a residual's Jacobian-vector
product by ``operator_from_jvp``; an iterative backend's solve works with it.

The cuDSS backend imports the native ``_native_solver`` extension at module
load; on a host without the built ``.so`` (gitignored, per-worktree) or
without a GPU the import fails and [CuDSS][diff_soup.linear.CuDSS] is
``None``, so the dense path stays usable. An extension that is present but fails to start is
reported with a ``RuntimeWarning`` naming the error, and ``CuDSS`` is ``None``.

``DenseBackend`` and ``CuDSSBackend`` (aliases) and ``cudss_available`` are
importable from this package but are not part of the public ``__all__``.

Structured backends serve operators with a known structure in pure JAX:
``BlockTridiagonal`` (block cyclic reduction of block-tridiagonal chains),
``MemberBlock`` (banded member blocks and a retained Schur complement,
described by a ``MemberBlockLayout``) and ``Woodbury`` (an operator with a
low-rank term ``U W^T``, declared by a ``LowRankTerm``, over a factor of the
base operator from any exact backend).

Iterative solves: ``Krylov`` is the inexact backend, restarted GMRES on the
current operator preconditioned by a held factor of an exact backend, with
its convergence report in ``solve_checked``; ``gmres`` runs the same
iteration on a ``LinearOperator`` with any preconditioner.
"""

import warnings

from diff_soup.linear._api import (
    LinearSolver,
    SparseTopology,
    after,
    analyze,
    empty_factor,
    factor,
    invalidate,
    refactor,
    solve,
    solve_checked,
)
from diff_soup.linear._availability import (
    cudss_available as cudss_available,  # host-side cuDSS probe (not public __all__)
)
from diff_soup.linear._backends import (
    KLU,
    CuDSSOptions,
)
from diff_soup.linear._dense import Dense
from diff_soup.linear._dense import DenseBackend as DenseBackend
from diff_soup.linear._factor import (
    Analysis,
    Factor,
    FactorContractError,
    SolveContext,
    SolveReport,
    SolveStatus,
    Tolerance,
)
from diff_soup.linear._operator import LinearOperator, operator_from_jvp, operator_from_values
from diff_soup.linear._refine import RefineReport, refine
from diff_soup.linear.superlu_backend import SuperLU

try:
    from diff_soup.linear._cudss._backend import CuDSS, CuDSSBackend
except ImportError:  # pragma: no cover - no built extension or CUDA plugin
    CuDSS = None
    CuDSSBackend = None
except Exception as exc:  # pragma: no cover - extension present, failed to start
    warnings.warn(
        "diff_soup's cuDSS backend is installed but failed to initialize, so it is "
        f"disabled for this process: {exc!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    CuDSS = None
    CuDSSBackend = None

from diff_soup.linear._krylov import Krylov, gmres
from diff_soup.linear._structured import (
    BlockTridiagonal,
    LowRankTerm,
    MemberBlock,
    MemberBlockLayout,
    Woodbury,
)

__all__ = [
    # Abstract base + contract types
    "LinearSolver",
    "SparseTopology",
    # Concrete backends
    "CuDSS",
    "Dense",
    "KLU",
    "SuperLU",
    # Typed option objects
    "CuDSSOptions",
    # The analyze / factor / solve contract
    "analyze",
    "factor",
    "empty_factor",
    "solve",
    "refactor",
    "invalidate",
    "after",
    "solve_checked",
    "refine",
    "RefineReport",
    "Analysis",
    "Factor",
    "SolveContext",
    "SolveReport",
    "SolveStatus",
    "Tolerance",
    "FactorContractError",
    # Matrix-free operators of inexact solves
    "LinearOperator",
    "operator_from_values",
    "operator_from_jvp",
    # Structured backends
    "BlockTridiagonal",
    "LowRankTerm",
    "MemberBlock",
    "MemberBlockLayout",
    "Woodbury",
    # Iterative solves
    "Krylov",
    "gmres",
]
