"""Structured backends of the linear-solver contract, in pure JAX.

- [BlockTridiagonal][diff_soup.linear.BlockTridiagonal]: block cyclic
  reduction of block-tridiagonal chains, ragged chains included.
- [Woodbury][diff_soup.linear.Woodbury] with
  [LowRankTerm][diff_soup.linear.LowRankTerm]: an operator ``A_b + U W^T``
  solved through a factor of ``A_b`` from any exact backend.
- [MemberBlock][diff_soup.linear.MemberBlock] with
  [MemberBlockLayout][diff_soup.linear.MemberBlockLayout]: banded member
  blocks, their member-local low-rank terms and a retained Schur complement.

Each is a [LinearSolver][diff_soup.linear.LinearSolver] of the contract:
its factors cross ``jit``, ``scan`` and ``vmap`` as ``Factor`` pytrees, one
factor serves solves with the operator and its transpose, and derivatives
follow the phased primitives' rules.
"""

from diff_soup.linear._structured._block_tridiagonal import (
    BlockTridiagonal,
    block_tridiagonal_factor,
    block_tridiagonal_matvec,
    block_tridiagonal_solve,
)
from diff_soup.linear._structured._low_rank import LowRankTerm, Woodbury
from diff_soup.linear._structured._member_block import MemberBlock, MemberBlockLayout

__all__ = [
    "BlockTridiagonal",
    "LowRankTerm",
    "MemberBlock",
    "MemberBlockLayout",
    "Woodbury",
    "block_tridiagonal_factor",
    "block_tridiagonal_matvec",
    "block_tridiagonal_solve",
]
