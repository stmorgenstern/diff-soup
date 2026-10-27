"""CuDSS linear-backend options and the KLU backend alias."""

from __future__ import annotations

import operator
from typing import Optional

import equinox as eqx

from diff_soup.linear.klu_backend import KLUBackend as KLU  # noqa: F401

# ---------------------------------------------------------------------------
# Typed option objects
# ---------------------------------------------------------------------------


class CuDSSOptions(eqx.Module):
    """The cuDSS settings as an `options=` object for `analyze` with `backend=CuDSS()`.

    The same settings are fields of `CuDSS`. Every field of this object, defaults included,
    overrides the backend's.

    Attributes:
        reorder: The fill-reducing ordering, `"amd"` or `"metis"`.
        hybrid_memory: `True` keeps part of the factor in host memory when it outgrows the GPU.
        ir_nsteps: The iterative refinement steps per solve, where `None` takes 0 for a float64
            factor and 1 otherwise.
    """

    reorder: str = eqx.field(static=True, default="amd")
    hybrid_memory: bool = eqx.field(static=True, default=False)
    ir_nsteps: Optional[int] = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Reject an invalid refinement count."""
        if self.ir_nsteps is None:
            return
        try:
            if isinstance(self.ir_nsteps, bool):
                raise TypeError
            ir_nsteps = operator.index(self.ir_nsteps)
        except TypeError as exc:
            raise ValueError(
                "CuDSSOptions.ir_nsteps must be None or a non-negative integer; got "
                f"{self.ir_nsteps!r}."
            ) from exc
        if ir_nsteps < 0:
            raise ValueError(
                "CuDSSOptions.ir_nsteps must be None or a non-negative integer; got "
                f"{self.ir_nsteps!r}."
            )
