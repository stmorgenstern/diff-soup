"""timestepping/diffrax_wrapper — Diffrax-backed integrator wrapper.

[DiffraxStepper][diff_soup.timestepping.diffrax_wrapper.DiffraxStepper] wraps
`diffrax.diffeqsolve` behind the shared solve protocol and returns the
shared [Solution][diff_soup.timestepping.vsvo.solution.Solution] every
engine shares.
"""

from .diffrax_wrapper import DiffraxStepper

__all__ = ["DiffraxStepper"]
