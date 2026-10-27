"""The periodic boundary tag read by the cartesian-grid operator bank.

[PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC] does **not**
replace rows in the residual the way
[DirichletBC][diff_soup.domains.mesh.fem.dirichlet_bc.DirichletBC] does. It is a
declarative tag on one grid axis: the upwind advection kernels
([diff_soup.domains.grid.advection.collocated][]) take neighbours across the
domain edge on that axis, and the finite-volume and finite-difference diffusion
operators wrap their boundary fluxes on it.
"""

from __future__ import annotations

import equinox as eqx


class PeriodicBC(eqx.Module):
    """Periodic wrap-around on a pair of opposite boundaries.

    Declares that values at one end of ``axis`` are equal to values at the
    other end.  The operator layer uses this tag to compute face stencils
    that roll over the boundary (``jnp.roll`` semantics) rather than
    one-sided-extrapolating.

    This is the usual BC for PDEBench periodic-box problems
    (1D advection, Burgers, diffusion-reaction, compressible NS).

    Attributes:
        axis: Spatial axis on which to apply periodicity (0, 1, 2).
    """

    axis: int = eqx.field(static=True)
