"""First-order upwind advection on collocated cartesian grids, and the periodic tag.

[Operators.advect_upwind][diff_soup.domains.operators.Operators.advect_upwind]
evaluates the advective term ``v . grad u`` on a cartesian
[Grid1D][diff_soup.domains.grid.Grid1D] or
[GridND][diff_soup.domains.grid.GridND] through the kernels in
[diff_soup.domains.grid.advection.collocated][]. An axis is periodic when a
[PeriodicBC][diff_soup.domains.grid.advection.bcs.PeriodicBC] tag names it;
every other axis takes a zero-gradient (outflow) condition. The finite-volume
and finite-difference diffusion operators read the same tag to wrap their
boundary fluxes.

All kernels are pure JAX functions: JIT-, grad- and vmap-safe.
"""
