"""Discrete field abstraction: arrays bound to a discretization with location metadata.

``DiscreteField`` wraps an array value together with a *location* tag indicating
where on a domain the values live (cell center, face, node, quadrature point).
[StaggeredGrid.face_shape][diff_soup.domains.staggered_grid.StaggeredGrid.face_shape]
maps a ``Location`` to the array shape a value there takes.

The name ``DiscreteField`` is chosen to avoid collision with the operator
coefficient ``Field`` (a scalar/callable coefficient wrapper for things like
``D(c)`` in [diff_soup.domains.operators][]) — the two concepts are unrelated.

Design
------
``DiscreteField`` is an ``eqx.Module`` pytree with a single traced leaf
(``array``) and static metadata (``location``).  A domain reference is **not**
stored inside the field; following the existing ``Operators(domain)`` pattern,
the domain is always carried by the operator / problem context that consumes
or produces the field.  This keeps the pytree minimal, compiles away at XLA
level, and avoids awkward vmap-over-domain issues.
"""

from diff_soup.domains.fields.field import DiscreteField, Location

__all__ = ["DiscreteField", "Location"]
