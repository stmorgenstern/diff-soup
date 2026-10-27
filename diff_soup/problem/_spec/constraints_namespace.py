# ``spec.constraint.*`` authoring namespace.
"""``spec.constraint.*`` namespace.

Each method appends a [ConstraintSpec][] of the appropriate ``kind``
to the parent ProblemSpec.  Every kind (galvanostatic, integral,
reduce_residual, boundary_integral, flux_total) closes one global variable
named via ``closes=``, and every kind accepts ``integrand_scale=``.

``integrand_scale`` reconciles a non-dimensional constrained quantity with a
target protocol written in physical units.  It **divides the target** and
never touches the integrand: whatever the kind, the residual is

    quantity(q, t) - target_fn(t) / integrand_scale

so pass the dimensional value of one unit of the non-dimensional quantity, and leave
it unset (default ``None`` -> 1.0) when quantity and target already share
units.  The same number can be declared centrally, for any kind, with
``spec.scales(integrand={constraint_name: value})``; declaring it both ways
with different values warns at build time and the central value wins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from diff_soup.domains.boundary_set import BoundarySet, from_nodes, tags

from .specs import ConstraintSpec


def _selector_and_name(
    on: Any, tag: Optional[str], *, what: str
) -> tuple[BoundarySet, Optional[str]]:
    """Resolve a boundary-constraint location to ``(selector, single_tag_name)``.

    Exactly one of ``on=`` / ``tag=`` must be given.  ``on=`` promotes a
    [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] (used as-is), a
    string (``tags(str)``), or a node-index array-like (``from_nodes(...)``);
    ``tag=`` is the single-name spelling of ``on=``.  The second return value is
    the tag name when the selection is a single tag and ``None`` otherwise, kept
    so callers that read a single tag name keep working.
    """
    if (on is not None) == (tag is not None):
        raise ValueError(f"{what}: provide exactly one of on= / tag=")
    if tag is not None:
        on = tag
    if isinstance(on, BoundarySet):
        selector = on
    elif isinstance(on, str):
        selector = tags(on)
    else:
        selector = from_nodes(on)
    single = selector.names[0] if selector.op == "tags" and len(selector.names) == 1 else None
    return selector, single


class _ConstraintNamespace:
    """The constraint declarations of a recipe, reached as `spec.constraint`."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def galvanostatic(
        self,
        *,
        name: str,
        target_fn: Callable,
        closes: str,
        current_fn: Callable | None = None,
        boundary: Callable | None = None,
        on: Any = None,
        tag: str | None = None,
        integrand_scale: float | None = None,
    ) -> None:
        """Close a global by driving a computed scalar, or a boundary integral, to a target.

        Pass exactly one of `current_fn` and `boundary`. With `boundary`, the
        constraint is a boundary integral over exactly one of `on` and `tag`.

        Args:
            name: Label of the constraint, also its key in `spec.scales(integrand=...)`.
            target_fn: Target protocol `lambda t: ...`, receiving dimensional time.
            closes: Name of the global the constraint determines.
            current_fn: Closure `lambda q: ...` returning the constrained scalar.
            boundary: Closure `lambda q: ...` returning the integrand over the boundary.
            on: Boundary to integrate over, as a [`BoundarySet`][diff_soup.problem.BoundarySet], a tag name, or node indices.
            tag: Name of the boundary tag to integrate over.
            integrand_scale: Divisor of the target, or `None` for the `spec.scales(integrand=...)` value or 1.
        """
        given = [
            k
            for k, v in (
                ("current_fn", current_fn),
                ("boundary", boundary),
            )
            if v is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"galvanostatic: pass exactly one of current_fn=, boundary= (got {given or 'none'})"
            )

        if current_fn is not None:
            self._spec._constraints.append(
                ConstraintSpec(
                    kind="galvanostatic",
                    name=name,
                    current_fn=current_fn,
                    target_fn=target_fn,
                    closes=closes,
                    integrand_scale=integrand_scale,
                )
            )
        else:
            selector, tag_name = _selector_and_name(on, tag, what="galvanostatic(boundary=...)")
            self._spec._constraints.append(
                ConstraintSpec(
                    kind="boundary_integral",
                    name=name,
                    field=None,
                    tag=tag_name,
                    selector=selector,
                    integrand_fn=boundary,
                    target_fn=target_fn,
                    closes=closes,
                    integrand_scale=integrand_scale,
                )
            )

    def domain_integral(
        self,
        *,
        name: str,
        field: str,
        integrand_fn: Callable,
        target_fn: Callable,
        closes: str,
        integrand_scale: float | None = None,
    ) -> None:
        """Close a global by driving the domain integral of an integrand to a target.

        The integrand returns one value per node, weighted by the node's volume, or
        a scalar, weighted by the domain's total volume. `spec.constraint.integral`
        is the same method.

        Args:
            name: Label of the constraint, also its key in `spec.scales(integrand=...)`.
            field: Field whose domain supplies the volume measure.
            integrand_fn: Closure `lambda q: ...` returning the integrand.
            target_fn: Target protocol `lambda t: ...`, receiving dimensional time.
            closes: Name of the global the constraint determines.
            integrand_scale: Divisor of the target, or `None` for the `spec.scales(integrand=...)` value or 1.
        """
        self._spec._constraints.append(
            ConstraintSpec(
                kind="integral",
                name=name,
                field=field,
                integrand_fn=integrand_fn,
                target_fn=target_fn,
                closes=closes,
                integrand_scale=integrand_scale,
            )
        )

    # ``spec.constraint.integral`` is a shorter alias of ``domain_integral``;
    # the two declare exactly the same constraint.
    integral = domain_integral

    def reduce_residual(
        self,
        *,
        name: str,
        field: str,
        reduce_fn: Callable,
        target_fn: Callable,
        closes: str,
        integrand_scale: float | None = None,
    ) -> None:
        """Close a global by driving a reduction of a field's residual rows to a target.

        `reduce_fn` receives the field's assembled rows of the right-hand side,
        which in the mass-matrix form are its rates.

        Args:
            name: Label of the constraint, also its key in `spec.scales(integrand=...)`.
            field: Field whose residual rows are reduced.
            reduce_fn: Function `lambda f: ...` returning the constrained scalar.
            target_fn: Target protocol `lambda t: ...`, receiving dimensional time.
            closes: Name of the global the constraint determines.
            integrand_scale: Divisor of the target, or `None` for the `spec.scales(integrand=...)` value or 1.
        """
        self._spec._constraints.append(
            ConstraintSpec(
                kind="reduce_residual",
                name=name,
                field=field,
                reduce=reduce_fn,
                target_fn=target_fn,
                closes=closes,
                integrand_scale=integrand_scale,
            )
        )

    def boundary_integral(
        self,
        *,
        name: str,
        field: str,
        on: Any = None,
        tag: str | None = None,
        integrand_fn: Callable,
        target_fn: Callable,
        closes: str,
        integrand_scale: float | None = None,
    ) -> None:
        """Close a global by driving the boundary integral of an integrand to a target.

        The integrand returns a scalar, one value per boundary node, per facet, or
        per mesh node, each weighted by the matching boundary measure. Pass exactly
        one of `on` and `tag`.

        Args:
            name: Label of the constraint, also its key in `spec.scales(integrand=...)`.
            field: Field whose domain holds the boundary.
            on: Boundary to integrate over, as a [`BoundarySet`][diff_soup.problem.BoundarySet], a tag name, or node indices.
            tag: Name of the boundary tag to integrate over.
            integrand_fn: Closure `lambda q: ...` returning the integrand.
            target_fn: Target protocol `lambda t: ...`, receiving dimensional time.
            closes: Name of the global the constraint determines.
            integrand_scale: Divisor of the target, or `None` for the `spec.scales(integrand=...)` value or 1.
        """
        selector, tag_name = _selector_and_name(on, tag, what="boundary_integral")
        self._spec._constraints.append(
            ConstraintSpec(
                kind="boundary_integral",
                name=name,
                field=field,
                tag=tag_name,
                selector=selector,
                integrand_fn=integrand_fn,
                target_fn=target_fn,
                closes=closes,
                integrand_scale=integrand_scale,
            )
        )

    def flux_total(
        self,
        *,
        from_bc: str | list[str],
        closes: str,
        target_fn: Callable,
        name: str | None = None,
        integrand_scale: float | None = None,
    ) -> None:
        """Close a global by driving the total flux through named flux conditions to a target.

        The total integrates each condition's own flux closure over the facets it
        acts on, so the constraint and the boundary term agree.

        Args:
            from_bc: The `name=` of a `spec.bc.flux` condition, or a list of names whose totals are summed.
            closes: Name of the global the constraint determines.
            target_fn: Target protocol `lambda t: ...`, receiving dimensional time.
            name: Label of the constraint, or `None` for `"flux_total_<closes>"`.
            integrand_scale: Divisor of the target, or `None` for the `spec.scales(integrand=...)` value or 1.
        """
        if isinstance(from_bc, str):
            names = (from_bc,)
        else:
            names = tuple(from_bc)
        if not names:
            raise ValueError("flux_total: from_bc must name at least one flux boundary condition")
        self._spec._constraints.append(
            ConstraintSpec(
                kind="flux_total",
                name=name if name is not None else f"flux_total_{closes}",
                from_bc=names,
                target_fn=target_fn,
                closes=closes,
                integrand_scale=integrand_scale,
            )
        )
