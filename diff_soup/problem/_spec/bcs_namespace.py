# ``spec.bc.*`` authoring namespace.
"""``spec.bc.*`` discriminated-union namespace.

Each method appends a [BCSpec][diff_soup.problem._spec.specs.BCSpec] of the appropriate ``kind`` to the
parent ProblemSpec.  Boundary location is selected by ``on=`` — a
[BoundarySet][diff_soup.domains.boundary_set.BoundarySet] (build with
``ds.tags(...)``, combine with ``|`` / ``-``), or a plain tag-name string, or an
integer node-index array-like.  ``tag=`` (a ``TaggedMesh`` tag string) and
``node_indices=`` (an explicit integer array) are the single-form spellings of
the matching ``on=`` value; exactly one of the three is required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from diff_soup.domains.boundary_set import BoundarySet, from_nodes, tags

from .specs import BCSpec


def _check_value(kind, value, value_fn, fn):
    """Require one unambiguous source of the prescribed boundary value."""
    given = [
        name
        for name, arg in (("value", value), ("value_fn", value_fn), ("fn", fn))
        if arg is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"spec.bc.{kind}: value=, value_fn=, and fn= are mutually exclusive; "
            f"provide exactly one (received {', '.join(given) or 'none'})."
        )


def _where(tag: Optional[str], node_indices: Any, on: Any) -> dict:
    """Resolve a BC location selector to ``{"selector": <BoundarySet>}``.

    Exactly one of ``on=`` / ``tag=`` / ``node_indices=`` must be given.  ``on=``
    promotes a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] (used as-is), a string (``tags(str)``), or an
    array-like / boolean mask (``from_nodes(...)``).  ``tag=`` and
    ``node_indices=`` are the single-form spellings of ``on=``.
    """
    given = [on is not None, tag is not None, node_indices is not None]
    if sum(given) != 1:
        raise ValueError("spec.bc.*: provide exactly one of on= / tag= / node_indices=")
    if tag is not None:
        on = tag
    elif node_indices is not None:
        on = node_indices
    if isinstance(on, BoundarySet):
        selector = on
    elif isinstance(on, str):
        selector = tags(on)
    else:
        selector = from_nodes(on)
    return {"selector": selector}


def _check_component(var: str, component: Any) -> int:
    """Coerce ``component=`` to a plain non-negative index.

    A component numbers one of the values a node carries, so it is a whole
    number counting from zero.  How many values the node actually carries
    depends on the field declaration, which may come after the boundary
    condition, so the range is checked when the two are matched up (in
    validation and again while lowering).
    """
    try:
        index = int(component)
    except (TypeError, ValueError):
        raise TypeError(
            f"spec.bc.dirichlet({var!r}, component={component!r}): component numbers one "
            "of the values a node carries and must be an integer counting from zero."
        ) from None
    if index != component or index < 0:
        raise ValueError(
            f"spec.bc.dirichlet({var!r}, component={component!r}): component numbers one "
            "of the values a node carries and must be an integer counting from zero."
        )
    return index


class _BCNamespace:
    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def dirichlet(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        component: Optional[int] = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix a field's value on a boundary, `u = g`.

        Pass exactly one of `on`, `tag` and `node_indices`, and exactly one of
        `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            component: Index of the one value to fix at each node of a vector field, or `None` for every value.
            value: Constant value, a scalar or one entry per selected location.
            value_fn: Value as a function of dimensional time, `lambda t: ...`.
            fn: Value as a function of the boundary view and dimensional time, `lambda q, t: ...`.
        """
        _check_value("dirichlet", value, value_fn, fn)
        if component is not None:
            component = _check_component(var, component)
        self._spec._bcs.append(
            BCSpec(
                kind="dirichlet",
                var=var,
                where=_where(tag, node_indices, on),
                component=component,
                value=value,
                value_fn=value_fn,
                fn=fn,
            )
        )

    def flux(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
        name: Optional[str] = None,
        cross_facet: bool = False,
    ) -> None:
        """Fix a field's flux density through a boundary, `-n . F = g`, positive into the domain.

        Pass exactly one of `on`, `tag` and `node_indices`, and exactly one of
        `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            value: Constant value, a scalar or one entry per selected location.
            value_fn: Value as a function of dimensional time, `lambda t: ...`.
            fn: Value as a function of the boundary view and dimensional time, `lambda q, t: ...`.
            name: Name that `spec.constraint.flux_total(from_bc=...)` refers to, on a finite-element mesh only.
            cross_facet: If `True`, the `fn` closure sees the whole boundary at once instead of one facet.
        """
        _check_value("flux", value, value_fn, fn)
        if cross_facet and fn is None:
            raise ValueError(
                "spec.bc.flux(cross_facet=True): the flag declares how the "
                "field-dependent fn= form reads the boundary; the value= and "
                "value_fn= forms read no facets at all. Drop the flag, or "
                "state the flux with fn=lambda q, t: ..."
            )
        if name is not None:
            self._refuse_unlinkable_flux_name(var, name)
        self._spec._bcs.append(
            BCSpec(
                kind="flux",
                var=var,
                where=_where(tag, node_indices, on),
                value=value,
                value_fn=value_fn,
                fn=fn,
                name=name,
                cross_facet=bool(cross_facet),
            )
        )

    def _refuse_unlinkable_flux_name(self, var: str, name: str) -> None:
        """Reject ``name=`` where no flux-total closure could ever resolve it.

        A flux name is only useful as the handle
        [flux_total][diff_soup.problem._spec.constraints_namespace._ConstraintNamespace.flux_total]
        links to, and that integral runs over boundary facets, which only a
        tagged FEM mesh has. Refusing here puts the error on the declaration that
        supplied the argument. A field declared after its boundary condition
        cannot be checked yet; lowering refuses that case instead.
        """
        domain_name = self._spec._field_domains.get(var)
        if domain_name is None:
            return
        domain = self._spec._domains.get(domain_name)
        if getattr(domain, "discretization_type", None) == "fem":
            return
        raise ValueError(
            f"spec.bc.flux(..., name={name!r}) on field {var!r}: a flux name is the "
            "handle spec.constraint.flux_total(from_bc=...) integrates over, and that "
            f"integral needs boundary facets, which domain {domain_name!r} does not "
            "have. Drop name=, or put the field on a tagged FEM mesh."
        )

    def neumann(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
        name: Optional[str] = None,
        cross_facet: bool = False,
    ) -> None:
        """Fix a field's flux density through a boundary, an alias of `flux`."""
        self.flux(
            var,
            on=on,
            tag=tag,
            node_indices=node_indices,
            value=value,
            value_fn=value_fn,
            fn=fn,
            name=name,
            cross_facet=cross_facet,
        )

    def periodic(self, var: str) -> None:
        """Wrap a field periodically on every axis of its cartesian grid.

        A periodic field takes no other condition, and every field on the same grid
        must be periodic too.

        Args:
            var: Name of the field the condition applies to.
        """
        self._spec._bcs.append(BCSpec(kind="periodic", var=var, where={}))

    def robin(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        alpha: Any,
        beta: Any,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix a mixed condition on a boundary, `alpha * u + beta * n . grad(u) = g`.

        Pass exactly one of `on`, `tag` and `node_indices`, and exactly one of
        `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            alpha: Coefficient of `u`, a constant or a function of time.
            beta: Coefficient of the normal derivative, a constant or a function of time, never zero.
            value: Constant value, a scalar or one entry per selected location.
            value_fn: Value as a function of dimensional time, `lambda t: ...`.
            fn: Value as a function of the boundary view and dimensional time, `lambda q, t: ...`.
        """
        _check_value("robin", value, value_fn, fn)
        self._spec._bcs.append(
            BCSpec(
                kind="robin",
                var=var,
                where=_where(tag, node_indices, on),
                value=value,
                value_fn=value_fn,
                alpha=alpha,
                beta=beta,
                fn=fn,
            )
        )

    def traction(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix the traction of a vector field on a boundary, `sigma . n = t`.

        Pass exactly one of `on`, `tag` and `node_indices`, and exactly one of
        `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            value: Constant traction vector `t`.
            value_fn: Traction as a function of dimensional time, `lambda t: ...`.
            fn: Traction as a function of the boundary view and dimensional time, `lambda q, t: ...`.
        """
        _check_value("traction", value, value_fn, fn)
        self._spec._bcs.append(
            BCSpec(
                kind="traction",
                var=var,
                where=_where(tag, node_indices, on),
                value=value,
                value_fn=value_fn,
                fn=fn,
            )
        )

    def _require_directional_field(self, var: str, method: str) -> None:
        """Reject a scalar field for a condition stated on the boundary normal.

        ``n·u = 0`` projects the field onto the outward normal, which a field
        with one value per node cannot supply.  Checked here so the recipe fails
        where the condition is written; lowering re-checks against the IR shape
        for boundary conditions built without this namespace.
        """
        fields = getattr(self._spec, "_fields", None)
        field_spec = fields.get(var) if isinstance(fields, dict) else None
        if field_spec is None:
            return  # declared before the field; lowering catches it.
        shape = getattr(field_spec, "shape", None)
        if shape is not None and len(shape) >= 2:
            return
        declared = "no components= or shape=" if shape is None else f"shape={tuple(shape)}"
        raise ValueError(
            f"{method}({var!r}): the condition n·u = 0 projects the field onto the "
            "boundary normal, so it needs one component per spatial dimension. Field "
            f"{var!r} carries a single value per node ({declared}) and has no "
            "direction. Declare it with components=<spatial dim>, or state the scalar "
            "condition directly with spec.bc.dirichlet or spec.bc.flux."
        )

    def symmetry(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
    ) -> None:
        """Fix the normal component of a vector field to zero on a boundary, `n . u = 0`.

        Pass exactly one of `on`, `tag` and `node_indices`. The field needs one
        component per spatial dimension.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
        """
        self._require_directional_field(var, "spec.bc.symmetry")
        self._spec._bcs.append(
            BCSpec(
                kind="symmetry",
                var=var,
                where=_where(tag, node_indices, on),
            )
        )

    def no_slip(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        wall_velocity: Any = None,
    ) -> None:
        """Fix a field to the wall velocity on a boundary, the same as `dirichlet` with `value=wall_velocity`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            wall_velocity: Velocity of the wall, or `None` for zero.
        """
        if wall_velocity is None:
            wall_velocity = 0.0
        self.dirichlet(var, on=on, tag=tag, node_indices=node_indices, value=wall_velocity)

    def slip_wall(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
    ) -> None:
        """Fix the normal component of a vector field to zero on a boundary, an alias of `symmetry`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
        """
        self._require_directional_field(var, "spec.bc.slip_wall")
        self.symmetry(var, on=on, tag=tag, node_indices=node_indices)
