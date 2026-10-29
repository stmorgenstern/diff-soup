# ``spec.bc.*`` authoring namespace.
"""``spec.bc.*`` discriminated-union namespace.

Each method appends a [BCSpec][diff_soup.problem._spec.specs.BCSpec] of the appropriate ``kind`` to the
parent ProblemSpec.  Boundary location is selected by ``on=`` — a
[BoundarySet][diff_soup.domains.boundary_set.BoundarySet] (build with
``ds.tags(...)``, combine with ``|`` / ``-``), or a plain tag-name string, or an
integer node-index array-like.  ``tag=`` (a ``TaggedMesh`` tag string) and
``node_indices=`` (an explicit integer array) are the single-form spellings of
the matching ``on=`` value. On a structured grid ``axis=`` with ``side=``
names one face instead. Exactly one location is required.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np

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


def _where(
    tag: Optional[str],
    node_indices: Any,
    on: Any,
    axis: Optional[int] = None,
    side: Optional[str] = None,
    *,
    kind: str = "bc",
) -> dict:
    """Resolve a BC location selector to ``{"selector": ...}`` or ``{"face": ...}``.

    Exactly one location must be given: ``on=``, ``tag=``, ``node_indices=``, or
    the grid face ``axis=`` with ``side=``.  ``on=`` promotes a
    [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] (used as-is), a
    string (``tags(str)``), or an array-like / boolean mask (``from_nodes(...)``).
    ``tag=`` and ``node_indices=`` are the single-form spellings of ``on=``.
    """
    if (axis is None) != (side is None):
        raise ValueError(
            f"spec.bc.{kind}: axis= and side= name one grid face together; pass both "
            f"(received axis={axis!r}, side={side!r})."
        )
    given = [on is not None, tag is not None, node_indices is not None, axis is not None]
    if sum(given) != 1:
        raise ValueError(
            f"spec.bc.{kind}: provide exactly one of on= / tag= / node_indices= / axis= with side="
        )
    if axis is not None:
        if side not in ("low", "high"):
            raise ValueError(f"spec.bc.{kind}: side must be 'low' or 'high', got {side!r}.")
        if isinstance(axis, bool) or int(axis) != axis or axis < 0:
            raise ValueError(
                f"spec.bc.{kind}: axis must be a grid axis number counting from 0, got {axis!r}."
            )
        return {"face": (int(axis), side)}
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
        axis: Optional[int] = None,
        side: Optional[str] = None,
        component: Optional[int] = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix a field's value on a boundary, `u = g`.

        Pass exactly one location (`on`, `tag`, `node_indices`, or `axis` with
        `side`), and exactly one of `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            axis: Grid axis of the face, with `side`, on a structured grid.
            side: `"low"` or `"high"`, the end of `axis` the face sits at.
            component: Index of the one value to fix at each node of a vector field, or `None` for every value.
            value: Constant value, a scalar or one entry per selected location, in storage
                order with a face axis dropped.
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
                where=_where(tag, node_indices, on, axis, side, kind="dirichlet"),
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
        axis: Optional[int] = None,
        side: Optional[str] = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
        name: Optional[str] = None,
        cross_facet: bool = False,
    ) -> None:
        """Fix a field's flux density through a boundary, `-n . F = g`, positive into the domain.

        Pass exactly one location (`on`, `tag`, `node_indices`, or `axis` with
        `side`), and exactly one of `value`, `value_fn` and `fn`. On a grid of two
        or three dimensions the cells must lie on one face, which weights them.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            axis: Grid axis of the face, with `side`, on a structured grid.
            side: `"low"` or `"high"`, the end of `axis` the face sits at.
            value: Constant value, a scalar or one entry per selected location, in storage
                order with a face axis dropped.
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
                where=_where(tag, node_indices, on, axis, side, kind="flux"),
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
        axis: Optional[int] = None,
        side: Optional[str] = None,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix a field's outward normal derivative on a boundary, `n . grad(u) = g`.

        Pass exactly one location (`on`, `tag`, `node_indices`, or `axis` with
        `side`), and exactly one of `value` and `value_fn`. The condition enters
        `laplacian` and `div_flux` wherever they are applied to `q.<var>`, with
        their own diffusivity, so it holds whatever `D` is.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            axis: Grid axis of the face, with `side`, on a structured grid.
            side: `"low"` or `"high"`, the end of `axis` the face sits at.
            value: Constant derivative, a scalar or one entry per selected location, in storage
                order with a face axis dropped.
            value_fn: Derivative as a function of dimensional time, `lambda t: ...`.
            fn: Refused, since a derivative that depends on the field is a Robin condition, `spec.bc.robin`.
        """
        _check_value("neumann", value, value_fn, fn)
        if fn is not None:
            raise ValueError(
                f"spec.bc.neumann({var!r}, fn=...): a normal derivative that depends on "
                "the field is a Robin condition. Declare it with spec.bc.robin, or state "
                "the flux density with spec.bc.flux(..., fn=...)."
            )
        self._spec._bcs.append(
            BCSpec(
                kind="neumann",
                var=var,
                where=_where(tag, node_indices, on, axis, side, kind="neumann"),
                value=value,
                value_fn=value_fn,
            )
        )

    def inflow(
        self,
        var: str,
        *,
        axis: int,
        side: str,
        velocity: Any,
        value: Any = None,
        value_fn: Optional[Callable] = None,
    ) -> None:
        """Feed a field through one grid face at a given speed and inlet value, `v c - D dc/dn = v c_in`.

        The face carries the total flux `velocity * value` into the domain, so the
        transport equation must use `q.ops.div_advect`, which carries no flux in
        through a face. Pass exactly one of `value` and `value_fn`.

        Args:
            var: Name of the field the condition applies to.
            axis: Grid axis of the inlet face.
            side: `"low"` or `"high"`, the end of `axis` the inlet sits at.
            velocity: Speed of the flow entering through the face, a non-negative scalar or one entry per face cell.
            value: Inlet value `c_in`, a scalar or one entry per face cell.
            value_fn: Inlet value as a function of dimensional time, `lambda t: ...`.
        """
        _check_value("inflow", value, value_fn, None)
        if callable(velocity) or bool(np.any(np.asarray(velocity, dtype=float) < 0)):
            raise ValueError(
                f"spec.bc.inflow({var!r}, velocity={velocity!r}): velocity is the speed "
                "of the flow entering through the face, a non-negative number or array. "
                "A face the flow leaves through needs no condition."
            )
        self._spec._bcs.append(
            BCSpec(
                kind="inflow",
                var=var,
                where=_where(None, None, None, axis, side, kind="inflow"),
                value=value,
                value_fn=value_fn,
                velocity=velocity,
            )
        )

    def periodic(self, var: str, *, axes: Optional[Any] = None) -> None:
        """Wrap a field periodically on the given axes of its cartesian grid.

        Every field on the same grid must be periodic on the same axes. The other
        faces take any other condition.

        Args:
            var: Name of the field the condition applies to.
            axes: Grid axes that wrap, or `None` (default) for every axis.
        """
        if axes is not None:
            if isinstance(axes, (int, np.integer)):
                axes = (axes,)
            axes = tuple(axes)
            bad = [a for a in axes if isinstance(a, bool) or int(a) != a or a < 0]
            if not axes or bad or len(set(axes)) != len(axes):
                raise ValueError(
                    f"spec.bc.periodic({var!r}, axes={axes!r}): axes lists distinct grid "
                    "axis numbers counting from 0, or is None for every axis."
                )
            axes = tuple(sorted(int(a) for a in axes))
        self._spec._bcs.append(BCSpec(kind="periodic", var=var, where={}, axes=axes))

    def robin(
        self,
        var: str,
        *,
        on: Optional[Any] = None,
        tag: Optional[str] = None,
        node_indices: Any = None,
        axis: Optional[int] = None,
        side: Optional[str] = None,
        alpha: Any,
        beta: Any,
        value: Any = None,
        value_fn: Optional[Callable] = None,
        fn: Optional[Callable] = None,
    ) -> None:
        """Fix a mixed condition on a boundary, `alpha * u + beta * n . grad(u) = g`.

        Pass exactly one location (`on`, `tag`, `node_indices`, or `axis` with
        `side`), and exactly one of `value`, `value_fn` and `fn`.

        Args:
            var: Name of the field the condition applies to.
            on: Boundary selector, a tag name, a `ds.tags(...)` expression, node indices, or a boolean node mask.
            tag: Boundary tag name, the single-tag spelling of `on`.
            node_indices: Node indices, the index spelling of `on`.
            axis: Grid axis of the face, with `side`, on a structured grid.
            side: `"low"` or `"high"`, the end of `axis` the face sits at.
            alpha: Coefficient of `u`, a constant or a function of time.
            beta: Coefficient of the normal derivative, a constant or a function of time, never zero.
            value: Constant value, a scalar or one entry per selected location, in storage
                order with a face axis dropped.
            value_fn: Value as a function of dimensional time, `lambda t: ...`.
            fn: Value as a function of the boundary view and dimensional time, `lambda q, t: ...`.
        """
        _check_value("robin", value, value_fn, fn)
        self._spec._bcs.append(
            BCSpec(
                kind="robin",
                var=var,
                where=_where(tag, node_indices, on, axis, side, kind="robin"),
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
