# Static validation pass for ProblemSpec.
"""Static validation pass for ProblemSpec.

``level="static"``, the only level, runs without any JIT trace and checks
structural consistency: writes coverage, reads references, BC vars/where,
constraint targets, and names (fields, globals, params and functions share one
pool, and no field or global may take a name the ``q`` accessor answers itself).

Errors collected during a single [_validate_spec][diff_soup.problem._spec.validate._validate_spec] call are raised
as a single [SpecValidationError][diff_soup.problem._spec.validate.SpecValidationError] carrying the full list, so
users see all problems on one pass.

The static pass folds multi-write ``spec._multi_equations`` into the
write-coverage check, so a field written only by a ``@spec.equations``
closure is not wrongly reported as unwritten.
"""

from __future__ import annotations

import numpy as np


class SpecValidationError(ValueError):
    """Raised by [_validate_spec][diff_soup.problem._spec.validate._validate_spec] with the full list of structural errors."""

    def __init__(self, spec_name: str, errors: list[str]) -> None:
        self.spec_name = spec_name
        self.errors = list(errors)
        joined = "\n  - " + "\n  - ".join(errors)
        super().__init__(f"ProblemSpec {spec_name!r} failed validation:{joined}")


class SparsityUnderDeclarationError(ValueError):
    """Raised when an equation reads a field or global it does not declare in ``reads=``."""


def _grid_shape(dom):
    """Cell counts per axis of a structured grid, or ``None`` for any other domain.

    Duck-typed: the ``problem`` layer does not import the ``domains`` layer. A
    ``GridND`` carries ``shape``, a ``Grid1D`` its cell count ``n``.
    """
    if getattr(dom, "discretization_type", None) not in ("fvm", "fdm"):
        return None
    shape = getattr(dom, "shape", None)
    if shape is not None:
        return tuple(int(n) for n in shape)
    n = getattr(dom, "n", None)
    return None if n is None else (int(n),)


def _selected_cells(bc, dom, shape):
    """Flat cell indices a grid condition selects, or ``None`` when unknown here."""
    face = bc.where.get("face")
    if face is not None:
        axis, side = face
        flat = np.arange(int(np.prod(shape))).reshape(shape)
        index = [slice(None)] * len(shape)
        index[axis] = 0 if side == "low" else shape[axis] - 1
        return flat[tuple(index)].reshape(-1)
    selector = bc.where.get("selector")
    if selector is None:
        return None
    nodes_for_tag = getattr(dom, "nodes_for_tag", None)

    def on_tags(names):
        if not callable(nodes_for_tag):
            return None
        try:
            return set().union(*[set(int(i) for i in nodes_for_tag(n)) for n in names])
        except (KeyError, ValueError):
            return None

    def combine(a, b, op):
        return None if a is None or b is None else op(a, b)

    cells = selector.reduce(
        on_tags=on_tags,
        on_nodes=lambda idx: set(int(i) for i in np.asarray(idx).reshape(-1)),
        on_union=lambda a, b: combine(a, b, set.union),
        on_difference=lambda a, b: combine(a, b, set.difference),
    )
    return None if cells is None else np.array(sorted(cells), dtype=np.int64)


def _faces_holding(cells, shape) -> list:
    """The ``(axis, side)`` grid faces that contain every one of ``cells``."""
    if cells.size == 0 or cells.min() < 0 or cells.max() >= int(np.prod(shape)):
        return []
    multi = np.unravel_index(cells, shape)
    return [
        (d, side)
        for d in range(len(shape))
        for side, end in (("low", 0), ("high", shape[d] - 1))
        if bool(np.all(multi[d] == end))
    ]


def _validate_grid_faces(spec) -> list[str]:
    """Checks for conditions stated on grid faces: ``axis=``/``side=``, neumann and inflow."""
    errors: list[str] = []
    for bc in spec._bcs:
        dname = spec._field_domains.get(bc.var)
        dom = spec._domains.get(dname) if dname is not None else None
        if dom is None:
            continue
        shape = _grid_shape(dom)
        face = bc.where.get("face")
        tagged_mesh = hasattr(dom, "boundary_facets")
        if shape is None and (bc.kind == "inflow" or (bc.kind == "neumann" and not tagged_mesh)):
            where = "a structured grid" if bc.kind == "inflow" else "a grid or a tagged mesh"
            errors.append(
                f"spec.bc.{bc.kind}({bc.var!r}): the condition is imposed on {where}, and "
                f"{bc.var!r} lives on domain {dname!r}, which is not one. State the flux "
                "density with spec.bc.flux."
            )
            continue
        if face is None:
            continue
        if shape is None:
            errors.append(
                f"spec.bc.{bc.kind}({bc.var!r}, axis=, side=): axis= and side= name a face "
                f"of a structured grid, and domain {dname!r} is not one. Select the "
                "boundary with on=, tag= or node_indices=."
            )
        elif not 0 <= face[0] < len(shape):
            errors.append(
                f"spec.bc.{bc.kind}({bc.var!r}, axis={face[0]}): the grid of domain "
                f"{dname!r} has axes 0 to {len(shape) - 1}."
            )
    return errors


def _validate_periodic(spec, field_names: set) -> list[str]:
    """Structural checks for ``spec.bc.periodic(...)`` declarations.

    Rejects: a periodic field on a non-cartesian grid, an axis the grid does not
    have, another condition stated on the cells of a periodic face, and a grid
    domain whose fields are not all periodic on the same axes. Returns an empty
    list when ``spec`` declares no periodic boundary conditions, so it is safe to
    call unconditionally.
    """

    def _is_cartesian_grid(dom) -> bool:
        # Duck-typed rather than an isinstance check against
        # ``diff_soup.domains`` types: the ``problem`` layer does not import
        # the ``domains`` layer.  A ``Grid1D`` carries ``coord_system``
        # directly; a ``GridND`` carries ``axes`` (a tuple of per-axis
        # ``Grid1D``); a mesh domain has neither.  ``CoordSystem`` is a
        # ``str`` enum, so comparing against the literal ``"cartesian"``
        # matches it without importing the enum type.
        axes = getattr(dom, "axes", None)
        if axes is not None:
            return all(getattr(ax, "coord_system", None) == "cartesian" for ax in axes)
        coord_system = getattr(dom, "coord_system", None)
        if coord_system is not None:
            return coord_system == "cartesian"
        return False  # mesh / non-grid domains have no periodicity concept

    errors: list[str] = []
    periodic_axes: dict[str, tuple] = {}
    for bc in spec._bcs:
        if bc.kind != "periodic" or bc.var not in field_names:
            continue  # an unknown field is reported elsewhere
        dname = spec._field_domains.get(bc.var)
        dom = spec._domains.get(dname) if dname is not None else None
        if dom is not None and not _is_cartesian_grid(dom):
            errors.append(
                f"periodic boundary condition on field {bc.var!r} requires a cartesian grid "
                f"domain (cartesian_grid_1d / cartesian_grid_2d / cartesian_grid_3d); field "
                f"{bc.var!r} lives on domain {dname!r}, which is not a cartesian grid. Use a "
                f"cartesian grid for periodic boundaries."
            )
            continue
        shape = _grid_shape(dom)
        dim = len(shape) if shape is not None else 1
        axes = tuple(range(dim)) if bc.axes is None else tuple(bc.axes)
        if any(a >= dim for a in axes):
            errors.append(
                f"spec.bc.periodic({bc.var!r}, axes={list(axes)}): the grid of domain "
                f"{dname!r} has axes 0 to {dim - 1}."
            )
            continue
        periodic_axes[bc.var] = tuple(sorted(set(periodic_axes.get(bc.var, ())) | set(axes)))

    for bc in spec._bcs:
        axes = periodic_axes.get(bc.var)
        if bc.kind == "periodic" or not axes:
            continue
        dom = spec._domains.get(spec._field_domains.get(bc.var))
        shape = _grid_shape(dom)
        if shape is None:
            continue
        cells = _selected_cells(bc, dom, shape)
        if cells is None:
            continue
        faces = (
            [bc.where["face"]] if bc.where.get("face") is not None else _faces_holding(cells, shape)
        )
        if faces and all(axis in axes for axis, _ in faces):
            axis, side = faces[0]
            errors.append(
                f"spec.bc.{bc.kind}({bc.var!r}) is stated on the grid face axis={axis}, "
                f"side={side!r}, but {bc.var!r} is periodic on axis {axis}, which has no "
                "boundary there. Drop the condition, or leave that axis out of "
                f"spec.bc.periodic({bc.var!r}, axes=...)."
            )

    for dname in spec._domains:
        fields_here = [f for f, d in spec._field_domains.items() if d == dname]
        declared = {f: periodic_axes.get(f, ()) for f in fields_here}
        if len(set(declared.values())) > 1:
            listing = ", ".join(f"{f!r}: {list(a) or 'none'}" for f, a in sorted(declared.items()))
            errors.append(
                f"periodic boundary conditions on domain {dname!r} must wrap the same axes "
                f"for every field on that domain; the periodic axes are {listing}. Declare "
                f"spec.bc.periodic with the same axes for every field on {dname!r}, or none."
            )
    return errors


def _validate_names(spec) -> list[str]:
    """Name-pool collisions and reserved query names across every declaration.

    Fields, globals, params and functions are addressed by bare strings in an
    equation's ``reads=`` list, so one identifier has to name exactly one
    declared object; two declarations sharing a name leave the read with no way
    to say which it means.  Fields and globals are read as ``q.<name>`` on top
    of that, and the query object answers six of those attributes out of its own
    machinery, so a field or global with one of those names is unreachable.

    The declaration verbs refuse both cases at the callsite; this pass repeats
    the check over the assembled spec so a spec built by any other route is
    still caught before it lowers.
    """
    from .q_accessor import RESERVED_Q_NAMES

    errors: list[str] = []
    claimed: dict[str, str] = {}
    for kind, registry in (
        ("field", spec._fields),
        ("global", spec._globals),
        ("param", spec._params),
        ("function", spec._functions),
    ):
        for name in registry:
            held = claimed.get(name)
            if held is None:
                claimed[name] = kind
            else:
                errors.append(
                    f"duplicate name {name!r}: declared as a {held} and again as a "
                    f"{kind}. Fields, globals, params, and functions share one name "
                    f"pool, because an equation's reads=({name!r},) carries no "
                    "namespace to tell them apart. Rename one of the two declarations."
                )
            if kind in ("field", "global") and name in RESERVED_Q_NAMES:
                reserved = ", ".join(sorted(RESERVED_Q_NAMES))
                errors.append(
                    f"{kind} {name!r} takes a name reserved by the equation query "
                    f"object, which answers q.{name} out of its own machinery, so the "
                    f"{kind} could never be read back as q.{name}. The reserved names "
                    f"are {reserved}. Rename the {kind}."
                )
    return errors


def _component_errors(spec, bc) -> list[str]:
    """Check a Dirichlet ``component=`` against the width its field declares.

    A field carries one value per node unless it was declared with
    ``components=d``, in which case node ``k`` owns the ``d`` consecutive slots
    starting at ``k*d``.  ``component=`` picks one of those slots, so it has to
    be a number the field's nodes actually have; a component on a
    one-value-per-node field, or one at or past the declared width, addresses a
    slot of a *different node* and is refused.

    Returns an empty list for every boundary condition that names no component.
    """
    component = getattr(bc, "component", None)
    if component is None:
        return []
    fspec = spec._fields.get(bc.var)
    if fspec is None:
        return []
    shape = fspec.shape
    width = 1 if shape is None or len(shape) < 2 else int(np.prod(shape[1:]))
    if width == 1:
        return [
            f"BC on {bc.var!r}: component={component} selects one of the values a node "
            f"carries, but {bc.var!r} carries a single value per node. Drop component=, "
            "or declare the field with components=."
        ]
    if int(component) >= width:
        return [
            f"BC on {bc.var!r}: component={component} is outside the {width} values each "
            f"node of {bc.var!r} carries (0 to {width - 1})."
        ]
    return []


def _validate_static(spec) -> list[str]:
    errors: list[str] = []

    field_names = set(spec._fields)
    global_names = set(spec._globals)
    # The pool an equation's ``reads=`` may name: state unknowns, plus the
    # read-through aliases a coupling injects.  An alias carries no state row of
    # its own — it is another component's value made readable by name inside
    # this one — but an equation names it in ``reads=`` exactly as it names a
    # field.  Params and functions reach closures through ``q.p`` / ``q.fn`` and
    # are not read declarations, but they do share the identifier pool — see
    # [_validate_names][diff_soup.problem._spec.validate._validate_names].
    alias_names = {alias for _comp, alias in getattr(spec, "_read_aliases", {})}
    name_pool = field_names | global_names | alias_names

    errors.extend(_validate_names(spec))

    # Fold multi-write closures into the write-coverage set so a field written
    # only by a ``@spec.equations`` closure is not falsely reported as unwritten.
    written = {eq.writes for eq in spec._equations}
    for meq in getattr(spec, "_multi_equations", []):
        written.update(meq.writes)
    # A coupling's phantom field holds the injected value; the build walk gives
    # it the zero right-hand side the coupling adds onto, so it is written even
    # though the recipe declares no equation for it.
    written.update(getattr(spec, "_phantom_fields", {}) or {})
    for fname, fspec in spec._fields.items():
        if fname not in written:
            errors.append(
                f"field {fname!r} has no equation writing to it "
                f"(mass={fspec.mass}); add @spec.equation(writes={fname!r}, ...)"
            )

    # An equation's name defaults to its closure's own name, so one closure
    # declaring an equation per field inside a loop legitimately repeats it.  A
    # name the author chose is a different thing: it is there to tell this
    # equation from the others, and two equations answering to it tell them
    # apart from nothing.  Only chosen names are held to uniqueness.
    chosen_names = set()
    for eq in spec._equations:
        if eq.writes not in field_names and eq.writes not in global_names:
            errors.append(
                f"equation {eq.name!r} writes to unknown name {eq.writes!r} "
                f"(known fields={sorted(field_names)}, globals={sorted(global_names)})"
            )
        for r in eq.reads:
            if r not in name_pool:
                errors.append(
                    f"equation {eq.name!r} reads unknown name {r!r} "
                    f"(known fields={sorted(field_names)}, "
                    f"globals={sorted(global_names)})"
                )
        if eq.name == getattr(eq.fn, "__name__", eq.name):
            continue
        if eq.name in chosen_names:
            errors.append(
                f"duplicate equation name {eq.name!r}: two equations were declared with "
                f"name={eq.name!r}, so neither can be told from the other. Give them "
                "different names, or drop name= and let each closure supply its own."
            )
        chosen_names.add(eq.name)

    for bc in spec._bcs:
        if bc.var not in field_names:
            errors.append(
                f"boundary condition (kind={bc.kind!r}) references unknown field {bc.var!r}"
            )
            continue
        domain_name = spec._field_domains.get(bc.var)
        domain = spec._domains.get(domain_name) if domain_name else None
        # Range-check any explicit node indices the selector carries.  A
        # tag-name leaf contributes no indices to check (it resolves against the
        # mesh at lowering); a raw node-index leaf does.  Indices anywhere in the
        # tree (either side of a union or difference) are collected, since a
        # nonexistent index is a typo regardless of position.
        selector = bc.where.get("selector")
        node_idx = None
        if selector is not None:
            node_idx = selector.reduce(
                on_tags=lambda names: np.empty(0, dtype=np.int64),
                on_nodes=lambda idx: np.asarray(idx, dtype=np.int64),
                on_union=lambda a, b: np.concatenate([a, b]),
                on_difference=lambda a, b: np.concatenate([a, b]),
            )
        elif "node_indices" in bc.where:
            arr = np.asarray(bc.where["node_indices"])
            if arr.dtype == bool:
                # Range-check the True positions of a boolean mask, not its
                # {0, 1} integer coercion (which would pass silently).
                arr = np.nonzero(arr.reshape(-1))[0]
            node_idx = np.asarray(arr, dtype=np.int64)
        if node_idx is not None and node_idx.size and domain is not None:
            n = getattr(domain, "num_nodes", None)
            if n is not None and ((node_idx < 0).any() or (node_idx >= n).any()):
                errors.append(
                    f"BC on {bc.var!r}: node_indices out of range "
                    f"(min={int(node_idx.min())}, max={int(node_idx.max())}, num_nodes={n})"
                )
        errors.extend(_component_errors(spec, bc))

    errors.extend(_validate_periodic(spec, field_names))
    errors.extend(_validate_grid_faces(spec))

    constraint_names: set = set()
    # ``closes=`` has to pair constraints and globals one-to-one: a global
    # holds one unknown slice (a scalar unless declared wider), a constraint
    # the rows that close it, and the pairing is what makes the declared
    # system square.
    closed_by: dict = {}
    for c in spec._constraints:
        if c.name in constraint_names:
            errors.append(f"duplicate constraint name {c.name!r}")
        constraint_names.add(c.name)
        if c.closes is None:
            errors.append(
                f"constraint {c.name!r} names no global to close; pass "
                "closes=<name of a spec.global_var>"
            )
        elif c.closes in field_names and c.closes not in global_names:
            errors.append(
                f"constraint {c.name!r} closes {c.closes!r}, which is a field, "
                "not a global; closes= names a scalar unknown declared with "
                "spec.global_var"
            )
        elif c.closes not in global_names:
            errors.append(
                f"constraint {c.name!r} closes {c.closes!r}, which is not a "
                "declared global; closes= names a global declared with "
                "spec.global_var"
            )
        elif c.closes in closed_by:
            errors.append(
                f"constraints {closed_by[c.closes]!r} and {c.name!r} both close "
                f"global {c.closes!r}; a global carries one multiplier and is "
                "determined by exactly one constraint row"
            )
        else:
            closed_by[c.closes] = c.name
        if c.field is not None and c.field not in field_names:
            errors.append(f"constraint {c.name!r} references unknown field {c.field!r}")

    obs_names: set = set()
    for o in spec._observations:
        if o.name in obs_names:
            errors.append(f"duplicate observation name {o.name!r}")
        obs_names.add(o.name)

    ctl_names: set = set()
    for ctl in spec._controls:
        if ctl.name in ctl_names:
            errors.append(f"duplicate control name {ctl.name!r}")
        ctl_names.add(ctl.name)
        if ctl.binds_to is not None and ctl.binds_to not in global_names:
            errors.append(f"control {ctl.name!r} binds_to unknown global {ctl.binds_to!r}")

    errors.extend(_validate_global_closures(spec, global_names))

    if getattr(spec, "_geometry", None):
        errors.extend(_validate_geometry_scales(spec))

    return errors


def _coupling_domains(spec, coupling) -> set:
    """Names of the declared domains a recorded coupling reads from or writes to."""
    domains = set()
    owner = getattr(coupling, "target_owner", None)
    if owner is not None:
        domains.add(owner)
    for end in (coupling.source, coupling.target):
        name = getattr(end, "name", None)
        if name is not None and spec._fields.get(name) is end:
            domains.add(spec._field_domains.get(name))
        for (consumer, _alias), alias_var in getattr(spec, "_read_aliases", {}).items():
            if alias_var is end:
                domains.add(consumer)
    return domains


def _declared_closures(spec):
    """Yield ``(kind, name, callable)`` for every closure a recipe declares."""
    for eq in spec._equations:
        yield "equation", eq.name, eq.fn
    for meq in getattr(spec, "_multi_equations", []):
        yield "equation", meq.name, meq.fn
    for bc in spec._bcs:
        for attribute in ("value", "value_fn", "fn", "alpha", "beta"):
            fn = getattr(bc, attribute, None)
            if callable(fn):
                yield f"{bc.kind} boundary condition", bc.var, fn
    for c in spec._constraints:
        for attribute in ("integrand_fn", "target_fn", "current_fn", "reduce"):
            fn = getattr(c, attribute, None)
            if callable(fn):
                yield "constraint", c.name, fn
    for o in spec._observations:
        yield "observation", o.name, o.fn
    for ctl in spec._controls:
        yield "control", ctl.name, ctl.fn


def _is_plain_number(value) -> bool:
    """Whether ``value`` is a Python, NumPy or zero-dimensional array number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, complex, np.number)):
        return True
    dtype = getattr(value, "dtype", None)
    if dtype is None or getattr(value, "ndim", None) != 0:
        return False
    try:
        return np.dtype(dtype).kind in "iufc"
    except TypeError:
        return False


def _number_held_under(fn, names, seen: set):
    """``(function, variable, value)`` of a number ``fn`` holds under one of ``names``.

    Looks at the function's captured variables and default arguments, and
    recursively at the functions it captures, the way the content hash does.
    Module globals are not looked at. Returns ``None`` when nothing matches.
    """
    from types import FunctionType

    # Unwrap ``functools.wraps``-style wrappers (the runtime type-check hook
    # wraps functions this way).
    while id(fn) not in seen:
        try:
            inner = getattr(fn, "__wrapped__", None)
        except Exception:  # noqa: BLE001 - an object whose attribute lookup raises
            inner = None
        if inner is None:
            break
        seen.add(id(fn))
        fn = inner
    if not isinstance(fn, FunctionType) or id(fn) in seen:
        return None
    seen.add(id(fn))
    code = fn.__code__
    held = []
    for variable, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            held.append((variable, cell.cell_contents))
        except ValueError:  # a cell not yet filled
            continue
    positional = code.co_varnames[: code.co_argcount]
    defaults = fn.__defaults__ or ()
    held.extend(zip(positional[len(positional) - len(defaults) :], defaults))
    held.extend((fn.__kwdefaults__ or {}).items())
    for variable, value in held:
        if variable in names and _is_plain_number(value):
            return fn, variable, value
    for _variable, value in held:
        if isinstance(value, FunctionType):
            found = _number_held_under(value, names, seen)
            if found is not None:
                return found
    return None


def _validate_geometry_scales(spec) -> list[str]:
    """Checks for grids scaled by a declared parameter (``spec.geometry``).

    A coupling may not touch a scaled domain: its transfer is resolved on the
    host from the unit grid and would not follow the scale. A declared closure
    may not hold a plain number under a scale's name: that number would stay
    fixed while the scale changes between evaluations, which the closure's
    author almost certainly did not mean; an unrelated constant of the same name
    is refused too, and renaming either one satisfies the check.
    """
    errors: list[str] = []
    scaled = dict(spec._geometry)
    for coupling in spec._couplings:
        for domain in sorted(d for d in _coupling_domains(spec, coupling) if d in scaled):
            errors.append(
                f"coupling {coupling.name!r} reads from or writes to domain {domain!r}, "
                f"which is scaled by {scaled[domain]!r} (spec.geometry); a coupling's "
                "transfer is resolved on the unit grid and would not follow the scale. "
                "Remove the coupling, or build the grid at its size without spec.geometry."
            )
    names = frozenset(scaled.values())
    for kind, declaration, fn in _declared_closures(spec):
        found = _number_held_under(fn, names, set())
        if found is None:
            continue
        holder, variable, value = found
        qualname = getattr(holder, "__qualname__", repr(holder))
        errors.append(
            f"{kind} {declaration!r}: the function {qualname} holds the number "
            f"{value!r} under the name {variable!r}, which is the grid scale declared "
            f"with spec.geometry({variable!r}); that number stays fixed while the scale "
            f"changes. Read the scale as q.p.{variable} inside the closure, or rename "
            "the variable if it is an unrelated constant."
        )
    return errors


def _validate_global_closures(spec, global_names: set) -> list[str]:
    """Enforce exactly one closure per global, matched to its mass role.

    A differential global is written by exactly one ordinary equation and by
    nothing else. An algebraic global is closed by exactly one constraint or
    bound control row and is written by no equation.
    """
    errors: list[str] = []
    eq_writers: dict[str, list[str]] = {}
    for eq in spec._equations:
        if eq.writes in global_names:
            eq_writers.setdefault(eq.writes, []).append(eq.name)
    for meq in getattr(spec, "_multi_equations", []):
        for w in meq.writes:
            if w in global_names:
                errors.append(
                    f"multi-equation {meq.name!r} writes global {w!r}; a global "
                    f"takes exactly one single-write @spec.equation"
                )
    closers: dict[str, list[str]] = {}
    for c in spec._constraints:
        if c.closes is not None and c.closes in global_names:
            closers.setdefault(c.closes, []).append(f"constraint {c.name!r}")
    bound_by: dict[str, list[str]] = {}
    for ctl in spec._controls:
        if ctl.binds_to is not None and ctl.binds_to in global_names:
            bound_by.setdefault(ctl.binds_to, []).append(f"control {ctl.name!r}")

    from .specs import OperationalControlSpec

    operational_tables = [
        constraint
        for constraint in spec._constraints
        if isinstance(constraint, OperationalControlSpec)
    ]
    if len(operational_tables) > 1:
        errors.append(
            f"a problem may declare only one operational control table; found "
            f"{len(operational_tables)}. Combine all current, voltage, and power "
            f"modes into one spec.control.operational(...) declaration."
        )

    obs_by_name = {o.name: o for o in spec._observations}
    for c in spec._constraints:
        if not isinstance(c, OperationalControlSpec):
            continue
        for role, obs_name in (("current", c.current), ("voltage", c.voltage)):
            if role == "current" and obs_name is None:
                # The current is the sum of a field's assembled residual rows;
                # the field itself is checked with every constraint's field.
                continue
            obs = obs_by_name.get(obs_name)
            if obs is None:
                errors.append(
                    f"operational control {c.name!r}: {role} observable "
                    f"{obs_name!r} is not a declared observation; declare it "
                    f"with spec.observe.scalar(name={obs_name!r}, fn=...)"
                )
            elif obs.kind != "scalar":
                errors.append(
                    f"operational control {c.name!r}: {role} observable "
                    f"{obs_name!r} has kind {obs.kind!r}; the table selects "
                    f"among scalar terminal rows; declare it with "
                    f"spec.observe.scalar"
                )
        gspec = spec._globals.get(c.closes)
        if gspec is None:
            errors.append(
                f"operational control {c.name!r} closes unknown global "
                f"{c.closes!r}; declare it with "
                f'spec.global_var({c.closes!r}, mass="algebraic")'
            )
        elif gspec.mass != "algebraic":
            errors.append(
                f"operational control {c.name!r}: current unknown {c.closes!r} "
                f"is declared mass={gspec.mass!r}; the table closes exactly "
                f"one algebraic global"
            )
        elif np.shape(gspec.initial) != ():
            errors.append(
                f"operational control {c.name!r}: current unknown {c.closes!r} "
                f"must be a scalar global; got initial shape "
                f"{np.shape(gspec.initial)}"
            )
        for ctl in spec._controls:
            if ctl.binds_to == c.closes:
                errors.append(
                    f"operational control {c.name!r}: global {c.closes!r} is "
                    f"also bound by control {ctl.name!r}; a bound value would "
                    f"overwrite the operational row; close the global with "
                    f"exactly one of the two"
                )

    for gname, gspec in spec._globals.items():
        writers = eq_writers.get(gname, [])
        closures = closers.get(gname, [])
        bindings = bound_by.get(gname, [])
        if bindings and gspec.mass == "differential":
            errors.append(
                f"differential global {gname!r} is bound by {', '.join(bindings)}; "
                f"control bindings drive algebraic globals only"
            )
        if gspec.mass == "differential":
            if len(writers) == 0:
                errors.append(
                    f"differential global {gname!r} has no equation writing it; add "
                    f"exactly one @spec.equation(writes={gname!r}, ...) returning "
                    f"its time derivative"
                )
            elif len(writers) > 1:
                errors.append(
                    f"differential global {gname!r} is written by multiple equations "
                    f"{writers!r}; declare exactly one"
                )
            if closures:
                errors.append(
                    f"differential global {gname!r} is also closed by "
                    f"{', '.join(closures)}; a differential global takes exactly one "
                    f"ordinary equation and no constraint/control closure"
                )
        else:
            if writers:
                errors.append(
                    f"equation(s) {writers!r} write algebraic global {gname!r}; an "
                    f"algebraic global is closed by exactly one constraint row; "
                    f'declare it with mass="differential" to give it an ordinary '
                    f"equation"
                )
            if len(closures) == 0:
                errors.append(
                    f"algebraic global {gname!r} has no closing row; close it with "
                    f"exactly one constraint (closes={gname!r}) or bound control "
                    f'(binds_to={gname!r}), or declare it mass="differential" '
                    f"with an equation"
                )
            elif len(closures) > 1:
                errors.append(
                    f"algebraic global {gname!r} is closed by multiple rows: "
                    f"{', '.join(closures)}; declare exactly one closure"
                )
    return errors


def _validate_spec(spec, *, level: str = "static") -> None:
    if level != "static":
        raise ValueError(
            f"unknown validation level {level!r}; 'static' is the only validation level."
        )
    errors = _validate_static(spec)
    if errors:
        raise SpecValidationError(spec.name, errors)
