"""Family-field initial-condition materialisation for ``assemble_state``.

Shared between ``ProblemSpec.assemble_state`` and
``LoweredProblem.assemble_state``: a [Family][diff_soup.problem._ir.family.Family] population's per-member initial conditions are supplied at
``assemble_state`` time.  Each member field's ``value`` callable accepts
``(coords, m)`` — the member domain's node coordinates plus the per-member
parameter slice ``m`` — and is vmapped over ``fam.member_params`` to a
``(count, member_field_size)`` slab.

The family slab is laid out MEMBER-MAJOR (member 0's full profile, then member
1's, ...) directly after all [Component][diff_soup.problem._ir.component.Component] blocks, in ``EquationSet.families`` order (matching
[_layout][diff_soup._lowering.layout._layout]).  Each member field's
``(count, member_field_size)`` materialisation is written into the field's
column band ``[var_local_off : var_local_off + var.size]`` of a
``(count, member_state_size)`` slab, which flattens to the member-major flat
block.

Values flow through as traced leaves, so the assembled state is differentiable
with respect to them.

This module lives under ``diff_soup/_lowering``.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def _materialise_family_field(fam, var, value, coords, m_batch):
    """Materialise one member field to a ``(count, member_field_size)`` slab.

    Args:
        fam: The [Family][diff_soup.problem._ir.family.Family] (for ``count`` and error naming).
        var: The member-field [Variable][diff_soup.problem._ir.variable.Variable] template.
        value: ``callable(coords, m) | array | scalar``.  A one-arg callable
            raises ``ValueError`` — family closures must accept both
            ``(coords, m)``.
        coords: Member-domain node coordinates ``(n_nodes, dim)`` (or ``None``).
        m_batch: Per-member parameter pytree; each leaf carries a leading
            ``count`` axis.  Vmapped on axis 0 so the closure sees a per-member
            scalar slice ``m``.

    Returns:
        ``jax.Array`` of shape ``(count, member_field_size)``.

    Raises:
        ValueError: If ``value`` is a one-arg callable, or if a materialised
            value does not flatten to ``count * member_field_size`` (the message
            names the family and field).
    """
    count = int(fam.count)
    member_field_size = int(var.size)

    if callable(value):
        _reject_one_arg(fam, var, value)

        def _one(m):
            return jnp.asarray(value(coords, m)).reshape(-1)

        y0_pop = jax.vmap(_one, in_axes=0)(m_batch)  # (count, member_field_size)
    elif jnp.ndim(value) == 0:
        y0_pop = jnp.broadcast_to(jnp.asarray(value), (count, member_field_size))
    else:
        y0_pop = jnp.asarray(value)

    flat = jnp.reshape(y0_pop, (-1,))
    if flat.size != count * member_field_size:
        raise ValueError(
            f"family {fam.name!r} field {var.name!r}: value materialised to "
            f"size {flat.size}, but the per-member field flat size is "
            f"{member_field_size} over count={count} "
            f"(expected {count * member_field_size}, shape {var.shape})"
        )
    return flat.reshape(count, member_field_size)


def _reject_one_arg(fam, var, value) -> None:
    """Raise a loud ValueError if a family-field closure is not ``(coords, m)``.

    Probes the callable's arity host-side (positional params excluding ``self``
    /``*args``). A one-argument closure usually confuses the family
    ``value(coords, member)`` contract with component ``value(coords)``.

    Raises:
        ValueError: when the callable declares fewer than two required
            positional args and no ``*args``.
    """
    import inspect

    try:
        sig = inspect.signature(value)
    except (TypeError, ValueError):
        return  # builtins / C callables — can't introspect; let the trace decide.
    has_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    if has_var_positional:
        return
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    if len(positional) < 2:
        raise ValueError(
            f"family {fam.name!r} field {var.name!r}: value callable must accept "
            "two positional args (coords, m) — the member domain's node "
            "coordinates and the per-member parameter slice. A one-arg "
            f"value(coords) is rejected (got {len(positional)} required "
            "positional arg(s))."
        )


def _member_batch(fam) -> Mapping[str, "jax.Array"]:
    """Per-member parameter pytree for the vmap, with a guaranteed count axis.

    Vmapping over an empty ``member_params`` cannot infer the batch size, so we
    inject a reserved ``member_index`` axis of length ``count`` (mirrors the
    facade's ``m.member_index``).  When params are present they already carry
    the leading ``count`` axis.
    """
    count = int(fam.count)
    m_batch = {k: jnp.asarray(v) for k, v in dict(fam.member_params).items()}
    if not m_batch:
        offset = int(getattr(fam, "member_index_offset", 0))
        m_batch = {"member_index": jnp.arange(count) + offset}
    return m_batch


_MISSING = object()


def _family_field_value(values: Mapping, fam_name: str, var_name: str, default):
    """Resolve a family field's initial value with a bucket-group fallback.

    A ragged family lowers to one population group per static size bucket,
    named ``"<family>[<bucket>]"``.  A value keyed by the bucket group
    (``"p[0].c"``) wins; otherwise a value keyed by the authored family name
    (``"p.c"``) applies to EVERY bucket (a ``(coords, m)`` closure then
    evaluates against each bucket's own geometry).
    """
    value = values.get(f"{fam_name}.{var_name}", _MISSING)
    if value is _MISSING and "[" in fam_name:
        base = fam_name.split("[", 1)[0]
        value = values.get(f"{base}.{var_name}", _MISSING)
    return default if value is _MISSING else value


def assemble_family_pieces(families, values, default):
    """Build the member-major flat slabs for every family.

    Walks ``families`` in order; for each family assembles a
    ``(count, member_state_size)`` slab by writing each member field's
    materialisation into its column band, then flattens to the member-major flat
    block.  Returns the list of flat slabs to concatenate AFTER the component
    pieces.

    Both ``ProblemSpec`` and ``LoweredProblem`` route here: a [Family][diff_soup.problem._ir.family.Family]
    always carries its own member ``domain`` (node coordinates) and
    ``member_params``, so family-field ``(coords, m)`` closures are evaluable on
    BOTH sides — unlike plain components, where the lowered problem has no node
    coordinates and rejects callables.

    Args:
        families: Iterable of [Family][diff_soup.problem._ir.family.Family] populations
            (``eqset.families`` or ``LoweredProblem._families``).
        values: Mapping ``"<fam.name>.<field>" -> array | scalar | callable``.
        default: Value for any family field not present in ``values``.

    Returns:
        ``list[jax.Array]`` of member-major flat slabs, one per family.
    """
    pieces = []
    for fam in families:
        count = int(fam.count)
        member_state_size = int(fam.member_state_size)
        coords_fn = getattr(fam.domain, "node_coordinates", None)
        coords = coords_fn() if callable(coords_fn) else None
        m_batch = _member_batch(fam)

        slab = jnp.zeros((count, member_state_size))
        var_local_off = 0
        for var in fam.member_variables:
            value = _family_field_value(values, fam.name, var.name, default)
            field_pop = _materialise_family_field(fam, var, value, coords, m_batch)
            slab = slab.at[:, var_local_off : var_local_off + var.size].set(field_pop)
            var_local_off += int(var.size)
        pieces.append(slab.reshape(-1))
    return pieces
