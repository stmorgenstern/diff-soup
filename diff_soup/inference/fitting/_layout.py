"""The parameter layout a fit is a point of, and the points the analysis verbs take.

A fit seeds or anchors another problem when the two lay their parameters out the same
way: the same handle names, kinds and sizes in the same columns, with the same owners.
The study content hash also covers the solver, the data and the starting values, so it
stays the key only where a verb reuses what a fit recorded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def schema_layout(schema: Any) -> tuple:
    """Each entry's handle name, kind and size, and the columns of each owner, sorted by name.

    Handle identity is left out, so a rebuilt recipe with the same declarations has the
    same layout.
    """
    return tuple(
        sorted(
            (
                str(entry.handle.name),
                str(getattr(entry.handle, "kind", "")),
                int(entry.size),
                tuple(
                    (
                        None if occurrence.owner is None else str(occurrence.owner),
                        tuple(int(column) for column in occurrence.columns),
                    )
                    for occurrence in entry.occurrences
                ),
            )
            for entry in schema.entries
        )
    )


def layout_text(schema: Any) -> str:
    """The layout of ``schema`` as JSON, or ``""`` for ``None``; `OptResult.save` writes it."""
    return "" if schema is None else json.dumps(schema_layout(schema))


def describe_layout(schema: Any) -> str:
    """The columns of ``schema`` as text, for example ``rate: 0, source@cell-a: 1``."""
    labels = []
    for name, _, _, occurrences in schema_layout(schema):
        for owner, columns in occurrences:
            label = name if owner is None else f"{name}@{owner}"
            labels.append((columns, f"{label}: {', '.join(str(c) for c in columns)}"))
    return "; ".join(text for _, text in sorted(labels))


_AS_POINT = (
    "Pass the point as a flat vector in this problem's column order (`fit.params` when the "
    "columns hold the same handles) or as a mapping from every fitted handle to its value."
)


def require_layout(problem: Any, schema: Any, *, where: str) -> None:
    """Refuse a fit whose columns hold other handles than ``problem``'s.

    Raises:
        ValueError: If the two layouts differ.
    """
    own = problem.parameter_schema
    if schema_layout(schema) != schema_layout(own):
        raise ValueError(
            f"{where}: the fit lays its parameters out as ({describe_layout(schema)}), and "
            f"this problem as ({describe_layout(own)}). {_AS_POINT}"
        )


def mapping_point(problem: Any, values: Mapping, *, where: str) -> np.ndarray:
    """The flat point of ``problem`` that a mapping from every fitted handle to its value names.

    A varying handle takes a mapping from owner to value.

    Raises:
        ValueError: If a handle is missing, unknown, or has the wrong number of values.
    """
    schema = problem.parameter_schema
    point = np.array(np.asarray(problem.x0), copy=True)
    assigned = set()
    for handle, value in values.items():
        try:
            entry = schema.entry(handle)
        except KeyError:
            raise ValueError(
                f"{where}: {getattr(handle, 'name', handle)!r} is not a fitted handle of this "
                f"problem, whose handles are {[e.handle.name for e in schema.entries]}."
            ) from None
        assigned.add(entry.handle.identity)
        for occurrence in entry.occurrences:
            if occurrence.owner is None:
                block = value
            elif isinstance(value, Mapping) and occurrence.owner in value:
                block = value[occurrence.owner]
            else:
                raise ValueError(
                    f"{where}: {entry.handle.name!r} varies by owner; give a mapping from "
                    f"each owner to its value, owner {occurrence.owner!r} included."
                )
            block = np.ravel(np.asarray(block))
            if block.size != entry.size:
                raise ValueError(
                    f"{where}: {entry.handle.name!r} needs {entry.size} values; got {block.size}."
                )
            point[np.asarray(occurrence.columns)] = block
    missing = [e.handle.name for e in schema.entries if e.handle.identity not in assigned]
    if missing:
        raise ValueError(
            f"{where}: a mapping point needs a value for every fitted handle; missing {missing}."
        )
    return point


def resolve_point(problem: Any, at: Any, *, where: str) -> Any:
    """``at`` as a flat point of ``problem``.

    A fit (an `OptResult` or `ExperimentFit`) gives its `params` after the layout check,
    and a mapping is assembled by handle. A caller that needs a schema refuses a fit
    without one first. Anything else is returned unchanged for the caller's own shape check.
    """
    from diff_soup.inference.fitting.inference_problem import _is_structural_schema

    if hasattr(at, "params") and hasattr(at, "parameter_schema"):
        schema = at.parameter_schema
        # A fit without a schema (one loaded without problem=) and a structural schema,
        # which labels one problem's own columns and may leave a handle out, are checked
        # by the caller's shape check alone.
        if schema is not None and not _is_structural_schema(schema):
            require_layout(problem, schema, where=where)
        return at.params
    if isinstance(at, Mapping):
        return mapping_point(problem, at, where=where)
    return at
