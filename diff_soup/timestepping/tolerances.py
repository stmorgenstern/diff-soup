"""Tolerance specification and layout-aware resolution for the solver suite.

[ToleranceSpec][] captures tolerance intent once, keeping the four
distinct uses separate — time-integration error control, consistent
initialization, the nonlinear corrector's acceptance band, and event
localization. [resolve_tolerances][] turns the absolute-tolerance intent
into one flat per-DOF vector against a problem layout, so engines consume a
[ResolvedTolerances][] record instead of re-interpreting the user's
spelling.

Absolute tolerance accepts four spellings:

* a positive scalar, broadcast to every row;
* a flat vector with one entry per DOF (state rows first, then global rows);
* a mapping from layout keys to scalars or per-field vectors. Keys may name
  an exact field (``("anode", "c")`` or ``"anode.c"``), a global row by
  name, a DAE role (``"differential"`` / ``"algebraic"``), a layout
  partition (``"state"`` / ``"global"``), or ``"default"``. When several
  keys cover the same row, the most specific wins: exact field over role
  over partition over default.

Resolution records, per layout entry, which rule produced its rows
(``provenance``) and a fingerprint of the layout it was resolved against,
so a resolved record can be validated, serialized, and compared across
solves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from diff_soup.timestepping.consistent_ic.pin_differential import _structure

_ROLE_KEYS = ("differential", "algebraic")
_PARTITION_KEYS = ("state", "global")
_DEFAULT_KEY = "default"
_ALGEBRAIC_ERROR_CONTROL = ("include", "exclude")


def _field_name(entry_key: Tuple[str, str]) -> str:
    return f"{entry_key[0]}.{entry_key[1]}"


def layout_fingerprint(
    layout_entries: Sequence[Tuple[Tuple[str, str], int, int, tuple]],
    global_entries: Sequence[Tuple[str, int, int]],
    n_dofs: int,
) -> str:
    """Stable 16-hex-digit digest of a flat state layout.

    The digest covers every field's name, offset, size, and shape, every
    global row's name, offset, and size, and the total DOF count. Two solves
    share a fingerprint exactly when their tolerance resolution would place
    every value on the same rows.
    """
    canonical = json.dumps(
        {
            "fields": [
                [list(key), int(off), int(size), list(shape)]
                for key, off, size, shape in layout_entries
            ],
            "globals": [[name, int(off), int(size)] for name, off, size in global_entries],
            "n": int(n_dofs),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ToleranceSpec:
    """Per-solve tolerance request for `JNDF`, `JBDF`, and `JIDA`, resolved row by row against the problem.

    Pass it as `tolerances=` to a solve, and read the resolved record from `Solution.tolerances`. In an `atol` mapping the most specific key wins, and every row must be covered.

    Args:
        rtol: Positive relative tolerance.
        atol: Positive absolute tolerance, a scalar, one value per row, or a mapping keyed by field, global, role, partition, or `"default"`.
        ic_rtol: Relative tolerance of consistent initialization, `rtol` when `None`.
        ic_atol: Absolute tolerance of consistent initialization, in the forms of `atol`, which it copies when `None`.
        newton_tol_factor: Positive scale on the corrector's acceptance threshold, where values above one loosen it.
        event_tol: Positive tolerance stored on the resolved record and not read by the engines.
        algebraic_error_control: `"include"` keeps algebraic rows in the local error norm, and `"exclude"`, on `JIDA` only, drops them.
    """

    rtol: float = 1e-6
    atol: Any = 1e-6
    ic_rtol: Optional[float] = None
    ic_atol: Any = None
    newton_tol_factor: float = 1.0
    event_tol: Optional[float] = None
    algebraic_error_control: str = "include"

    def __post_init__(self) -> None:
        """Validate positivity and the algebraic-error-control spelling."""
        if not float(self.rtol) > 0.0:
            raise ValueError(f"rtol must be positive; got {self.rtol!r}.")
        if not float(self.newton_tol_factor) > 0.0:
            raise ValueError(f"newton_tol_factor must be positive; got {self.newton_tol_factor!r}.")
        if self.event_tol is not None and not float(self.event_tol) > 0.0:
            raise ValueError(f"event_tol must be positive; got {self.event_tol!r}.")
        if self.algebraic_error_control not in _ALGEBRAIC_ERROR_CONTROL:
            raise ValueError(
                "algebraic_error_control must be 'include' or 'exclude'; "
                f"got {self.algebraic_error_control!r}."
            )


class ResolvedTolerances(eqx.Module):
    """Tolerances resolved to one value per row of a problem's state, as carried on `Solution.tolerances`.

    It can be passed back as `tolerances=` for a problem with the same number of rows. `to_dict` and `from_dict` convert it to and from plain data.

    Attributes:
        rtol: Relative tolerance.
        atol: Absolute tolerance per row.
        ic_rtol: Relative tolerance of consistent initialization.
        ic_atol: Absolute tolerance per row for consistent initialization.
        newton_tol_factor: Scale on the corrector's acceptance threshold.
        event_tol: Event tolerance from the request, or `None`.
        algebraic_error_control: `"include"` or `"exclude"`.
        layout_fingerprint: Hash of the problem's row layout.
        provenance: For each field and global, the rule that set its `atol`.
    """

    rtol: float = eqx.field(static=True)
    atol: jnp.ndarray = eqx.field(static=False)
    ic_rtol: float = eqx.field(static=True)
    ic_atol: jnp.ndarray = eqx.field(static=False)
    newton_tol_factor: float = eqx.field(static=True)
    event_tol: Optional[float] = eqx.field(static=True)
    algebraic_error_control: str = eqx.field(static=True)
    layout_fingerprint: str = eqx.field(static=True)
    provenance: Tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def n_dofs(self) -> int:
        """Total DOF count the vectors were resolved for."""
        return int(self.atol.shape[0])

    def to_dict(self) -> dict:
        """JSON-safe dictionary capturing every resolved value."""
        return {
            "rtol": float(self.rtol),
            "atol": [float(v) for v in np.asarray(self.atol)],
            "ic_rtol": float(self.ic_rtol),
            "ic_atol": [float(v) for v in np.asarray(self.ic_atol)],
            "newton_tol_factor": float(self.newton_tol_factor),
            "event_tol": None if self.event_tol is None else float(self.event_tol),
            "algebraic_error_control": self.algebraic_error_control,
            "layout_fingerprint": self.layout_fingerprint,
            "provenance": list(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResolvedTolerances":
        """Rebuild a resolved record from ResolvedTolerances.to_dict output.

        See [ResolvedTolerances.to_dict][diff_soup.timestepping.tolerances.ResolvedTolerances.to_dict].
        """
        return cls(
            rtol=float(data["rtol"]),
            atol=jnp.asarray(data["atol"], dtype=jnp.float64),
            ic_rtol=float(data["ic_rtol"]),
            ic_atol=jnp.asarray(data["ic_atol"], dtype=jnp.float64),
            newton_tol_factor=float(data["newton_tol_factor"]),
            event_tol=None if data["event_tol"] is None else float(data["event_tol"]),
            algebraic_error_control=str(data["algebraic_error_control"]),
            layout_fingerprint=str(data["layout_fingerprint"]),
            provenance=tuple(data["provenance"]),
        )

    def stable_hash(self) -> str:
        """Content digest (16 hex digits) stable across processes."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _uses_role_keys(value: Any) -> bool:
    """Whether an atol spelling addresses rows through a DAE-role key."""
    return isinstance(value, Mapping) and any(str(k) in _ROLE_KEYS for k in value)


def resolve_for_problem(spec: ToleranceSpec, problem: Any) -> ResolvedTolerances:
    """Resolve ``spec`` against the flat layout of a solver-facing problem.

    Reads the field and global-row layout from the problem itself or from
    the authored problem behind it (a solver view keeps its
    authored problem as ``source``), sizes the vectors from the augmented
    state (state rows then global rows), and — when a spelling addresses
    rows by DAE role — derives the differential-row mask from the problem's
    structure. A problem with no field layout (one authored directly against
    the solver problem) still resolves scalar, flat-vector, role, and
    partition spellings; field-name keys then fail with the resolver's
    unknown-key error.

    Args:
        spec: The tolerance specification to resolve.
        problem: A problem satisfying the solver contract (accepts the
            object handed to a solver's ``solve``, before or after
            coercion).

    Returns:
        The [ResolvedTolerances][] for this problem's layout.
    """
    getter = getattr(problem, "layout_entries", None)
    if callable(getter):
        entries, global_entries = getter()
    else:
        entries = tuple(getattr(problem, "_layout_entries", ()) or ())
        global_entries = tuple(getattr(problem, "_global_layout_entries", ()) or ())
    n_dofs = int(getattr(problem, "n_u", 0) or 0) + int(getattr(problem, "n_lam", 0) or 0)
    if n_dofs <= 0:
        raise ValueError(
            "tolerance resolution needs the problem's augmented state size; "
            f"the problem reports n_u={getattr(problem, 'n_u', None)!r} and "
            f"n_lam={getattr(problem, 'n_lam', None)!r}. Pass a problem that "
            "satisfies the solver contract."
        )
    differential_mask = None
    if _uses_role_keys(spec.atol) or _uses_role_keys(spec.ic_atol):
        differential_mask = np.asarray(_structure(problem)[2])
    return resolve_tolerances(
        spec,
        n_dofs=n_dofs,
        layout_entries=entries,
        global_entries=global_entries,
        differential_mask=differential_mask,
    )


def _rows_and_provenance_from_mapping(
    atol: Mapping[Any, Any],
    layout_entries: Sequence[Tuple[Tuple[str, str], int, int, tuple]],
    global_entries: Sequence[Tuple[str, int, int]],
    n_dofs: int,
    differential_mask,
    *,
    what: str,
) -> tuple:
    field_names = {_field_name(key): (off, size) for key, off, size, _ in layout_entries}
    global_names = {name: (off, size) for name, off, size in global_entries}

    normalized: dict = {}
    for key, value in atol.items():
        if isinstance(key, tuple):
            if len(key) != 2 or not all(isinstance(part, str) for part in key):
                raise ValueError(
                    f"unknown {what} key {key!r}; a field key is a "
                    "(component, variable) pair of strings."
                )
            name = _field_name(key)
        elif isinstance(key, str):
            name = key
        else:
            raise ValueError(
                f"unknown {what} key {key!r}; keys are field names, global-row "
                "names, 'differential'/'algebraic', 'state'/'global', or 'default'."
            )
        if (
            name not in field_names
            and name not in global_names
            and name not in _ROLE_KEYS
            and name not in _PARTITION_KEYS
            and name != _DEFAULT_KEY
        ):
            known = sorted(field_names) + sorted(global_names)
            raise ValueError(
                f"unknown {what} key {name!r}; known fields and globals are "
                f"{known}, plus 'differential'/'algebraic', 'state'/'global', "
                "and 'default'."
            )
        normalized[name] = value

    if any(role in normalized for role in _ROLE_KEYS) and differential_mask is None:
        raise ValueError(
            f"the {what} mapping uses a role key ('differential'/'algebraic') "
            "but no differential_mask was provided; pass the per-row "
            "differential mask (1 differential, 0 algebraic) to resolve roles."
        )

    values = np.full(n_dofs, np.nan)
    rules: list = [None] * n_dofs

    def assign(rows, value, rule: str) -> None:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(len(rows), float(arr))
        elif arr.shape != (len(rows),):
            raise ValueError(
                f"the {what} value for {rule!r} has shape {arr.shape}; expected "
                f"a scalar or shape ({len(rows)},) matching its rows."
            )
        for i, row in enumerate(rows):
            values[row] = arr[i]
            rules[row] = rule

    state_rows = [row for _, off, size, _ in layout_entries for row in range(off, off + size)]
    global_rows = [row for _, off, size in global_entries for row in range(off, off + size)]

    # Least-specific first; later assignments overwrite, so the most
    # specific spelling wins: exact field > role > partition > default.
    if _DEFAULT_KEY in normalized:
        assign(list(range(n_dofs)), normalized[_DEFAULT_KEY], "default")
    for partition, rows in (("state", state_rows), ("global", global_rows)):
        if partition in normalized:
            assign(rows, normalized[partition], f"partition:{partition}")
    if differential_mask is not None:
        mask = np.asarray(differential_mask)
        for role, selector in (("differential", mask > 0), ("algebraic", mask <= 0)):
            if role in normalized:
                assign(list(np.flatnonzero(selector)), normalized[role], f"role:{role}")
    for name, (off, size) in {**field_names, **global_names}.items():
        if name in normalized:
            assign(list(range(off, off + size)), normalized[name], f"field:{name}")

    uncovered = [i for i in range(n_dofs) if rules[i] is None]
    if uncovered:
        raise ValueError(
            f"the {what} mapping does not cover rows {uncovered}; add a "
            "'default' entry or keys covering every row."
        )

    provenance = []
    for key, off, size, _ in layout_entries:
        entry_rules = {rules[row] for row in range(off, off + size)}
        provenance.append(entry_rules.pop() if len(entry_rules) == 1 else "mixed")
    for name, off, size in global_entries:
        entry_rules = {rules[row] for row in range(off, off + size)}
        provenance.append(entry_rules.pop() if len(entry_rules) == 1 else "mixed")
    return values, tuple(provenance)


def _resolve_atol(
    atol,
    layout_entries,
    global_entries,
    n_dofs: int,
    differential_mask,
    *,
    what: str,
) -> tuple:
    n_entries = len(layout_entries) + len(global_entries)
    if isinstance(atol, Mapping):
        values, provenance = _rows_and_provenance_from_mapping(
            atol, layout_entries, global_entries, n_dofs, differential_mask, what=what
        )
    elif np.ndim(atol) == 0:
        values = np.full(n_dofs, float(atol))
        provenance = ("scalar",) * n_entries
    else:
        values = np.asarray(atol, dtype=np.float64)
        if values.shape != (n_dofs,):
            raise ValueError(
                f"a flat {what} vector must have length {n_dofs} (one entry per "
                f"DOF); got shape {values.shape}."
            )
        provenance = ("flat",) * n_entries
    if not np.all(values > 0.0):
        bad = np.flatnonzero(~(values > 0.0))
        raise ValueError(f"every {what} entry must be positive; rows {bad.tolist()} are not.")
    return jnp.asarray(values), provenance


def resolve_tolerances(
    spec: ToleranceSpec,
    *,
    n_dofs: int,
    layout_entries: Sequence[Tuple[Tuple[str, str], int, int, tuple]] = (),
    global_entries: Sequence[Tuple[str, int, int]] = (),
    differential_mask=None,
) -> ResolvedTolerances:
    """Resolve a [ToleranceSpec][] against a flat state layout.

    Args:
        spec: The tolerance specification to resolve.
        n_dofs: Total number of DOFs (state rows plus global rows).
        layout_entries: Static field layout —
            ``((component, variable), offset, size, shape)`` per field.
        global_entries: Static global-row layout — ``(name, offset, size)``
            per global.
        differential_mask: Optional per-row mask (1 differential,
            0 algebraic). Required only when a mapping uses a role key.

    Returns:
        A [ResolvedTolerances][] with flat vectors, a layout
        fingerprint, and per-entry provenance.
    """
    atol, provenance = _resolve_atol(
        spec.atol, layout_entries, global_entries, n_dofs, differential_mask, what="atol"
    )
    if spec.ic_atol is None:
        ic_atol = atol
    else:
        ic_atol, _ = _resolve_atol(
            spec.ic_atol,
            layout_entries,
            global_entries,
            n_dofs,
            differential_mask,
            what="ic_atol",
        )
    return ResolvedTolerances(
        rtol=float(spec.rtol),
        atol=atol,
        ic_rtol=float(spec.rtol if spec.ic_rtol is None else spec.ic_rtol),
        ic_atol=ic_atol,
        newton_tol_factor=float(spec.newton_tol_factor),
        event_tol=None if spec.event_tol is None else float(spec.event_tol),
        algebraic_error_control=spec.algebraic_error_control,
        layout_fingerprint=layout_fingerprint(layout_entries, global_entries, n_dofs),
        provenance=provenance,
    )
