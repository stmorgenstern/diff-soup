# ``spec.control.*`` authoring namespace.
"""``spec.control.*`` namespace.

Each method appends a [ControlSpec][] to the parent ProblemSpec.  A control is
a named time-dependent protocol.  By default ``fn`` is called at
residual-evaluation time; a solve may instead override every declared control
through an explicit ``control_state0`` mapping, resolved once before the solve
runs, which replaces ``fn`` with a frozen value for the rest of the solve.

Two verbs, split by where the controlled value comes from:

* [direct][diff_soup.problem._spec.control_namespace._ControlNamespace.direct]
  prescribes the value outright.  Nothing has to be solved for to realize it,
  so it declares no global variable and no constraint.
* [derived][diff_soup.problem._spec.control_namespace._ControlNamespace.derived]
  drives a quantity the solve assembles — a total current, an integrated flux.
  ``binds_to`` names the free global that carries it, and the control replaces
  the closing row of the constraint that closes that global.

[potentiostatic][diff_soup.problem._spec.control_namespace._ControlNamespace.potentiostatic]
and
[galvanostatic][diff_soup.problem._spec.control_namespace._ControlNamespace.galvanostatic]
are the electrochemical spellings of the two verbs: a set cell voltage is a
direct control, a set cell current is a derived-quantity one.  Each alias
declares exactly what the general verb declares.

[operational][diff_soup.problem._spec.control_namespace._ControlNamespace.operational]
declares something else again: a static table of terminal residual rows —
current, voltage, power — closing one algebraic current unknown, of which
exactly one is active at runtime.  It appends a constraint rather than a
``ControlSpec``, and its mode and target are carried in the solve's
[ControlState][diff_soup.timestepping.control_state.ControlState] instead of
being authored as a callable.

A control is authored in dimensional units, like a Dirichlet ``value_fn``: ``fn`` is called
with dimensional time and returns a dimensional value, whatever ``spec.time_scale`` and the
driven variable's ``scale`` are.  The recipe's scales convert on both sides,
so a control and a boundary condition in one model always run on the same
clock.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real

from .specs import ControlSpec, OperationalControlSpec

_OPERATIONAL_MODES = ("current", "voltage", "power")


class _ControlNamespace:
    """The control declarations of a recipe, reached as `spec.control`."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:
        self._spec = spec

    def direct(self, *, name: str, fn: Callable) -> None:
        """Declare a direct control, a time-dependent input read in closures as `q.control.<name>`.

        `fn` receives dimensional time and returns the dimensional value.
        `pdesolve(..., control_state0={name: value})` replaces it for one solve.

        Args:
            name: Control name.
            fn: Function `lambda t: ...` of dimensional time.
        """
        self._spec._controls.append(ControlSpec(name=name, kind="direct", fn=fn))

    def derived(self, *, name: str, fn: Callable, binds_to: str) -> None:
        """Declare a derived control that drives a constrained global to `fn(t)`.

        Exactly one constraint must close the global `binds_to`, and the control
        replaces that constraint's row with `global = fn(t)`. Each global takes one
        driving control.

        Args:
            name: Control name.
            fn: Function `lambda t: ...` of dimensional time returning the global's dimensional value.
            binds_to: Name of the global the control drives.
        """
        self._spec._controls.append(
            ControlSpec(name=name, kind="derived", fn=fn, binds_to=binds_to)
        )

    def potentiostatic(self, *, name: str, fn: Callable) -> None:
        """Declare a set-voltage control, the same as `direct`.

        Args:
            name: Control name.
            fn: Voltage as a function `lambda t: ...` of dimensional time.
        """
        self.direct(name=name, fn=fn)

    def galvanostatic(self, *, name: str, fn: Callable, binds_to: str) -> None:
        """Declare a set-current control, the same as `derived`.

        Args:
            name: Control name.
            fn: Current as a function `lambda t: ...` of dimensional time.
            binds_to: Name of the global current the control drives.
        """
        self.derived(name=name, fn=fn, binds_to=binds_to)

    def operational(
        self,
        *,
        name: str,
        current: str | None = None,
        voltage: str,
        current_unknown: str,
        modes: Sequence[str] = _OPERATIONAL_MODES,
        seed_voltage_threshold: float = 1e-6,
        current_from_residual: str | None = None,
    ) -> None:
        """Declare an operating table of current, voltage and power rows that close one current unknown.

        The active row and its target come from an operating
        [`ControlState`][diff_soup.ControlState] passed as `control_state0`, and
        switching them reuses the compiled solve. Pass exactly one of `current` and
        `current_from_residual`.

        Args:
            name: Table name.
            current: Name of the scalar observation giving the current.
            voltage: Name of the scalar observation giving the voltage.
            current_unknown: Name of the global current the table closes.
            modes: Rows of the table, distinct entries of `"current"`, `"voltage"`, `"power"`, indexed by the mode.
            seed_voltage_threshold: Voltage magnitude above which a power mode without a current seed starts from `target / V`.
            current_from_residual: Name of a field whose summed residual rows give the current.
        """
        if (current is None) == (current_from_residual is None):
            given = "both" if current is not None else "neither"
            raise ValueError(
                f"operational control {name!r} needs exactly one source of its "
                f"current, and {given} was given: pass current=<name of a scalar "
                "observation> or current_from_residual=<name of a field whose "
                "assembled residual rows sum to the current>"
            )
        modes = tuple(modes)
        if not modes:
            raise ValueError(
                f"operational control {name!r} declares an empty mode tuple; "
                f"list at least one mode from {_OPERATIONAL_MODES!r}"
            )
        seen: set = set()
        for m in modes:
            if m not in _OPERATIONAL_MODES:
                raise ValueError(
                    f"operational control {name!r} declares unknown mode {m!r}; "
                    f"supported modes are {_OPERATIONAL_MODES!r}"
                )
            if m in seen:
                raise ValueError(
                    f"operational control {name!r} declares duplicate mode {m!r}; "
                    f"list each mode once"
                )
            seen.add(m)
        if isinstance(seed_voltage_threshold, (bool, str)) or not isinstance(
            seed_voltage_threshold, Real
        ):
            raise TypeError(
                f"operational control {name!r} seed_voltage_threshold must be "
                f"a real scalar, got {type(seed_voltage_threshold).__name__}; pass "
                "the voltage magnitude below which a default power current seed is unsafe"
            )
        if not math.isfinite(float(seed_voltage_threshold)) or float(seed_voltage_threshold) < 0.0:
            raise ValueError(
                f"operational control {name!r} seed_voltage_threshold must be "
                "a finite non-negative real scalar; pass the voltage magnitude "
                "below which a default power current seed is unsafe"
            )
        threshold = float(seed_voltage_threshold)
        self._spec._constraints.append(
            OperationalControlSpec(
                name=name,
                current=current,
                voltage=voltage,
                closes=current_unknown,
                modes=modes,
                seed_voltage_threshold=threshold,
                field=current_from_residual,
            )
        )
