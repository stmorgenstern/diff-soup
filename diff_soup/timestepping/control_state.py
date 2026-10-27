"""Bound control values and the control epoch carried through a solve.

[ControlState][diff_soup.timestepping.control_state.ControlState] pairs the
currently resolved control values with an integer *epoch* that increments at every
control discontinuity — an event affect that mutates a control, or a scheduled
control update. Both control-verb kinds bump this one epoch: a direct control
that switches value increments it through its affect event exactly as a derived
control does, even though a direct control contributes no residual row of its
own. Consumers that cache work derived from control values (linearizations,
factorizations, border solves) key that work on the epoch, so a control change
invalidates exactly the cached quantities that depended on it, and the
operational row table selects its active row from the same counter.

A problem that declares an operational control table (a static ordered set of
``current`` / ``voltage`` / ``power`` residual rows closing one algebraic
current unknown) additionally carries the *operating point* here: an ``int32``
``mode`` index into the table's static mode-name tuple, a scalar float
``target``, and an optional scalar ``current_seed`` for the algebraic current
unknown. The target may also vary in time as a polynomial in the time since a
given origin (``target_terms`` and ``target_origin``). Mode and target are
runtime data — changing them reuses the compiled solve — and a real change
bumps the epoch exactly once.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup._errors import ConfigurationError


def _as_scalar_float(value: Any, *, label: str) -> jax.Array:
    """Validate and return one scalar float leaf."""
    arr = jnp.asarray(value)
    if arr.shape != ():
        raise ConfigurationError(
            f"{label} must be a scalar (shape ()); got shape {arr.shape}. Pass one float value."
        )
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        raise ConfigurationError(
            f"{label} must be a float scalar; got dtype {arr.dtype}. "
            f"Pass a float value (it is cast to the solve dtype)."
        )
    return arr


def _as_terms(terms: Any) -> jax.Array:
    """Validate and return the one-dimensional float leaf of a target's time terms."""
    arr = jnp.asarray(terms)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ConfigurationError(
            f"target_terms must be a one-dimensional array of at least one coefficient "
            f"(of tau, tau**2, ...); got shape {arr.shape}."
        )
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        raise ConfigurationError(
            f"target_terms must hold floats; got dtype {arr.dtype}. Pass float coefficients "
            "(they are cast to the solve dtype)."
        )
    return arr


def _as_mode_index(mode: Union[int, jax.Array], *, label: str = "mode") -> jax.Array:
    """Validate and return one int32 mode-index leaf."""
    arr = jnp.asarray(mode)
    if arr.shape != () or not jnp.issubdtype(arr.dtype, jnp.integer):
        raise ConfigurationError(
            f"{label} must be a scalar integer index into the operational "
            f"table's mode tuple (or a mode name string); got shape "
            f"{arr.shape}, dtype {arr.dtype}."
        )
    if not isinstance(arr, jax.core.Tracer) and int(arr) < 0:
        raise ConfigurationError(
            f"{label} must be a non-negative index into the operational "
            f"table's mode tuple; got {int(arr)}."
        )
    return arr.astype(jnp.int32)


class ControlState(eqx.Module):
    """The control values a solve carries, with an epoch that counts their changes.

    Pass one as `control_state0=` to a solve. On a problem with an operating table, `ControlState.operating` builds it, and an affect event returns its update.

    Attributes:
        values: Control mapping, or `None`.
        epoch: Number of control changes so far, an `int32` scalar starting at zero.
        mode: Index of the active row of the operating table, or `None`.
        target: Value the active row drives to, or `None` outside an operating protocol.
    """

    values: Any
    epoch: jax.Array
    mode: Optional[jax.Array] = None
    target: Optional[jax.Array] = None
    current_seed: Optional[jax.Array] = None
    mode_name: Optional[str] = eqx.field(static=True, default=None)
    target_terms: Optional[jax.Array] = None
    target_origin: Optional[jax.Array] = None

    @classmethod
    def initial(cls, values: Any) -> "ControlState":
        """The state at solve start: the given values at epoch zero."""
        return cls(values=values, epoch=jnp.zeros((), dtype=jnp.int32))

    @classmethod
    def operating(
        cls,
        *,
        mode: Union[int, str, jax.Array],
        target: Any,
        current_seed: Any = None,
        values: Any = None,
        target_terms: Any = None,
        target_origin: Any = None,
    ) -> "ControlState":
        """Build the state of an operating protocol, an active row and its target, at epoch zero.

        Args:
            mode: Index of the active row in the operating table, or its name.
            target: Float scalar the active row drives to.
            current_seed: Float scalar starting value of the current unknown, or `None`.
            values: Control mapping carried alongside, or `None`.
            target_terms: Coefficients of `tau`, `tau**2`, and so on, added to `target`, with `tau` the time since `target_origin`.
            target_origin: Time that `tau` counts from, given together with `target_terms`.

        Returns:
            A `ControlState` with `epoch` zero.
        """
        if isinstance(mode, str):
            mode_idx, mode_name = None, mode
        else:
            mode_idx, mode_name = _as_mode_index(mode), None
        seed = (
            None if current_seed is None else _as_scalar_float(current_seed, label="current_seed")
        )
        if (target_terms is None) != (target_origin is None):
            raise ConfigurationError(
                "a target that varies in time needs both target_terms (the coefficients of "
                "tau, tau**2, ...) and target_origin (the time tau counts from); got only "
                f"{'target_terms' if target_origin is None else 'target_origin'}."
            )
        terms = None if target_terms is None else _as_terms(target_terms)
        origin = (
            None
            if target_origin is None
            else _as_scalar_float(target_origin, label="target_origin")
        )
        return cls(
            values=values,
            epoch=jnp.zeros((), dtype=jnp.int32),
            mode=mode_idx,
            target=_as_scalar_float(target, label="target"),
            current_seed=seed,
            mode_name=mode_name,
            target_terms=terms,
            target_origin=origin,
        )

    @property
    def is_operating(self) -> bool:
        """Whether this state carries an operating mode and target."""
        return self.target is not None

    def target_at(self, t: Any) -> jax.Array:
        """Return the target at time `t`, with the polynomial terms added when present.

        Args:
            t: Time.

        Returns:
            The target as a scalar array.
        """
        target = jnp.asarray(self.target)
        if self.target_terms is None:
            return target
        terms = jnp.asarray(self.target_terms)
        tau = (jnp.asarray(t) - jnp.asarray(self.target_origin)).astype(target.dtype)
        rising = jnp.zeros_like(target)
        for index in range(int(terms.shape[0]) - 1, -1, -1):
            rising = (rising + terms[index].astype(target.dtype)) * tau
        return target + rising

    def bump(self) -> "ControlState":
        """Return a copy with the epoch advanced by one."""
        return eqx.tree_at(lambda s: s.epoch, self, self.epoch + 1)

    def with_values(self, values: Any) -> "ControlState":
        """Return a copy with new control values and the epoch advanced by one.

        Args:
            values: The new control mapping.

        Returns:
            The updated `ControlState`.
        """
        bumped = self.bump()
        return eqx.tree_at(lambda s: s.values, bumped, values, is_leaf=lambda leaf: leaf is None)

    def with_operating(
        self,
        *,
        mode: Optional[Union[int, jax.Array]] = None,
        target: Any = None,
        current_seed: Any = None,
    ) -> "ControlState":
        """Return a copy with a new mode or target, advancing the epoch only when either changes.

        Use it in an affect event to switch rows mid-solve. It needs an operating state whose mode is an integer index.

        Args:
            mode: New row index, or `None` to keep the mode.
            target: New float target, or `None` to keep it.
            current_seed: New starting value of the current unknown, or `None` to keep it.

        Returns:
            The updated `ControlState`.
        """
        if self.target is None:
            raise ConfigurationError(
                "with_operating() requires an operational state; construct it "
                "with ControlState.operating(mode=..., target=...)."
            )
        if isinstance(mode, str):
            raise ConfigurationError(
                "with_operating() takes an integer mode index; resolve the "
                "name with the problem's operational_modes() tuple first."
            )
        if self.mode is None:
            raise ConfigurationError(
                "with_operating() requires a resolved mode index; this state "
                "carries an unresolved mode name. Construct with an integer "
                "mode, or pass the state through a solve entry first."
            )
        new_mode = self.mode if mode is None else _as_mode_index(mode)
        new_target = (
            self.target
            if target is None
            else _as_scalar_float(target, label="target").astype(self.target.dtype)
        )
        if current_seed is None:
            new_seed = self.current_seed
        else:
            new_seed = _as_scalar_float(current_seed, label="current_seed").astype(
                self.target.dtype
            )
        changed = jnp.logical_or(new_mode != self.mode, new_target != self.target)
        new_epoch = self.epoch + changed.astype(jnp.int32)
        return ControlState(
            values=self.values,
            epoch=new_epoch,
            mode=new_mode,
            target=new_target,
            current_seed=new_seed,
            mode_name=None,
            target_terms=self.target_terms,
            target_origin=self.target_origin,
        )
