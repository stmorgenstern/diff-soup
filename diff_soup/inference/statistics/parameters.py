"""parameters.py — coordinate-aware parameterization for inference.

Statistical work on a fitted model happens in a flat *inference coordinate*
``z`` while the model itself consumes the *physical coordinate* ``x`` (the
learnable theta vector of a [InferenceProblem][diff_soup.inference.InferenceProblem]).
This module makes the map between the two explicit and keeps every result
labeled with the coordinate it lives in:

* [Transform][diff_soup.inference.statistics.parameters.Transform] and its concrete maps [Identity][diff_soup.inference.statistics.parameters.Identity], [Log][diff_soup.inference.statistics.parameters.Log],
  [Logit][diff_soup.inference.statistics.parameters.Logit], [BoundedAffine][diff_soup.inference.statistics.parameters.BoundedAffine], and [ScaledTransform][diff_soup.inference.statistics.parameters.ScaledTransform] are
  smooth elementwise changes of variables ``x = forward(z)`` with exact
  ``log |dx/dz|`` bookkeeping.
* [ParamBlock][diff_soup.inference.statistics.parameters.ParamBlock] binds one named slice of the inference vector to its
  transform plus metadata (shape, problem leaf path, parameter group, scale,
  and the coordinate its prior density is declared in).
* [Parameterization][diff_soup.inference.statistics.parameters.Parameterization] assembles blocks into the full ``z <-> x`` map,
  sums the change-of-variables correction, and is built from a
  [InferenceProblem][diff_soup.inference.InferenceProblem] via [Parameterization.from_inference_problem][diff_soup.inference.statistics.parameters.Parameterization.from_inference_problem]. Identity is the default; ``transforms="auto"`` infers a
  transform from the problem's declared bounds but always stores and
  displays the resolved choice. There is no global log default.
* [Parameterization.convert_covariance][diff_soup.inference.statistics.parameters.Parameterization.convert_covariance] is a *local delta-method*
  conversion of a covariance between coordinates; it returns a
  [ConvertedCovariance][diff_soup.inference.statistics.parameters.ConvertedCovariance] carrying the linearization point and the
  source/target labels, never a bare array.
* [ModeLabel][diff_soup.inference.statistics.parameters.ModeLabel] pairs a mode (e.g. a MAP point) with the coordinate of
  the density it maximizes — a MAP is not invariant under reparameterization,
  so a mode without its coordinate is meaningless.
* [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint], [Gauge][diff_soup.inference.statistics.parameters.Gauge], and [constrained][diff_soup.inference.statistics.parameters.constrained] impose
  hard linear equalities ``A x = b`` (physical constraints and gauge fixes)
  by exact null-space elimination: the returned
  [ConstrainedParameterization][diff_soup.inference.statistics.parameters.ConstrainedParameterization] works in a reduced coordinate ``u``
  entering through ``x_particular + null_basis @ u`` — the equalities are
  eliminated exactly, never through penalty or ridge terms.
* [non_centered][diff_soup.inference.statistics.parameters.non_centered] replaces one named block coordinate by its standardized
  coordinate while preserving the same physical parameter space. It composes
  with the block's existing transform and records the declared scale.

Everything here is an `equinox` module: pure, immutable, and safe under
``jax.jit`` and ``jax.vmap``. The differentiable paths use ``jax.numpy``
only; host numpy appears only in name/metadata handling.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "Transform",
    "Identity",
    "Log",
    "Logit",
    "BoundedAffine",
    "ScaledTransform",
    "ParamBlock",
    "Parameterization",
    "ConvertedCovariance",
    "ModeLabel",
    "COORDINATES",
    "LinearConstraint",
    "Gauge",
    "ConstrainedParameterization",
    "constrained",
    "non_centered",
]


#: The two coordinate labels used throughout this module: ``"z"`` is the flat
#: inference coordinate, ``"x"`` the physical (model/theta) coordinate.
COORDINATES = ("z", "x")


def _check_coordinate(coordinate: Any, *, what: str) -> None:
    """Reject a coordinate label outside the ``z``/``x`` vocabulary."""
    if coordinate not in COORDINATES:
        raise ValueError(
            f"{what} must be one of {COORDINATES}: 'z' is the flat inference "
            f"coordinate, 'x' the physical parameter coordinate; got {coordinate!r}."
        )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class Transform(eqx.Module):
    """A smooth, strictly monotone elementwise change of variables.

    ``forward`` maps the inference coordinate ``z`` to the physical
    coordinate ``x``; ``inverse`` maps back. ``log_abs_det_jacobian`` returns
    the **elementwise** ``log |dx/dz|`` with the same shape as ``z`` — callers
    that need the total correction sum it (as
    [Parameterization.log_abs_det_jacobian][diff_soup.inference.statistics.parameters.Parameterization.log_abs_det_jacobian] does).

    All methods must be pure ``jax.numpy`` functions, safe under ``jax.jit``
    and ``jax.vmap``.
    """

    @abc.abstractmethod
    def forward(self, z: Array) -> Array:
        """Map inference values ``z`` to physical values ``x``."""

    @abc.abstractmethod
    def inverse(self, x: Array) -> Array:
        """Map physical values ``x`` back to inference values ``z``."""

    @abc.abstractmethod
    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return the elementwise ``log |dx/dz|`` evaluated at ``z``."""


class Identity(Transform):
    """The identity map: the inference and physical coordinates coincide."""

    def forward(self, z: Array) -> Array:
        """Return ``z`` unchanged."""
        return jnp.asarray(z)

    def inverse(self, x: Array) -> Array:
        """Return ``x`` unchanged."""
        return jnp.asarray(x)

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return zeros: ``|dx/dz| = 1`` everywhere."""
        return jnp.zeros_like(jnp.asarray(z))


class Log(Transform):
    """Log-scale inference for a positive parameter: ``x = exp(z)``.

    The inference coordinate is ``z = log(x)``, so a parameter constrained to
    ``(0, inf)`` becomes unconstrained.
    """

    def forward(self, z: Array) -> Array:
        """Return ``exp(z)``."""
        return jnp.exp(jnp.asarray(z))

    def inverse(self, x: Array) -> Array:
        """Return ``log(x)``."""
        return jnp.log(jnp.asarray(x))

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return ``z`` itself: ``dx/dz = exp(z)``."""
        return jnp.asarray(z)


class Logit(Transform):
    """Logit-scale inference for a parameter in ``(0, 1)``: ``x = sigmoid(z)``.

    The inference coordinate is ``z = logit(x)``, so a unit-interval
    parameter becomes unconstrained.
    """

    def forward(self, z: Array) -> Array:
        """Return ``sigmoid(z)``."""
        return jax.nn.sigmoid(jnp.asarray(z))

    def inverse(self, x: Array) -> Array:
        """Return ``logit(x) = log(x) - log(1 - x)``."""
        x = jnp.asarray(x)
        return jnp.log(x) - jnp.log1p(-x)

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return ``log sigmoid(z) + log sigmoid(-z)``.

        ``dx/dz = sigmoid(z) (1 - sigmoid(z))``, evaluated in log space for
        numerical stability at large ``|z|``.
        """
        z = jnp.asarray(z)
        return jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)


class BoundedAffine(Transform):
    """Logit-style inference for a parameter in a finite interval.

    ``x = lower + (upper - lower) * sigmoid(z)`` maps the whole real line
    onto ``(lower, upper)``; the inverse is the affine-rescaled logit
    ``z = log(x - lower) - log(upper - x)``.

    The bounds are scalar metadata (compile-time constants), so the transform
    stays hashable and works directly under ``jax.jit``. Parameters with
    different bounds use separate blocks/instances.

    Attributes:
        lower: The (exclusive) scalar lower bound of the physical value.
        upper: The (exclusive) scalar upper bound of the physical value.
    """

    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)

    def __init__(self, lower: Any, upper: Any):
        """Validate and store the scalar interval bounds.

        Args:
            lower: Finite scalar lower bound.
            upper: Finite scalar upper bound, strictly greater than ``lower``.

        Raises:
            ValueError: If the bounds are not finite scalars or not strictly
                ordered.
        """
        try:
            lo = float(lower)
            hi = float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BoundedAffine bounds must be finite scalars; got lower={lower!r}, "
                f"upper={upper!r}."
            ) from exc
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"BoundedAffine bounds must be finite; got lower={lower!r}, upper={upper!r}."
            )
        if not hi > lo:
            raise ValueError(
                f"BoundedAffine requires lower < upper; got lower={lower!r}, upper={upper!r}."
            )
        self.lower = lo
        self.upper = hi

    def forward(self, z: Array) -> Array:
        """Return ``lower + (upper - lower) * sigmoid(z)``."""
        return self.lower + (self.upper - self.lower) * jax.nn.sigmoid(jnp.asarray(z))

    def inverse(self, x: Array) -> Array:
        """Return ``log(x - lower) - log(upper - x)``."""
        x = jnp.asarray(x)
        return jnp.log(x - self.lower) - jnp.log(self.upper - x)

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return ``log(upper - lower) + log sigmoid(z) + log sigmoid(-z)``."""
        z = jnp.asarray(z)
        width = jnp.log(self.upper - self.lower)
        return width + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)


class ScaledTransform(Transform):
    """Standardize another elementwise transform by a positive scale.

    If ``base`` maps a centered coordinate ``u`` to the physical value, this
    transform introduces a standardized coordinate ``z`` through
    ``u = scale * z`` and therefore maps

    ``x = base.forward(scale * z)``.

    The map is bijective whenever ``base`` is, and its elementwise Jacobian
    correction is the base correction at ``scale * z`` plus ``log(scale)``.
    This is the zero-offset affine composition used by [non_centered][diff_soup.inference.statistics.parameters.non_centered];
    keeping it as an explicit transform means covariance conversion, priors,
    and normalized posterior densities all see the same coordinate map.

    Attributes:
        base: The existing physical-coordinate transform being standardized.
        scale: Positive finite scalar multiplying the standardized coordinate.
    """

    base: Transform
    scale: float = eqx.field(static=True)

    def __init__(self, base: Transform, scale: Any):
        """Validate and store the composed transform and scalar scale."""
        if not isinstance(base, Transform):
            raise TypeError(
                f"ScaledTransform 'base' must be a Transform instance; got {type(base).__name__}."
            )
        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ScaledTransform 'scale' must be a positive finite scalar; got {scale!r}."
            ) from exc
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError(
                f"ScaledTransform 'scale' must be a positive finite scalar; got {scale!r}."
            )
        self.base = base
        self.scale = scale_value

    def forward(self, z: Array) -> Array:
        """Map standardized ``z`` through ``base(scale * z)``."""
        return self.base.forward(self.scale * jnp.asarray(z))

    def inverse(self, x: Array) -> Array:
        """Return the scaled inverse physical coordinate."""
        return self.base.inverse(jnp.asarray(x)) / self.scale

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return the elementwise Jacobian correction of the composition."""
        z = jnp.asarray(z)
        centered = self.scale * z
        return self.base.log_abs_det_jacobian(centered) + jnp.log(self.scale)


def _transform_label(transform: Transform) -> str:
    """A short display label for a resolved transform (used in reprs)."""
    if isinstance(transform, BoundedAffine):
        return f"BoundedAffine({transform.lower:g}, {transform.upper:g})"
    if isinstance(transform, ScaledTransform):
        return f"ScaledTransform({_transform_label(transform.base)}, {transform.scale:g})"
    return type(transform).__name__


# ---------------------------------------------------------------------------
# Parameter blocks
# ---------------------------------------------------------------------------


class ParamBlock(eqx.Module):
    """One named slice of the flat inference vector, with its transform.

    Attributes:
        name: The block name (a parameter scalar name, or a leaf/field name
            for multi-element blocks).
        transform: The block's [Transform][diff_soup.inference.statistics.parameters.Transform] (applied elementwise to the
            flat slice).
        shape: The original array shape of the parameter leaf; ``()`` for a
            scalar. Its element count must equal the slice length.
        path: The problem leaf path string, when known; ``""`` otherwise.
        group: The owning parameter-group name, or ``None``.
        scale: An optional characteristic scale of the physical value (used
            by callers for scaling decisions; not applied here).
        prior_coordinate: The coordinate — ``"z"`` or ``"x"`` — in which this
            block's prior density is declared. Defaults to ``"x"`` (a prior
            on the physical value).

    The block's position is exposed as the [`index`][diff_soup.inference.statistics.parameters.ParamBlock.index] property (a
    contiguous ``slice``); it is stored internally as a ``(start, stop)``
    tuple so the module stays hashable.
    """

    name: str = eqx.field(static=True)
    _span: tuple = eqx.field(static=True)
    transform: Transform
    shape: tuple = eqx.field(static=True)
    path: str = eqx.field(static=True)
    group: Optional[str] = eqx.field(static=True)
    scale: Optional[float] = eqx.field(static=True)
    prior_coordinate: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        index: slice,
        transform: Transform,
        *,
        shape: tuple = (),
        path: str = "",
        group: Optional[str] = None,
        scale: Optional[float] = None,
        prior_coordinate: str = "x",
    ):
        """Validate and freeze one parameter block.

        Args:
            name: Non-empty block name.
            index: Contiguous ``slice(start, stop)`` with ``0 <= start < stop``.
            transform: The block's [Transform][diff_soup.inference.statistics.parameters.Transform] instance.
            shape: Original leaf shape; element count must match the slice.
            path: Problem leaf path string (may be ``""``).
            group: Owning parameter-group name, or ``None``.
            scale: Optional positive characteristic scale.
            prior_coordinate: ``"z"`` or ``"x"``.

        Raises:
            TypeError: If ``name``/``index``/``transform`` have the wrong type.
            ValueError: If the slice is malformed, the shape does not match
                the slice length, ``scale`` is not positive, or
                ``prior_coordinate`` is not a known coordinate label.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"ParamBlock 'name' must be a non-empty string; got {name!r}.")
        if not isinstance(index, slice):
            raise TypeError(f"ParamBlock 'index' must be a slice; got {type(index).__name__}.")
        start, stop, step = index.start, index.stop, index.step
        if step not in (None, 1) or not isinstance(start, int) or not isinstance(stop, int):
            raise ValueError(
                f"ParamBlock 'index' must be a contiguous slice(start, stop) with "
                f"integer bounds and unit step; got {index!r}."
            )
        if start < 0 or stop <= start:
            raise ValueError(f"ParamBlock 'index' must satisfy 0 <= start < stop; got {index!r}.")
        if not isinstance(transform, Transform):
            raise TypeError(
                f"ParamBlock 'transform' must be a Transform instance (Identity(), "
                f"Log(), Logit(), BoundedAffine(lo, hi), ...); got "
                f"{type(transform).__name__}."
            )
        shape = tuple(int(s) for s in shape)
        if int(np.prod(shape, dtype=np.int64)) != stop - start:
            raise ValueError(
                f"ParamBlock 'shape' {shape} has {int(np.prod(shape, dtype=np.int64))} "
                f"element(s) but the index slice {index!r} covers {stop - start}."
            )
        if scale is not None and not float(scale) > 0.0:
            raise ValueError(f"ParamBlock 'scale' must be positive when given; got {scale!r}.")
        _check_coordinate(prior_coordinate, what="ParamBlock 'prior_coordinate'")
        self.name = name
        self._span = (start, stop)
        self.transform = transform
        self.shape = shape
        self.path = str(path)
        self.group = group
        self.scale = None if scale is None else float(scale)
        self.prior_coordinate = prior_coordinate

    @property
    def index(self) -> slice:
        """The block's contiguous position in the flat inference vector."""
        return slice(*self._span)

    @property
    def size(self) -> int:
        """The number of scalar entries this block covers."""
        return self._span[1] - self._span[0]

    def scalar_names(self) -> tuple:
        """Per-scalar names: the block name, indexed when the block is not scalar."""
        if self.size == 1:
            return (self.name,)
        return tuple(f"{self.name}[{i}]" for i in range(self.size))


# ---------------------------------------------------------------------------
# Mode labeling
# ---------------------------------------------------------------------------


class ModeLabel(eqx.Module):
    """A mode (e.g. a MAP point) labeled by its density coordinate.

    A posterior mode is the maximizer of a *density*, and densities change
    under reparameterization by the ``log |dx/dz|`` correction — so the mode
    in ``z`` does not map to the mode in ``x`` in general. Carrying the
    coordinate with the point keeps that distinction explicit: converting a
    mode to another coordinate means re-maximizing the transformed density,
    not just mapping the point through the transform.

    Attributes:
        value: The mode's parameter vector.
        coordinate: ``"z"`` (inference) or ``"x"`` (physical) — the
            coordinate whose density this point maximizes.
    """

    value: Array
    coordinate: str = eqx.field(static=True)

    def __init__(self, value: Any, coordinate: str):
        """Validate the coordinate label and store the mode.

        Raises:
            ValueError: If ``coordinate`` is not ``"z"`` or ``"x"``.
        """
        _check_coordinate(coordinate, what="ModeLabel 'coordinate'")
        self.value = jnp.asarray(value)
        self.coordinate = coordinate


# ---------------------------------------------------------------------------
# Covariance conversion result
# ---------------------------------------------------------------------------


class ConvertedCovariance(eqx.Module):
    """A covariance converted between coordinates by the local delta method.

    The conversion is a first-order (linearized) approximation: the covariance
    is propagated through the Jacobian of the coordinate map evaluated at one
    point, so it is accurate only in a neighborhood where the map is close to
    its linearization. The result therefore always carries that point and the
    coordinate labels — it is not interchangeable with a covariance computed
    directly in the target coordinate.

    Attributes:
        covariance: The converted covariance matrix, in the ``target``
            coordinate.
        at: The linearization point, expressed in the ``source`` coordinate.
        source: The coordinate (``"z"`` or ``"x"``) the input covariance
            lives in.
        target: The coordinate the converted covariance lives in.
    """

    covariance: Array
    at: Array
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------


class Parameterization(eqx.Module):
    """The full map between the flat inference vector ``z`` and physical ``x``.

    A tuple of [ParamBlock][diff_soup.inference.statistics.parameters.ParamBlock] covering the inference vector contiguously
    from index ``0``. Each block's transform is applied elementwise to its
    slice; [log_abs_det_jacobian][diff_soup.inference.statistics.parameters.Parameterization.log_abs_det_jacobian] sums the elementwise corrections into
    the scalar total used by change-of-variables density evaluation.

    Build one from a fit declaration with [from_inference_problem][diff_soup.inference.statistics.parameters.Parameterization.from_inference_problem] (identity
    transforms by default), or assemble blocks directly.

    Attributes:
        blocks: The parameter blocks, ordered by their position in the flat
            inference vector.
    """

    blocks: tuple

    def __init__(self, blocks: Any):
        """Validate that the blocks tile the inference vector contiguously.

        Args:
            blocks: An iterable of [ParamBlock][diff_soup.inference.statistics.parameters.ParamBlock].

        Raises:
            TypeError: If any entry is not a [ParamBlock][diff_soup.inference.statistics.parameters.ParamBlock].
            ValueError: If no blocks are given, block names collide, or the
                index slices do not cover ``0..n`` contiguously in order.
        """
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("Parameterization requires at least one ParamBlock.")
        for b in blocks:
            if not isinstance(b, ParamBlock):
                raise TypeError(
                    f"Parameterization blocks must be ParamBlock instances; got {type(b).__name__}."
                )
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"Parameterization block names must be unique; got {names}.")
        expected_start = 0
        for b in blocks:
            if b.index.start != expected_start:
                raise ValueError(
                    f"Parameterization blocks must tile the inference vector "
                    f"contiguously from index 0 in order; block {b.name!r} starts at "
                    f"{b.index.start} where {expected_start} was expected."
                )
            expected_start = b.index.stop
        self.blocks = blocks

    # -- structure ------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """The length of the flat inference (and physical) vector."""
        return self.blocks[-1].index.stop

    def names(self) -> tuple:
        """Per-scalar names of the flat vector, in order."""
        out: list = []
        for b in self.blocks:
            out.extend(b.scalar_names())
        return tuple(out)

    def resolved_transforms(self) -> dict:
        """The resolved transform of each block, keyed by block name."""
        return {b.name: b.transform for b in self.blocks}

    def __repr__(self) -> str:
        """Display each block with its resolved transform."""
        inner = ", ".join(f"{b.name}={_transform_label(b.transform)}" for b in self.blocks)
        return f"Parameterization({inner})"

    # -- the coordinate map ----------------------------------------------

    def transform(self, z: Array) -> Array:
        """Map a flat inference vector ``z`` to the flat physical vector ``x``."""
        z = jnp.asarray(z)
        return jnp.concatenate([b.transform.forward(z[b.index]) for b in self.blocks])

    def inverse(self, x: Array) -> Array:
        """Map a flat physical vector ``x`` back to the inference vector ``z``."""
        x = jnp.asarray(x)
        return jnp.concatenate([b.transform.inverse(x[b.index]) for b in self.blocks])

    def log_abs_det_jacobian(self, z: Array) -> Array:
        """Return the scalar total ``log |dx/dz|`` at ``z`` (summed over blocks)."""
        z = jnp.asarray(z)
        total = jnp.zeros((), dtype=z.dtype)
        for b in self.blocks:
            total = total + jnp.sum(b.transform.log_abs_det_jacobian(z[b.index]))
        return total

    # -- construction from a fit declaration ------------------------------

    @classmethod
    def from_inference_problem(
        cls,
        problem: Any,
        *,
        transforms: Union[str, dict] = "identity",
    ) -> "Parameterization":
        """Build a parameterization over a problem's learnable theta vector.

        One scalar block is created per entry of ``problem.theta_names``,
        in the flat-vector order the problem uses, with parameter-group
        attribution read from the problem's group/column map. Only the
        theta block is parameterized; auxiliary decision variables (e.g.
        multishoot window states) are outside the inference vector.

        Args:
            problem: An [InferenceProblem][diff_soup.inference.InferenceProblem] (or an
                object exposing ``theta_names`` and, optionally,
                ``param_group_names``/``param_group_cols`` and the bounds
                surface used by ``transforms="auto"``).
            transforms: How to choose each block's transform:

                * ``"identity"`` (default) — every parameter is inferred on
                  its physical scale.
                * a dict mapping parameter names and/or group names to
                  [Transform][diff_soup.inference.statistics.parameters.Transform] instances; a parameter-name key takes
                  precedence over its group's key, and unnamed parameters
                  stay identity.
                * ``"auto"`` — infer from the problem's declared bounds:
                  a ``(0, inf)`` bound gives [Log][diff_soup.inference.statistics.parameters.Log], a finite
                  ``(lo, hi)`` bound gives [BoundedAffine][diff_soup.inference.statistics.parameters.BoundedAffine], and
                  anything else (including no declared bound) stays
                  [Identity][diff_soup.inference.statistics.parameters.Identity]. The resolved choice is stored on each
                  block and shown by ``repr`` and
                  [resolved_transforms][diff_soup.inference.statistics.parameters.Parameterization.resolved_transforms].

        Returns:
            The assembled [Parameterization][diff_soup.inference.statistics.parameters.Parameterization].

        Raises:
            ValueError: If the problem has no learnable theta scalars, a
                dict key names no parameter or group, or ``transforms`` is a
                string other than ``"identity"``/``"auto"`` — in particular
                there is no global log mode: a log scale must be chosen per
                parameter (or inferred from a declared ``(0, inf)`` bound) so
                the choice stays visible.
            TypeError: If a dict value is not a [Transform][diff_soup.inference.statistics.parameters.Transform] instance.
        """
        names = tuple(getattr(problem, "theta_names", ()) or ())
        if not names:
            raise ValueError(
                "Parameterization.from_inference_problem: the problem exposes no "
                "learnable theta scalars (theta_names is empty)."
            )
        group_names = tuple(getattr(problem, "param_group_names", ()) or ())
        group_cols = tuple(getattr(problem, "param_group_cols", ()) or ())
        group_of: dict = {}
        for gname, cols in zip(group_names, group_cols):
            for col in cols:
                if col < len(names):
                    group_of[int(col)] = gname

        resolved = cls._resolve_transforms(transforms, names, group_of, problem)
        blocks = tuple(
            ParamBlock(
                name,
                slice(i, i + 1),
                resolved[i],
                shape=(),
                path="",
                group=group_of.get(i),
            )
            for i, name in enumerate(names)
        )
        return cls(blocks)

    @staticmethod
    def _resolve_transforms(
        transforms: Union[str, dict],
        names: tuple,
        group_of: dict,
        problem: Any,
    ) -> tuple:
        """Resolve the per-scalar transforms for [from_inference_problem][diff_soup.inference.statistics.parameters.Parameterization.from_inference_problem]."""
        n = len(names)
        if isinstance(transforms, str):
            if transforms == "identity":
                return tuple(Identity() for _ in range(n))
            if transforms == "auto":
                return Parameterization._auto_transforms(names, problem)
            raise ValueError(
                f"transforms must be 'identity' (the default), 'auto' (infer from "
                f"declared bounds), or a dict mapping parameter/group names to "
                f"Transform instances; got {transforms!r}. There is no global log "
                f"mode: a log scale must be chosen per parameter (or inferred from "
                f"a declared (0, inf) bound) so the choice stays visible."
            )
        if not isinstance(transforms, dict):
            raise ValueError(
                f"transforms must be 'identity', 'auto', or a dict of Transform "
                f"instances; got {type(transforms).__name__}."
            )

        valid_keys = set(names) | set(group_of.values())
        unknown = sorted(set(transforms) - valid_keys)
        if unknown:
            raise ValueError(
                f"transforms names unknown parameter/group key(s) {unknown}; valid "
                f"parameter names are {sorted(names)} and group names "
                f"{sorted(set(group_of.values()))}."
            )
        for key, value in transforms.items():
            if not isinstance(value, Transform):
                raise TypeError(
                    f"transforms[{key!r}] must be a Transform instance (e.g. Log()); "
                    f"got {type(value).__name__}."
                )

        out: list = []
        for i, name in enumerate(names):
            if name in transforms:
                out.append(transforms[name])
            elif group_of.get(i) in transforms:
                out.append(transforms[group_of[i]])
            else:
                out.append(Identity())
        return tuple(out)

    @staticmethod
    def _auto_transforms(names: tuple, problem: Any) -> tuple:
        """Infer per-scalar transforms from the problem's declared bounds.

        Recognized patterns per parameter: a ``(0, inf)`` bound gives
        [Log][diff_soup.inference.statistics.parameters.Log]; a finite ``(lo, hi)`` bound gives
        [BoundedAffine][diff_soup.inference.statistics.parameters.BoundedAffine]. Any other bound — including a positive lower
        bound with no upper bound, a one-sided upper bound, or no declared
        bound at all — stays [Identity][diff_soup.inference.statistics.parameters.Identity]; use an explicit transforms
        dict for those.
        """
        from diff_soup.inference.fitting.bounds import resolve_decision_bounds

        resolved = resolve_decision_bounds(problem)
        n = len(names)
        if resolved is None:
            return tuple(Identity() for _ in range(n))
        lb = np.asarray(resolved[0], dtype=np.float64)
        ub = np.asarray(resolved[1], dtype=np.float64)
        out: list = []
        for i in range(n):
            lo, hi = lb[i], ub[i]
            if lo == 0.0 and np.isposinf(hi):
                out.append(Log())
            elif np.isfinite(lo) and np.isfinite(hi):
                out.append(BoundedAffine(lo, hi))
            else:
                out.append(Identity())
        return tuple(out)

    # -- labeled covariance conversion ------------------------------------

    def convert_covariance(
        self,
        cov: Array,
        *,
        at: Array,
        source: str,
        target: str,
    ) -> ConvertedCovariance:
        """Convert a covariance between coordinates by the local delta method.

        The input covariance is propagated through the Jacobian of the
        coordinate map linearized at ``at``: for ``source="z"``,
        ``target="x"`` the result is ``J cov Jᵀ`` with ``J = dx/dz`` at
        ``at``; the reverse direction uses ``dz/dx``. This is a first-order
        local approximation — the result object records the linearization
        point and both coordinate labels so that provenance travels with the
        numbers.

        Args:
            cov: The ``(n, n)`` covariance matrix in the ``source`` coordinate.
            at: The linearization point, expressed in the ``source``
                coordinate.
            source: The coordinate ``cov`` lives in (``"z"`` or ``"x"``).
            target: The coordinate to convert to (``"z"`` or ``"x"``).

        Returns:
            A [ConvertedCovariance][diff_soup.inference.statistics.parameters.ConvertedCovariance] carrying the converted matrix, the
            linearization point, and the coordinate labels.

        Raises:
            ValueError: If a coordinate label is unknown or the shapes do not
                match this parameterization.
        """
        _check_coordinate(source, what="convert_covariance 'source'")
        _check_coordinate(target, what="convert_covariance 'target'")
        cov = jnp.asarray(cov)
        at = jnp.asarray(at)
        n = self.n_parameters
        if cov.shape != (n, n) or at.shape != (n,):
            raise ValueError(
                f"convert_covariance expects cov of shape {(n, n)} and at of shape "
                f"{(n,)} for this {n}-parameter parameterization; got cov "
                f"{cov.shape} and at {at.shape}."
            )
        if source == target:
            converted = cov
        elif source == "z":
            jac = jax.jacfwd(self.transform)(at)
            converted = jac @ cov @ jac.T
        else:
            jac = jax.jacfwd(self.inverse)(at)
            converted = jac @ cov @ jac.T
        return ConvertedCovariance(covariance=converted, at=at, source=source, target=target)


# ---------------------------------------------------------------------------
# Hard linear constraints and gauge fixing
# ---------------------------------------------------------------------------


class LinearConstraint(eqx.Module):
    """Rows of ``A x = b`` on the flat physical coordinate ``x``.

    Coefficients are addressed through named blocks of a
    [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]: ``coefficients`` maps a block name to that
    block's coefficient array, and every entry of ``x`` outside the named
    blocks has coefficient zero. One [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint] may carry
    several rows — give each block a ``(n_rows, block_size)`` coefficient
    array (a scalar or 1-D array means one row) and an ``rhs`` with one entry
    per row (a scalar right-hand side is broadcast to every row).

    The rows are eliminated *exactly* by [constrained][diff_soup.inference.statistics.parameters.constrained]; they are never
    turned into penalty or ridge terms. Because the rows are linear in ``x``,
    they may only address blocks whose transform is [Identity][diff_soup.inference.statistics.parameters.Identity] — a
    linear relation in ``x`` is nonlinear in the inference coordinate of a
    transformed block.

    Example: ``LinearConstraint({"c": [1.0, 1.0, 1.0]}, rhs=1.0)`` declares
    that the three scalars of block ``"c"`` sum to one.

    Attributes:
        coefficients: Mapping of block name to a ``(n_rows, block_size)``
            coefficient array.
        rhs: The right-hand-side vector, one entry per row.
    """

    coefficients: dict
    rhs: Array

    def __init__(self, coefficients: Any, rhs: Any = 0.0):
        """Validate and normalize the rows.

        Args:
            coefficients: Mapping of block name to coefficients — a scalar or
                1-D array for a single row, or a ``(n_rows, block_size)``
                array for several rows. All named blocks must describe the
                same number of rows. Widths are checked against the block
                sizes by
                [constrained][diff_soup.inference.statistics.parameters.constrained].
            rhs: Scalar (broadcast to every row) or length-``n_rows`` vector.

        Raises:
            TypeError: If ``coefficients`` is not a mapping with non-empty
                string keys.
            ValueError: If the mapping is empty, a coefficient array has more
                than two dimensions or is empty, values are non-finite, row
                counts disagree between blocks, or ``rhs`` has the wrong
                length.
        """
        try:
            items = dict(coefficients)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"LinearConstraint 'coefficients' must be a mapping of block name "
                f"to coefficient array; got {type(coefficients).__name__}."
            ) from exc
        if not items:
            raise ValueError(
                "LinearConstraint 'coefficients' must name at least one block; an "
                "empty mapping declares no row."
            )
        normalized: dict = {}
        n_rows: Optional[int] = None
        for name, value in items.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    f"LinearConstraint coefficient keys must be non-empty block "
                    f"names; got {name!r}."
                )
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim != 2:
                raise ValueError(
                    f"LinearConstraint coefficients for block {name!r} must be a "
                    f"scalar, a 1-D array (one row), or a (n_rows, block_size) "
                    f"array; got an array with ndim={arr.ndim}."
                )
            if arr.size == 0:
                raise ValueError(
                    f"LinearConstraint coefficients for block {name!r} are empty; "
                    f"give one coefficient per scalar of the block for each row."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"LinearConstraint coefficients for block {name!r} must be "
                    f"finite; got {value!r}."
                )
            if n_rows is None:
                n_rows = int(arr.shape[0])
            elif arr.shape[0] != n_rows:
                raise ValueError(
                    f"LinearConstraint blocks must all describe the same number of "
                    f"rows; block {name!r} has {arr.shape[0]} row(s) where an "
                    f"earlier block declared {n_rows}."
                )
            normalized[name] = arr
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if rhs_arr.ndim == 0:
            rhs_arr = np.full((n_rows,), float(rhs_arr))
        elif rhs_arr.shape != (n_rows,):
            raise ValueError(
                f"LinearConstraint 'rhs' must be a scalar (broadcast to every row) "
                f"or a vector with one entry per row ({n_rows}); got shape "
                f"{rhs_arr.shape}."
            )
        if not np.all(np.isfinite(rhs_arr)):
            raise ValueError(f"LinearConstraint 'rhs' must be finite; got {rhs!r}.")
        self.coefficients = {k: jnp.asarray(v) for k, v in normalized.items()}
        self.rhs = jnp.asarray(rhs_arr)

    @property
    def n_rows(self) -> int:
        """The number of ``A x = b`` rows this constraint declares."""
        return int(self.rhs.shape[0])


class Gauge(eqx.Module):
    """A linear equality that fixes a direction the data leave free, for `gauges=` of [TrustRegion][diff_soup.inference.TrustRegion].

    Build one with `Gauge.pin`, `Gauge.zero_mean`, `Gauge.zero_mean_across` or `Gauge.declared`.
    """

    kind: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    value: Optional[float] = eqx.field(static=True)
    weights: Optional[Array]
    owners: Optional[dict]

    def __init__(
        self,
        kind: str,
        name: str,
        *,
        value: Any = None,
        weights: Any = None,
        owners: Any = None,
    ):
        """Validate one gauge declaration; prefer the factory methods.

        Raises:
            TypeError: If ``name`` is not a non-empty string.
            ValueError: If ``kind`` is unknown, a ``pin`` value is missing or
                non-finite, ``zero_mean`` weights are malformed, non-finite,
                or all zero, or an argument does not belong to the declared
                kind.
        """
        kinds = ("pin", "zero_mean", "zero_mean_across", "declared")
        if kind not in kinds:
            raise ValueError(
                f"Gauge 'kind' must be one of {kinds}; got {kind!r}. Build gauges "
                f"with Gauge.pin(...), Gauge.zero_mean(...), "
                "Gauge.zero_mean_across(...), or Gauge.declared(...)."
            )
        if not isinstance(name, str) or not name:
            raise TypeError(f"Gauge 'name' must be a non-empty string; got {name!r}.")
        if kind == "pin":
            if weights is not None or owners is not None:
                raise ValueError(
                    "Gauge.pin takes no weights; give the pinned scalar value as "
                    "Gauge.pin(name, value)."
                )
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Gauge.pin value must be a finite scalar; got {value!r}."
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"Gauge.pin value must be finite; got {value!r}.")
        elif kind == "zero_mean":
            if value is not None or owners is not None:
                raise ValueError(
                    "Gauge.zero_mean takes no pinned value; only an optional weight vector."
                )
            if weights is not None:
                w = np.asarray(weights, dtype=np.float64)
                if w.ndim != 1 or w.size == 0:
                    raise ValueError(
                        f"Gauge.zero_mean weights must be a non-empty 1-D vector "
                        f"(one weight per scalar of the block); got shape {w.shape}."
                    )
                if not np.all(np.isfinite(w)):
                    raise ValueError(f"Gauge.zero_mean weights must be finite; got {weights!r}.")
                if not np.any(w != 0.0):
                    raise ValueError(
                        "Gauge.zero_mean weights must include at least one nonzero "
                        "entry; an all-zero row constrains nothing."
                    )
                weights = jnp.asarray(w)
        elif kind == "zero_mean_across":
            if value is not None:
                raise ValueError("Gauge.zero_mean_across takes no pinned value.")
            try:
                owner_items = dict(owners)
            except (TypeError, ValueError):
                try:
                    owner_items = {owner_name: [1.0] for owner_name in owners}
                except TypeError as exc:
                    raise TypeError(
                        "Gauge.zero_mean_across owners must be an ordered mapping "
                        "from joint owner-block name to a within-block functional, "
                        "or a sequence of scalar block names."
                    ) from exc
            if len(owner_items) < 2:
                raise ValueError(
                    "Gauge.zero_mean_across needs at least two owner blocks; use "
                    "Gauge.zero_mean for one block."
                )
            normalized_owners = {}
            for owner_name, functional in owner_items.items():
                if not isinstance(owner_name, str) or not owner_name:
                    raise TypeError(
                        "Gauge.zero_mean_across owner-block names must be non-empty "
                        f"strings; got {owner_name!r}."
                    )
                functional_array = np.asarray(functional, dtype=np.float64)
                if functional_array.ndim == 0:
                    functional_array = functional_array.reshape((1,))
                if functional_array.ndim != 1 or not functional_array.size:
                    raise ValueError(
                        "Gauge.zero_mean_across functionals must be non-empty 1-D "
                        f"vectors; block {owner_name!r} has shape {functional_array.shape}."
                    )
                if not np.all(np.isfinite(functional_array)) or not np.any(functional_array != 0.0):
                    raise ValueError(
                        "Gauge.zero_mean_across functionals must be finite and not "
                        f"all zero; got {functional!r} for {owner_name!r}."
                    )
                normalized_owners[owner_name] = jnp.asarray(functional_array)
            if weights is None:
                weights = jnp.ones((len(normalized_owners),))
            else:
                outer = np.asarray(weights, dtype=np.float64)
                if outer.shape != (len(normalized_owners),):
                    raise ValueError(
                        "Gauge.zero_mean_across weights must have one entry per "
                        f"owner block ({len(normalized_owners)}); got shape {outer.shape}."
                    )
                if not np.all(np.isfinite(outer)) or np.any(outer < 0.0) or not np.any(outer):
                    raise ValueError(
                        "Gauge.zero_mean_across weights must be finite, nonnegative, "
                        "and include at least one positive entry."
                    )
                weights = jnp.asarray(outer)
            owners = normalized_owners
        else:  # declared
            if value is not None or weights is not None or owners is not None:
                raise ValueError("Gauge.declared is metadata only and takes no value or weights.")
        self.kind = kind
        self.name = name
        self.value = value if kind == "pin" else None
        self.weights = weights if kind in ("zero_mean", "zero_mean_across") else None
        self.owners = owners if kind == "zero_mean_across" else None

    @classmethod
    def pin(cls, name: str, value: Any) -> "Gauge":
        """Hold one coordinate of the fitted vector at a value.

        Args:
            name: The name of the coordinate, matching exactly one of the fit's parameter names.
            value: The finite value it is held at.

        Returns:
            The gauge.
        """
        return cls("pin", name, value=value)

    @classmethod
    def zero_mean(cls, name: str, weights: Any = None) -> "Gauge":
        """Hold the weighted mean of one parameter group at zero.

        Args:
            name: The name of the parameter group.
            weights: One weight per scalar of the group, or a basis object with `mean_weights()`, or `None` for equal weights.

        Returns:
            The gauge.
        """
        if hasattr(weights, "mean_weights") and not hasattr(weights, "shape"):
            weights = weights.mean_weights()
        return cls("zero_mean", name, weights=weights)

    @classmethod
    def zero_mean_across(cls, owners: Any, weights: Any = None) -> "Gauge":
        """Hold at zero the weighted mean, across the members of a one-level study, of one number per member.

        Each member's number is a weighted sum of one of its parameter groups, and at least two
        members are needed.

        Args:
            owners: A mapping from each member's group name to the weights that sum the group to one
                number, or a sequence of scalar group names.
            weights: One nonnegative weight per entry of `owners`, multiplied by the solver's cluster
                weights, or `None` for equal weights.

        Returns:
            The gauge.
        """
        return cls(
            "zero_mean_across",
            "cross-owner mean",
            weights=weights,
            owners=owners,
        )

    @classmethod
    def declared(cls, name: str) -> "Gauge":
        """Record a known free direction without constraining it.

        Args:
            name: A label for the free direction.

        Returns:
            The gauge, which constrains nothing.
        """
        return cls("declared", name)


def _null_space_elimination(A: np.ndarray, b: np.ndarray, scalar_names: tuple) -> tuple:
    """Eliminate ``A x = b`` exactly, host-side, by SVD per coupled component.

    Returns ``(x_particular, null_basis, reduced_names)``: the minimum-norm
    particular solution of the system, an orthonormal basis of ``null(A)``
    (``Nᵀ N = I``), and names for the reduced coordinates. Columns no row
    touches keep their exact unit direction and their scalar name, in their
    original order; the remaining null-space directions are appended and
    named ``nullspace[i]``. Rows are grouped into components that share
    columns and factored independently, so coordinates in unrelated
    components stay exactly untouched (no roundoff leaks between independent
    rows — in particular a plain pin reproduces its value bitwise).

    Redundant (linearly dependent) rows are absorbed by the SVD rank
    tolerance; rows admitting no common solution raise ``ValueError``.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    nonzero = A != 0.0
    for i in range(m):
        if not nonzero[i].any() and b[i] != 0.0:
            raise ValueError(
                f"The linear constraints are inconsistent: one row has no nonzero "
                f"coefficient but right-hand side {b[i]:g}, so it reads "
                f"0 = {b[i]:g}. Give the row a nonzero coefficient or a zero "
                f"right-hand side."
            )

    # Union-find over columns: rows sharing a column belong to one component.
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(m):
        cols = np.flatnonzero(nonzero[i])
        for c in cols[1:]:
            root_a, root_b = _find(int(cols[0])), _find(int(c))
            if root_a != root_b:
                parent[root_b] = root_a

    touched = nonzero.any(axis=0)
    grouped: dict = {}
    for j in range(n):
        if touched[j]:
            grouped.setdefault(_find(j), []).append(j)
    components = sorted(grouped.values(), key=lambda cols: cols[0])
    free_cols = [j for j in range(n) if not touched[j]]

    eps = np.finfo(np.float64).eps
    x_particular = np.zeros(n)
    null_cols: list = []
    for cols in components:
        rows_idx = np.flatnonzero(nonzero[:, cols].any(axis=1))
        sub = A[np.ix_(rows_idx, cols)]
        b_sub = b[rows_idx]
        u_svd, s, vt = np.linalg.svd(sub, full_matrices=True)
        tol = s[0] * max(sub.shape) * eps
        rank = int(np.sum(s > tol))
        xp_sub = vt[:rank].T @ ((u_svd[:, :rank].T @ b_sub) / s[:rank])
        residual = sub @ xp_sub - b_sub
        res_max = float(np.abs(residual).max())
        scale = max(1.0, float(np.abs(b_sub).max()), float(s[0]))
        if res_max > 1e-10 * scale:
            involved = [scalar_names[j] for j in cols]
            raise ValueError(
                f"The linear constraint rows coupling {involved} are inconsistent: "
                f"no x satisfies all of them (least-squares residual "
                f"{res_max:.3e}). Remove or reconcile the conflicting rows, for "
                f"example the same scalar pinned to two different values."
            )
        x_particular[cols] = xp_sub
        for direction in vt[rank:]:
            col = np.zeros(n)
            col[cols] = direction
            null_cols.append(col)

    n_reduced = len(free_cols) + len(null_cols)
    null_basis = np.zeros((n, n_reduced))
    for pos, j in enumerate(free_cols):
        null_basis[j, pos] = 1.0
    for pos, col in enumerate(null_cols):
        null_basis[:, len(free_cols) + pos] = col
    reduced_names = tuple(scalar_names[j] for j in free_cols) + tuple(
        f"nullspace[{i}]" for i in range(len(null_cols))
    )
    return x_particular, null_basis, reduced_names


class ConstrainedParameterization(Parameterization):
    """A [Parameterization][diff_soup.inference.statistics.parameters.Parameterization] with hard linear equalities eliminated.

    Built by [constrained][diff_soup.inference.statistics.parameters.constrained]. Writing ``z`` for the full inference
    coordinate of the underlying blocks, the reduced coordinate ``u`` enters
    through ``z = x_particular + null_basis @ u`` and the physical ``x`` is
    the usual blockwise transform of ``z``. ``x_particular`` is the
    minimum-norm particular solution of the declared rows and ``null_basis``
    an orthonormal basis (``Nᵀ N = I``) of their null space, so every row
    ``A x = b`` holds exactly for every ``u`` — no penalty terms and no ridge
    regularization are involved. Because every constrained block is
    identity-transformed, rows that hold in ``z`` hold identically in ``x``.

    The object keeps the full [Parameterization][diff_soup.inference.statistics.parameters.Parameterization] surface over the
    reduced coordinate: [transform][diff_soup.inference.statistics.parameters.Parameterization.transform] maps ``u`` to ``x``,
    [inverse][diff_soup.inference.statistics.parameters.Parameterization.inverse] maps ``x`` back (orthogonally projecting onto the
    constraint subspace if ``x`` violates a row), [`n_parameters`][diff_soup.inference.statistics.parameters.Parameterization.n_parameters] is the
    reduced dimension, [names][diff_soup.inference.statistics.parameters.Parameterization.names] labels the reduced coordinates
    (coordinates no row touches keep their scalar names, in their original
    order, followed by one ``"nullspace[i]"`` label per mixed null-space
    direction), and [convert_covariance][diff_soup.inference.statistics.parameters.Parameterization.convert_covariance] converts between the reduced
    ``"z"`` coordinate and the full physical ``"x"`` coordinate.

    ``x_particular`` and ``null_basis`` are stored as ``jax.numpy`` arrays,
    so all coordinate maps stay safe under ``jax.jit``, ``jax.vmap``, and
    autodiff.

    Attributes:
        blocks: The underlying full parameter blocks (inherited).
        x_particular: The minimum-norm particular solution of the rows, in
            the full inference coordinate.
        null_basis: The ``(n_full, n_reduced)`` orthonormal null-space basis.
        constraints: The [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint] declarations, as given.
        gauges: Every [Gauge][diff_soup.inference.statistics.parameters.Gauge] declaration, including ``declared``
            metadata.
    """

    x_particular: Array
    null_basis: Array
    constraints: tuple
    gauges: tuple
    _reduced_names: tuple = eqx.field(static=True)

    def __init__(
        self,
        blocks: Any,
        *,
        x_particular: Any,
        null_basis: Any,
        reduced_names: Any,
        constraints: Any = (),
        gauges: Any = (),
    ):
        """Assemble from prepared pieces; use [constrained][diff_soup.inference.statistics.parameters.constrained] to build one.

        Raises:
            ValueError: If the array shapes do not match the blocks, or the
                reduced names do not match the null-basis width (in addition
                to everything [Parameterization][diff_soup.inference.statistics.parameters.Parameterization] validates).
        """
        super().__init__(blocks)
        x_particular = jnp.asarray(x_particular)
        null_basis = jnp.asarray(null_basis)
        n_full = self.blocks[-1].index.stop
        if x_particular.shape != (n_full,) or null_basis.ndim != 2 or null_basis.shape[0] != n_full:
            raise ValueError(
                f"ConstrainedParameterization needs x_particular of shape "
                f"{(n_full,)} and a (n_full, n_reduced) null_basis with "
                f"n_full = {n_full}; got x_particular {x_particular.shape} and "
                f"null_basis {null_basis.shape}."
            )
        reduced_names = tuple(str(s) for s in reduced_names)
        if len(reduced_names) != null_basis.shape[1]:
            raise ValueError(
                f"ConstrainedParameterization needs one reduced-coordinate name "
                f"per null_basis column ({null_basis.shape[1]}); got "
                f"{len(reduced_names)} names."
            )
        self.x_particular = x_particular
        self.null_basis = null_basis
        self.constraints = tuple(constraints)
        self.gauges = tuple(gauges)
        self._reduced_names = reduced_names

    # -- structure ------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """The reduced dimension: full size minus the independent rows."""
        return int(self.null_basis.shape[1])

    @property
    def n_full_parameters(self) -> int:
        """The length of the flat physical vector ``x``."""
        return self.blocks[-1].index.stop

    @property
    def declared_gauges(self) -> tuple:
        """Directions asserted as gauge via [Gauge.declared][diff_soup.inference.statistics.parameters.Gauge.declared] (metadata)."""
        return tuple(g.name for g in self.gauges if g.kind == "declared")

    def names(self) -> tuple:
        """Per-scalar names of the reduced coordinate, in order."""
        return self._reduced_names

    def __repr__(self) -> str:
        """Display the underlying blocks plus the elimination summary."""
        inner = ", ".join(f"{b.name}={_transform_label(b.transform)}" for b in self.blocks)
        return (
            f"ConstrainedParameterization({inner}; "
            f"{self.n_full_parameters - self.n_parameters} eliminated, "
            f"{self.n_parameters} reduced)"
        )

    # -- the coordinate map ----------------------------------------------

    def transform(self, u: Array) -> Array:
        """Map a reduced vector ``u`` to the flat physical vector ``x``."""
        u = jnp.asarray(u)
        z = self.x_particular + self.null_basis @ u
        return super().transform(z)

    def inverse(self, x: Array) -> Array:
        """Map a flat physical vector ``x`` back to the reduced vector ``u``.

        If ``x`` violates a constraint row, the offending component is
        removed by orthogonal projection onto the constraint subspace; for
        feasible ``x`` the round trip with [transform][diff_soup.inference.statistics.parameters.Parameterization.transform] is exact.
        """
        z = super().inverse(jnp.asarray(x))
        return self.null_basis.T @ (z - self.x_particular)

    def log_abs_det_jacobian(self, u: Array) -> Array:
        """Return the scalar volume correction of the map ``u -> x`` at ``u``.

        The map embeds the reduced coordinate into the constraint subspace, so
        its Jacobian ``J = dx/du`` is rectangular; the change-of-variables
        term for densities on that subspace is the Gram (generalized)
        determinant ``0.5 * log det(Jᵀ J)``. For all-identity blocks this is
        exactly zero (the embedding is an isometry); free transformed blocks
        contribute their usual elementwise ``log |dx/dz|``.
        """
        u = jnp.asarray(u)
        z = self.x_particular + self.null_basis @ u
        elementwise = jnp.concatenate(
            [block.transform.log_abs_det_jacobian(z[block.index]) for block in self.blocks]
        )
        scaled = self.null_basis * jnp.exp(elementwise)[:, None]
        _, logdet = jnp.linalg.slogdet(scaled.T @ scaled)
        return 0.5 * logdet

    # -- labeled covariance conversion ------------------------------------

    def convert_covariance(
        self,
        cov: Array,
        *,
        at: Array,
        source: str,
        target: str,
    ) -> ConvertedCovariance:
        """Convert a covariance between the reduced and physical coordinates.

        Identical to [Parameterization.convert_covariance][diff_soup.inference.statistics.parameters.Parameterization.convert_covariance] except that
        here ``"z"`` labels the *reduced* coordinate ``u`` (dimension
        [`n_parameters`][diff_soup.inference.statistics.parameters.Parameterization.n_parameters]) while ``"x"`` keeps the full physical dimension
        ([`n_full_parameters`][diff_soup.inference.statistics.parameters.ConstrainedParameterization.n_full_parameters]): a reduced covariance pushed to ``x`` is
        the ``(n_full, n_full)`` covariance supported on the constraint
        subspace, and an ``x`` covariance pulled back to ``"z"`` is its
        restriction to that subspace.

        Raises:
            ValueError: If a coordinate label is unknown or the shapes do not
                match the ``source`` coordinate's dimension.
        """
        _check_coordinate(source, what="convert_covariance 'source'")
        _check_coordinate(target, what="convert_covariance 'target'")
        cov = jnp.asarray(cov)
        at = jnp.asarray(at)
        dims = {"z": self.n_parameters, "x": self.n_full_parameters}
        n_source = dims[source]
        if cov.shape != (n_source, n_source) or at.shape != (n_source,):
            raise ValueError(
                f"convert_covariance expects cov of shape {(n_source, n_source)} "
                f"and at of shape {(n_source,)} for source={source!r} of this "
                f"constrained parameterization ({self.n_parameters} reduced / "
                f"{self.n_full_parameters} physical parameters); got cov "
                f"{cov.shape} and at {at.shape}."
            )
        if source == target:
            converted = cov
        elif source == "z":
            jac = jax.jacfwd(self.transform)(at)
            converted = jac @ cov @ jac.T
        else:
            jac = jax.jacfwd(self.inverse)(at)
            converted = jac @ cov @ jac.T
        return ConvertedCovariance(covariance=converted, at=at, source=source, target=target)


def constrained(
    parameterization: Parameterization,
    constraints: Any = (),
    gauges: Any = (),
) -> Parameterization:
    """Eliminate hard linear equalities from a parameterization, exactly.

    Lowers every [Gauge][diff_soup.inference.statistics.parameters.Gauge] (``pin``/``zero_mean``) and every
    [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint] into rows of one system ``A x = b`` on the flat
    physical coordinate, computes the minimum-norm particular solution and an
    orthonormal null-space basis host-side (via SVD), and returns a
    [ConstrainedParameterization][diff_soup.inference.statistics.parameters.ConstrainedParameterization] over the reduced coordinate ``u``:
    the equalities hold to numerical precision for every ``u``, and nothing is
    penalized or ridge-regularized. Redundant (linearly dependent) rows are
    detected by the SVD rank tolerance and simply do not reduce the dimension
    further; rows that admit no common solution raise ``ValueError``.

    Every block addressed by a row must use the [Identity][diff_soup.inference.statistics.parameters.Identity] transform —
    a linear relation in ``x`` is nonlinear in the inference coordinate of a
    transformed block. Unconstrained blocks may use any transform and keep
    their coordinates unchanged.

    Args:
        parameterization: The [Parameterization][diff_soup.inference.statistics.parameters.Parameterization] to constrain. It must
            not already be constrained — declare every row in a single call.
        constraints: An iterable of [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint] (a single
            instance is also accepted).
        gauges: An iterable of [Gauge][diff_soup.inference.statistics.parameters.Gauge] (a single instance is also
            accepted). ``Gauge.declared`` entries are recorded on the result
            but eliminate nothing.

    Returns:
        A [ConstrainedParameterization][diff_soup.inference.statistics.parameters.ConstrainedParameterization] (still a
        [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]) whose ``n_parameters`` is the full size
        minus the number of independent rows.

    Raises:
        TypeError: If the inputs are not a [Parameterization][diff_soup.inference.statistics.parameters.Parameterization],
            [LinearConstraint][diff_soup.inference.statistics.parameters.LinearConstraint] instances, and [Gauge][diff_soup.inference.statistics.parameters.Gauge] instances.
        ValueError: If a name addresses no block, an addressed block is not
            identity-transformed, a coefficient or weight width does not
            match its block size, a pin does not resolve to a single scalar,
            or the rows are inconsistent.
    """
    if isinstance(parameterization, ConstrainedParameterization):
        raise ValueError(
            "constrained() cannot re-constrain a ConstrainedParameterization: its "
            "reduced coordinates no longer address the physical x directly. "
            "Declare every constraint and gauge in a single constrained(...) call "
            "on the underlying parameterization."
        )
    if not isinstance(parameterization, Parameterization):
        raise TypeError(
            f"constrained() expects a Parameterization; got {type(parameterization).__name__}."
        )
    if isinstance(constraints, LinearConstraint):
        constraints = (constraints,)
    if isinstance(gauges, Gauge):
        gauges = (gauges,)
    constraints = tuple(constraints)
    gauges = tuple(gauges)
    for con in constraints:
        if not isinstance(con, LinearConstraint):
            raise TypeError(
                f"constrained() 'constraints' entries must be LinearConstraint "
                f"instances; got {type(con).__name__}."
            )
    for gauge in gauges:
        if not isinstance(gauge, Gauge):
            raise TypeError(
                f"constrained() 'gauges' entries must be Gauge instances (built "
                f"with Gauge.pin/zero_mean/declared); got {type(gauge).__name__}."
            )

    block_by_name = {b.name: b for b in parameterization.blocks}
    n_full = parameterization.n_parameters
    scalar_names = parameterization.names()
    scalar_index = {nm: i for i, nm in enumerate(scalar_names)}

    def _require_identity(block: ParamBlock, addressed_by: str) -> None:
        if not isinstance(block.transform, Identity):
            raise ValueError(
                f"Linear constraints and gauges are eliminated exactly in the "
                f"physical coordinate x, so every addressed block must use the "
                f"Identity transform, because a linear relation in x is nonlinear in the "
                f"inference coordinate of a transformed block. {addressed_by} "
                f"addresses block {block.name!r}, which uses "
                f"{_transform_label(block.transform)}; give that block the "
                f"Identity transform or leave it unconstrained."
            )

    rows: list = []
    rhs_values: list = []

    for ci, con in enumerate(constraints):
        for name, coeff in con.coefficients.items():
            block = block_by_name.get(name)
            if block is None:
                raise ValueError(
                    f"constraints[{ci}] addresses unknown block {name!r}; this "
                    f"parameterization has blocks {sorted(block_by_name)}."
                )
            _require_identity(block, f"constraints[{ci}]")
            width = np.asarray(coeff).shape[1]
            if width != block.size:
                raise ValueError(
                    f"constraints[{ci}] coefficients for block {name!r} have "
                    f"width {width} but the block has {block.size} scalar(s); "
                    f"provide one coefficient per scalar of the block."
                )
        rhs_np = np.asarray(con.rhs, dtype=np.float64)
        for i in range(con.n_rows):
            row = np.zeros(n_full)
            for name, coeff in con.coefficients.items():
                row[block_by_name[name].index] = np.asarray(coeff, dtype=np.float64)[i]
            rows.append(row)
            rhs_values.append(float(rhs_np[i]))

    for gauge in gauges:
        if gauge.kind == "declared":
            continue
        row = np.zeros(n_full)
        if gauge.kind == "pin":
            idx = scalar_index.get(gauge.name)
            if idx is None:
                block = block_by_name.get(gauge.name)
                if block is not None:
                    raise ValueError(
                        f"Gauge.pin fixes a single scalar, but block "
                        f"{gauge.name!r} has {block.size} scalars; pin one of "
                        f"them by its indexed scalar name (e.g. "
                        f"'{gauge.name}[0]') or use a LinearConstraint row."
                    )
                raise ValueError(
                    f"Gauge.pin addresses unknown scalar {gauge.name!r}; known "
                    f"scalars are {list(scalar_names)}."
                )
            owner = next(b for b in parameterization.blocks if b.index.start <= idx < b.index.stop)
            _require_identity(owner, f"Gauge.pin({gauge.name!r})")
            row[idx] = 1.0
            rows.append(row)
            rhs_values.append(float(gauge.value))
        elif gauge.kind == "zero_mean":
            block = block_by_name.get(gauge.name)
            if block is None:
                raise ValueError(
                    f"Gauge.zero_mean addresses unknown block {gauge.name!r}; "
                    f"this parameterization has blocks {sorted(block_by_name)}."
                )
            _require_identity(block, f"Gauge.zero_mean({gauge.name!r})")
            if gauge.weights is None:
                weights = np.full(block.size, 1.0 / block.size)
            else:
                weights = np.asarray(gauge.weights, dtype=np.float64)
                if weights.shape != (block.size,):
                    raise ValueError(
                        f"Gauge.zero_mean weights for block {gauge.name!r} must "
                        f"have one entry per scalar of the block "
                        f"({block.size}); got shape {weights.shape}."
                    )
            row[block.index] = weights
            rows.append(row)
            rhs_values.append(0.0)
        else:  # zero_mean_across
            outer = np.asarray(gauge.weights, dtype=np.float64)
            outer = outer / np.sum(outer)
            for owner_weight, (name, functional) in zip(outer, gauge.owners.items()):
                block = block_by_name.get(name)
                if block is None:
                    raise ValueError(
                        f"Gauge.zero_mean_across addresses unknown block {name!r}; "
                        f"this parameterization has blocks {sorted(block_by_name)}."
                    )
                _require_identity(block, f"Gauge.zero_mean_across({name!r})")
                functional = np.asarray(functional, dtype=np.float64)
                if functional.shape != (block.size,):
                    raise ValueError(
                        "Gauge.zero_mean_across functional for block "
                        f"{name!r} must have one entry per scalar ({block.size}); "
                        f"got shape {functional.shape}."
                    )
                row[block.index] = owner_weight * functional
            rows.append(row)
            rhs_values.append(0.0)

    A = np.vstack(rows) if rows else np.zeros((0, n_full))
    b = np.asarray(rhs_values, dtype=np.float64)
    x_particular, null_basis, reduced_names = _null_space_elimination(A, b, scalar_names)
    return ConstrainedParameterization(
        parameterization.blocks,
        x_particular=jnp.asarray(x_particular),
        null_basis=jnp.asarray(null_basis),
        reduced_names=reduced_names,
        constraints=constraints,
        gauges=gauges,
    )


def non_centered(
    parameterization: Parameterization,
    block: Any,
    *,
    scale: Any,
) -> Parameterization:
    """Return a copy with one block expressed in a standardized coordinate.

    For the selected block, the old inference coordinate ``u`` is replaced by
    ``u = scale * z`` before its existing transform is applied. Thus the new
    physical map is ``x = old_transform(scale * z)``: it spans exactly the
    same physical values, has the exact inverse, and contributes one
    ``log(scale)`` Jacobian term per scalar in the block. Other blocks and all
    public scalar names are unchanged.

    Args:
        parameterization: An unconstrained [Parameterization][diff_soup.inference.statistics.parameters.Parameterization].
        block: The selected block name, or the corresponding
            [ParamBlock][diff_soup.inference.statistics.parameters.ParamBlock] instance from ``parameterization.blocks``.
        scale: Positive finite scalar standard deviation/characteristic scale.

    Returns:
        A new [Parameterization][diff_soup.inference.statistics.parameters.Parameterization]; the input remains unchanged.

    Raises:
        TypeError: If the arguments do not identify a parameter block.
        ValueError: If the block is unknown, ``scale`` is not positive and
            finite, or the parameterization is already constraint-reduced.
            Constraint elimination changes the dimension and must therefore
            be applied after choosing centered/non-centered block coordinates.
    """
    if isinstance(parameterization, ConstrainedParameterization):
        raise ValueError(
            "non_centered() cannot rewrite a ConstrainedParameterization: its "
            "reduced coordinates no longer align one-to-one with ParamBlock "
            "slices. Apply non_centered() before constrained()."
        )
    if not isinstance(parameterization, Parameterization):
        raise TypeError(
            f"non_centered() expects a Parameterization; got {type(parameterization).__name__}."
        )

    if isinstance(block, ParamBlock):
        candidates = tuple(
            candidate
            for candidate in parameterization.blocks
            if candidate is block
            or (
                candidate.name == block.name
                and candidate.index == block.index
                and candidate.shape == block.shape
            )
        )
        block_name = block.name
    elif isinstance(block, str) and block:
        candidates = tuple(
            candidate for candidate in parameterization.blocks if candidate.name == block
        )
        block_name = block
    else:
        raise TypeError(
            "non_centered() 'block' must be a non-empty block name or a "
            f"ParamBlock instance; got {block!r}."
        )
    if len(candidates) != 1:
        raise ValueError(
            f"non_centered() addresses unknown block {block_name!r}; this "
            f"parameterization has blocks {[b.name for b in parameterization.blocks]}."
        )

    # ScaledTransform owns the scalar validation and supplies one
    # float for both the executable map and the block's characteristic-scale
    # metadata.
    selected = candidates[0]
    transform = ScaledTransform(selected.transform, scale)
    replacement = ParamBlock(
        selected.name,
        selected.index,
        transform,
        shape=selected.shape,
        path=selected.path,
        group=selected.group,
        scale=transform.scale,
        prior_coordinate=selected.prior_coordinate,
    )
    blocks = tuple(
        replacement if candidate is selected else candidate for candidate in parameterization.blocks
    )
    return Parameterization(blocks)
