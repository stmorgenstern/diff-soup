"""Fisher information of a measurement channel from the sensitivities of a solve.

A channel ``h(u, t, theta)`` maps a saved state frame ``u`` (shape
``(n_state,)``) to its ``n_channel`` measured values. With state
sensitivities ``S[f, :, j] = du_f / dtheta_j`` (``jax.jacfwd`` of a
[SensitivityRun][diff_soup.inference.design.solve.SensitivityRun], shape
``(n_frames, n_state, n_directions)``), the channel sensitivities are
``s[f, c, j] = dh_c(u_f) / du . S[f, :, j] + dh_c / dtheta_j`` and the
Fisher information of independent Gaussian noise with standard deviations
``sigma[f, c]`` is

``F[i, j] = sum_{f, c} s[f, c, i] s[f, c, j] / sigma[f, c]^2``,

that is ``F = A^T Sigma^-1 A`` with ``A`` the ``(n_frames * n_channel,
n_directions)`` matrix of the rows ``s[f, c, :]`` and ``Sigma`` the diagonal
of the variances. Independent channels add: sum their matrices.

Every function is one batched JAX computation without Python loops over
frames or directions: it composes with ``jax.vmap`` over a leading sample
axis (prior samples, candidate protocols) and is differentiable with respect
to the frames, the sensitivities and the channel's parameters, so
``jax.grad`` of a criterion of ``F`` reaches the design parameters of the
solve.

* [observable_trajectory][diff_soup.inference.design.fisher.observable_trajectory]:
  the channel on every frame, ``(n_frames, n_channel)``.
* [observable_sensitivities][diff_soup.inference.design.fisher.observable_sensitivities]:
  ``s``, ``(n_frames, n_channel, n_directions)``, by one ``jax.jvp`` per
  direction, including the channel's direct ``dh / dtheta`` term.
* [fisher_information][diff_soup.inference.design.fisher.fisher_information]:
  ``F``, ``(n_directions, n_directions)``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from diff_soup.inference.design.criteria import fisher_with_prior
from diff_soup.inference.fitting.observables import DiagonalNoise

__all__ = [
    "observable_trajectory",
    "observable_sensitivities",
    "fisher_information",
    "fisher_with_prior",
]


def observable_trajectory(
    observable: Any,
    ys: Array,
    *,
    ts: Optional[Array] = None,
    theta: Any = None,
) -> Array:
    """Evaluate a channel on every frame of a solve: ``h(ys[f], ts[f], theta)``.

    Args:
        observable: The measurement channel, a callable
            ``observable(u, t=..., theta=...)`` returning ``(n_channel,)``
            values for one frame ``u``.
        ys: Saved state frames, shape ``(n_frames, n_state)``.
        ts: Optional frame times, shape ``(n_frames,)``.
        theta: Optional parameter pytree passed to the channel.

    Returns:
        Channel values, shape ``(n_frames, n_channel)``.
    """
    if ts is None:
        return jax.vmap(lambda u: observable(u, t=None, theta=theta))(ys)
    return jax.vmap(lambda u, t: observable(u, t=t, theta=theta))(ys, jnp.asarray(ts))


def observable_sensitivities(
    observable: Any,
    ys: Array,
    S: Array,
    *,
    ts: Optional[Array] = None,
    theta: Any = None,
    theta_tangents: Any = None,
) -> Array:
    """Chain state sensitivities through a channel: ``dh/dθ`` per frame and direction.

    ``out[f, c, j] = dh_c(ys[f]) / du . S[f, :, j] + dh_c / dtheta .
    theta_tangents[j]``, differentiable with respect to ``ys``, ``S`` and
    ``theta``.

    For each sensitivity direction ``j``, propagates the state tangent
    ``S[:, :, j]`` (and, when given, the direct parameter tangent
    ``theta_tangents[j]``) through the channel with one ``jax.jvp`` of the
    vmapped frame map — no Python loops over frames or directions.

    Args:
        observable: The measurement channel, a callable
            ``observable(u, t=..., theta=...)``.
        ys: Saved state frames, shape ``(n_frames, n_state)``.
        S: State sensitivities, shape ``(n_frames, n_state, n_directions)``
            (``jax.jacfwd`` of a sensitivity run).
        ts: Optional frame times, shape ``(n_frames,)``.
        theta: Optional parameter pytree passed to the channel.
        theta_tangents: Optional pytree matching ``theta`` whose leaves carry a
            leading ``n_directions`` axis — the tangent of ``theta`` along each
            sensitivity direction. Supplies the channel's direct ``∂h/∂θ``
            term; omit it for channels that only read the state.

    Returns:
        Channel sensitivities, shape ``(n_frames, n_channel, n_directions)``.
    """
    ys = jnp.asarray(ys)
    S = jnp.asarray(S)
    has_t = ts is not None
    t_arr = jnp.zeros((ys.shape[0],), dtype=ys.dtype) if not has_t else jnp.asarray(ts)
    n_dirs = S.shape[-1]

    if theta is not None and theta_tangents is None:
        theta_tangents = jax.tree.map(
            lambda leaf: jnp.zeros((n_dirs,) + jnp.shape(leaf), dtype=ys.dtype), theta
        )

    if theta is None:

        def trajectory(u_stack, t_stack):
            if has_t:
                return jax.vmap(lambda u, t: observable(u, t=t, theta=None))(u_stack, t_stack)
            del t_stack
            return jax.vmap(lambda u: observable(u, t=None, theta=None))(u_stack)

        def per_direction(s_col):
            _, dot = jax.jvp(trajectory, (ys, t_arr), (s_col, jnp.zeros_like(t_arr)))
            return dot

        return jax.vmap(per_direction, in_axes=2, out_axes=2)(S)

    def trajectory_theta(u_stack, t_stack, th):
        if has_t:
            return jax.vmap(lambda u, t: observable(u, t=t, theta=th))(u_stack, t_stack)
        del t_stack
        return jax.vmap(lambda u: observable(u, t=None, theta=th))(u_stack)

    def per_direction(s_col, th_dot):
        _, dot = jax.jvp(
            trajectory_theta,
            (ys, t_arr, theta),
            (s_col, jnp.zeros_like(t_arr), th_dot),
        )
        return dot

    return jax.vmap(per_direction, in_axes=(2, 0), out_axes=2)(S, theta_tangents)


def fisher_information(sens: Array, noise: Any = None) -> Array:
    """Assemble the Fisher information matrix from channel sensitivities.

    ``F[i, j] = sum_{f, c} sens[f, c, i] sens[f, c, j] / sigma[f, c]^2``, one
    batched contraction: ``A^T Sigma^-1 A`` for the rows ``A[(f, c), :] =
    sens[f, c, :]`` and independent Gaussian noise of variances
    ``Sigma = diag(sigma^2)``. Linear in the precisions and quadratic in
    ``sens``, so differentiable with respect to both. Independent channels
    add: compute one matrix per channel and sum them.

    Args:
        sens: Channel sensitivities, shape ``(n_frames, n_channel, n_directions)``.
        noise: The channel's noise: the standard deviations ``sigma``, a
            positive scalar or an array that broadcasts to
            ``(n_frames, n_channel)``; a ``DiagonalNoise``; or ``None`` for
            ``sigma = 1``.

    Returns:
        The symmetric positive semidefinite ``(n_directions, n_directions)``
        Fisher information matrix.
    """
    sens = jnp.asarray(sens)
    if noise is None:
        weights = jnp.ones(sens.shape[:2], dtype=sens.dtype)
    elif isinstance(noise, DiagonalNoise):
        weights = noise.precisions(shape=sens.shape[:2])
    else:
        sigma = jnp.asarray(noise, dtype=sens.dtype)
        weights = jnp.broadcast_to(1.0 / sigma**2, sens.shape[:2])
    return jnp.einsum("fci,fc,fcj->ij", sens, weights, sens)
