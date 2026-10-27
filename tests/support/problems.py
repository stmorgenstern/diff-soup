"""Small modeling and simulation problems shared by the suite (public imports only)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.linear import SparseTopology

_CONSTANT = ds.Dependencies(time=False, state=False, parameters=False, control=False)


class DecayODE:
    """``y0' = -k y0``, ``y1' = -2 k y1`` from ``(1, 1)``: two decoupled decays."""

    n_u, n_lam = 2, 0

    def __init__(self, precision=None):
        self.problem = ds.MassProblem(
            rhs=lambda t, y, theta, control: jnp.stack((-theta * y[0], -2.0 * theta * y[1])),
            mass=ds.MassOperator.identity(2),
            structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),
            precision=precision,
        )

    @staticmethod
    def initial_state(dtype=jnp.float64):
        """The start ``(1, 1)``."""
        return jnp.ones((2,), dtype=dtype)

    @staticmethod
    def analytical(t, theta):
        """The exact state at ``t``."""
        return jnp.stack((jnp.exp(-theta * t), jnp.exp(-2.0 * theta * t)))


class ConstrainedDAE:
    """Index-one DAE: ``p' = -k p``, ``0 = p - q - lam``, ``0 = p^2 - lam`` (``lam`` global).

    The exact solution is ``p = e^{-kt}``, ``lam = p^2``, ``q = p - p^2``; the start
    ``(1, 0, 1)`` is consistent.
    """

    n_u, n_lam = 2, 1

    def __init__(self, precision=None):
        self.problem = ds.MassProblem(
            rhs=lambda t, y, theta, control: jnp.stack(
                (-theta * y[0], y[0] - y[1] - y[2], y[0] * y[0] - y[2])
            ),
            mass=ds.MassOperator.sparse(
                [0],
                [0],
                lambda t, y, theta, control: jnp.ones((1,), y.dtype),
                n=3,
                dependencies=_CONSTANT,
            ),
            structure=ds.DAEStructure(
                differential_vars=np.asarray([True, False, False]),
                partitions=(("state", 0, 2), ("global", 2, 3)),
            ),
            precision=precision,
        )

    @staticmethod
    def initial_state(dtype=jnp.float64):
        """The consistent start ``(1, 0, 1)``."""
        return jnp.asarray([1.0, 0.0, 1.0], dtype=dtype)

    @staticmethod
    def analytical(t, theta):
        """The exact augmented state at ``t``."""
        p = jnp.exp(-theta * t)
        return jnp.stack((p, p - p * p, p * p))


class SparseChainDAE:
    """Sparse DAE on ``2n + 1`` unknowns with a declared pattern.

    ``u_i' = -k u_i``; ``0 = u_i^2 - v_i + a (v_{i-1} + v_{i+1}) + b lam``;
    ``0 = mean(v) - lam``. The start holds the exact ``u`` and zero ``v`` and
    ``lam``, off the algebraic manifold, so a solve must initialize consistently.
    """

    A, B = 0.25, 0.05

    def __init__(self, n=4, precision=None):
        self.n = n
        self.n_u, self.n_lam = 2 * n, 1
        size = 2 * n + 1
        i = np.arange(n)
        rows = [i, n + i, n + i[1:], n + i[:-1], n + i, n + i, np.full(n, 2 * n), [2 * n]]
        cols = [i, n + i, n + i[1:] - 1, n + i[:-1] + 1, i, np.full(n, 2 * n), n + i, [2 * n]]
        self.topology = SparseTopology(
            rows=np.concatenate(rows), cols=np.concatenate(cols), n_dofs=size
        )
        self.problem = ds.MassProblem(
            rhs=self._rhs,
            mass=ds.MassOperator.sparse(
                i,
                i,
                lambda t, y, theta, control: jnp.ones((n,), y.dtype),
                n=size,
                dependencies=_CONSTANT,
            ),
            structure=ds.DAEStructure(
                differential_vars=np.arange(size) < n,
                partitions=(("state", 0, 2 * n), ("global", 2 * n, size)),
            ),
            rhs_sparsity=self.topology,
            precision=precision,
        )

    def _rhs(self, t, y, theta, control):
        n = self.n
        u, v, lam = y[:n], y[n : 2 * n], y[2 * n]
        near = jnp.zeros_like(v).at[1:].add(v[:-1]).at[:-1].add(v[1:])
        f_v = u * u - v + self.A * near + self.B * lam
        return jnp.concatenate([-theta * u, f_v, (jnp.mean(v) - lam)[None]])

    def _u0(self, dtype):
        return jnp.linspace(0.5, 1.0, self.n, dtype=dtype)

    def initial_state(self, dtype=jnp.float64):
        """Exact ``u``, zero ``v`` and ``lam``: off the manifold."""
        return jnp.concatenate([self._u0(dtype), jnp.zeros((self.n + 1,), dtype)])

    def analytical(self, t, theta):
        """The exact augmented state: ``v = T^{-1}(u^2 + b lam)``."""
        u = self._u0(jnp.float64) * jnp.exp(-theta * t)
        chain = jnp.eye(self.n) - self.A * (jnp.eye(self.n, k=1) + jnp.eye(self.n, k=-1))
        w_sq = jnp.linalg.solve(chain, u * u)
        w_one = jnp.linalg.solve(chain, jnp.full((self.n,), self.B))
        lam = jnp.mean(w_sq) / (1.0 - jnp.mean(w_one))
        return jnp.concatenate([u, w_sq + lam * w_one, lam[None]])


def endpoint(problem, solver, theta, y0, t1=0.05, **keywords):
    """The augmented state at ``t1`` of one solve (``ys`` then ``lambdas``)."""
    solution = ds.pdesolve(
        problem,
        solver,
        (0.0, t1),
        y0,
        theta=theta,
        saveat=ds.SaveAt(t1=True),
        max_steps=keywords.pop("max_steps", 200),
        **keywords,
    )
    state = solution.ys[-1]
    if solution.lambdas is not None:
        state = jnp.concatenate((state, solution.lambdas[-1]))
    return state


def reaction_diffusion_recipe(name="reaction_diffusion", n=6, method="fvm"):
    """``u' = 0.05 u_xx - rate u`` on ``[0, 1]`` with a learnable ``rate`` (not built)."""
    spec = ds.problem(name).domain("line", ds.cartesian_grid_1d(n, 0.0, 1.0, method=method))
    spec.param("rate", value=0.4, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + 0.5 * jnp.sin(3.0 * x))

    @spec.equation(writes="u", reads=("u",))
    def reaction_diffusion(q):
        return 0.05 * q.ops.laplacian(q.u) - q.p.rate * q.u

    return spec


def operational_cell(*, modes=("current", "voltage", "power"), decay=None, **control):
    """``u' = -u + I`` on four cells, voltage ``mean(u) - 2 I``, one operational table."""
    spec = ds.problem("operational_cell").domain(
        "line", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm")
    )
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.full_like(x, 2.0))
    spec.global_var("I_cell", initial=0.0, mass="algebraic")
    if decay is not None:
        spec.param("decay", value=decay, learnable=True)
    spec.observe.scalar(name="terminal_current", fn=lambda q: q["I_cell"])
    spec.observe.scalar(name="terminal_voltage", fn=lambda q: jnp.mean(q.u) - 2.0 * q["I_cell"])

    @spec.equation(writes="u", reads=("u", "I_cell"))
    def u_balance(q):
        return -(1.0 if decay is None else q.p.decay) * q.u + q["I_cell"]

    spec.control.operational(
        name="terminal",
        current="terminal_current",
        voltage=control.pop("voltage", "terminal_voltage"),
        current_unknown="I_cell",
        modes=modes,
        **control,
    )
    return spec


def observed_recipe():
    """Two four-cell fields closed by one algebraic global, with two declared observations
    (``mean_u``, and ``load_gap``, which reads the global); returns the built problem, its
    assembled start and the learnable ``rate`` handle."""
    spec = ds.problem("observed_pair").domain(
        "line", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm")
    )
    rate = spec.param("rate", value=1.3, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + x)
    spec.field("c_e", mass="differential", scale=1.0, init=lambda x: 2.0 - x)
    spec.global_var("load", initial=0.0, scale=1.0)
    spec.equation(writes="u", reads=("u", "load"))(lambda q: -q.p.rate * q.u + 0.1 * q["load"])
    spec.equation(writes="c_e", reads=("c_e", "u"))(lambda q: -0.5 * q.c_e + q.u)
    spec.constraint.galvanostatic(
        name="load_closure",
        closes="load",
        current_fn=lambda q: q["load"] - jnp.mean(q.u),
        target_fn=lambda time: jnp.asarray(0.0),
    )
    spec.observe.scalar(name="mean_u", fn=lambda q: jnp.mean(q.u))
    spec.observe.scalar(name="load_gap", fn=lambda q: jnp.mean(q.c_e) - q["load"])
    return spec.build(), spec.assemble_state({}), rate
