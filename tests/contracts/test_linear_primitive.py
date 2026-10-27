"""Linear primitives: the order word every operation takes and returns, its tangent and its
batching, the int32 operands, rematerialisation, the lowered native calls and the pattern
matvec, on the dense backend and on KLU (a backend whose factors live in native slots)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_soup.linear import (
    KLU,
    Dense,
    FactorContractError,
    SolveContext,
    SparseTopology,
    after,
    analyze,
    factor,
    invalidate,
    refactor,
    solve,
)
from diff_soup.linear._primitive import ORDER_DTYPE, pattern_matvec, phased_factor_p
from tests.support.programs import equations

N = 5
ROWS = np.array([0, 1, 2, 3, 4, 0, 1, 1, 2, 3, 4, 2, 2, 3], dtype=np.int32)
COLS = np.array([0, 1, 2, 3, 4, 1, 0, 2, 3, 4, 0, 0, 2, 1], dtype=np.int32)
_VALUES = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, ROWS.size) + 5.0 * (ROWS == COLS))
_B = jnp.linspace(-1.0, 1.0, N)
_OPERATIONS = ("phased_factor", "phased_solve", "phased_refactor", "phased_invalidate")
# measured at most 2.4e-16 relative against NumPy and the plain gradient, CPU VM 2026-10-22,
# pytest tests/contracts/test_linear_primitive.py
EXACT = {"rtol": 1e-12, "atol": 1e-13}


def _analysis(backend=None, bc=None):
    topology = SparseTopology(rows=ROWS, cols=COLS, n_dofs=N, bc_indices=bc)
    return analyze(topology, backend=Dense() if backend is None else backend)


def _matrix(values):
    a = np.zeros((N, N))
    np.add.at(a, (ROWS, COLS), np.asarray(values))
    return a


def _named(fn, *args):
    return [e for e in equations(jax.make_jaxpr(fn)(*args)) if e.primitive.name in _OPERATIONS]


def test_the_order_word_chains_every_operation_and_no_operation_carries_an_effect():
    """In factor, solve, refactor, solve, invalidate each operation takes the order word the
    one before returned, every word is an int32 scalar, and the program has no JAX effect."""
    analysis = _analysis()

    def program(v, w):
        x1, f = solve(factor(analysis, v), _B)
        x2, f = solve(refactor(f, w, True), _B)
        return x1, x2, invalidate(f)

    jaxpr = jax.make_jaxpr(program)(_VALUES, 1.1 * _VALUES)
    chain = _named(program, _VALUES, 1.1 * _VALUES)
    names = "phased_factor phased_solve phased_refactor phased_solve phased_invalidate"
    assert [e.primitive.name for e in chain] == names.split()
    for before, eqn in zip(chain, chain[1:]):
        assert before.outvars[-1] in eqn.invars
    assert all(
        e.outvars[-1].aval.shape == () and np.dtype(e.outvars[-1].aval.dtype) == ORDER_DTYPE
        for e in chain
    )
    assert not jaxpr.effects and not any(e.effects for e in equations(jaxpr))
    x1, x2, _ = jax.jit(program)(_VALUES, 1.1 * _VALUES)
    np.testing.assert_allclose(
        [x1, x2],
        [np.linalg.solve(_matrix(_VALUES), _B), np.linalg.solve(_matrix(1.1 * _VALUES), _B)],
        **EXACT,
    )


@pytest.mark.requires_klu
def test_the_order_word_has_a_float0_tangent_and_batches_like_the_handle():
    """The order word's tangent is float0; under vmap over values a native batch (dense)
    carries one word per lane, while a sequential backend (KLU) maps its lanes with an
    unbatched word; a vmapped factor holds one int32 word per lane."""
    analysis = _analysis()
    assert (
        jax.jvp(lambda v: factor(analysis, v).order, (_VALUES,), (_VALUES,))[1].dtype
        == jax.dtypes.float0
    )
    lanes = jnp.stack([_VALUES, 1.1 * _VALUES, 1.2 * _VALUES])
    assert jax.vmap(lambda v: factor(analysis, v))(lanes).order.shape == (3,)
    (native,) = [
        e
        for e in _named(jax.vmap(lambda v: factor(analysis, v).order), lanes)
        if e.primitive.name == "phased_factor"
    ]
    assert native.invars[-1].aval.shape == (3,)
    with SolveContext() as ctx:
        klu = _analysis(KLU())
        program = jax.vmap(lambda v: factor(klu, v, ctx=ctx).order)
        (sequential,) = [e for e in _named(program, lanes) if e.primitive.name == "phased_factor"]
        assert sequential.invars[-1].aval.shape == ()
        assert "scan" in [e.primitive.name for e in equations(jax.make_jaxpr(program)(lanes))]


def _bind(order=None, token=None, context=None, values=_VALUES):
    return phased_factor_p.bind(
        values,
        jnp.zeros((), jnp.int32) if token is None else token,
        jnp.zeros((), jnp.int32) if context is None else context,
        jnp.zeros((), ORDER_DTYPE) if order is None else order,
        analysis=_analysis(),
        orientation="A",
        precision="fp64",
        tol=None,
    )


def test_the_order_word_slot_token_and_context_id_are_int32_scalars_without_a_vmap_axis():
    """A factorization refuses an order word, a slot token or a context id that is not an
    int32 scalar, at trace time, and refuses a vmap over context ids: a factor call belongs
    to one context."""
    for keywords, match in (
        ({"order": jnp.zeros((), jnp.float32)}, "order word as int32"),
        ({"order": jnp.zeros((2,), ORDER_DTYPE)}, "order word as int32"),
        ({"token": jnp.zeros((), jnp.int64)}, "slot token as an int32 scalar"),
        ({"token": jnp.zeros((2,), jnp.int32)}, "slot token as an int32 scalar"),
        ({"context": jnp.asarray(3.0)}, "int32 scalar"),
    ):
        with pytest.raises(FactorContractError, match=match):
            _bind(**keywords)
    with pytest.raises(FactorContractError, match="belongs to one SolveContext"):
        jax.vmap(lambda c: _bind(context=c)[-1])(jnp.asarray([3, 4], jnp.int32))


def _klu_calls(text):
    return [line for line in text.splitlines() if "custom_call @diffsoup_klu_" in line]


@pytest.mark.requires_klu
def test_a_checkpoint_saves_the_factorization_and_a_solve_on_native_storage():
    """Under jax.checkpoint the factorization is saved, never recomputed; a solve on native
    storage is saved too (two solves in the gradient program) while a solve on array factors
    is recomputed (three); both gradients equal the plain one."""
    counts = {}
    with SolveContext() as ctx:
        for name, backend in (("dense", Dense()), ("klu", KLU())):
            analysis = _analysis(backend)

            def loss(v, analysis=analysis):
                return jnp.sum(jnp.sin(solve(factor(analysis, v, ctx=ctx), _B)[0]))

            program = jax.grad(jax.checkpoint(loss))
            names = [e.primitive.name for e in _named(program, _VALUES)]
            counts[name] = (names.count("phased_factor"), names.count("phased_solve"))
            np.testing.assert_allclose(jax.jit(program)(_VALUES), jax.grad(loss)(_VALUES), **EXACT)
    assert counts == {"dense": (1, 3), "klu": (1, 2)}


@pytest.mark.requires_klu
def test_native_calls_take_the_order_word_last_and_after_keeps_what_they_need():
    """Each native call takes the int32 order word last and returns the next one last, is kept
    although its results are tokens, and an ahead-of-time program reruns exactly; after(f,
    g) keeps f's leaves on g's word, so g's factorization survives, and refuses
    anything but a factor."""
    with SolveContext() as ctx:
        analysis = _analysis(KLU())

        def program(v, w):
            x1, f = solve(factor(analysis, v, ctx=ctx), _B)
            return x1, solve(refactor(f, w, True), _B)[0]

        lowered = jax.jit(program).lower(_VALUES, 1.1 * _VALUES)
        calls = _klu_calls(lowered.as_text())
        assert len(calls) >= 4
        for line in calls:
            assert "tensor<i32>) -> (" in line and line.rstrip().endswith("tensor<i32>)")
            assert "has_side_effect = true" in line
        compiled = lowered.compile()
        for scale in (1.0, 2.0, 0.5):
            x1, x2 = compiled(scale * _VALUES, 1.1 * _VALUES)
            np.testing.assert_allclose(x1, np.linalg.solve(_matrix(scale * _VALUES), _B), **EXACT)
            np.testing.assert_allclose(x2, np.linalg.solve(_matrix(1.1 * _VALUES), _B), **EXACT)
        f, g = factor(analysis, _VALUES, ctx=ctx), factor(analysis, 1.1 * _VALUES, ctx=ctx)
        h = after(f, g)
        assert h.values is f.values and h.handle is f.handle and after(f) is f

        def linked(v, w):
            return after(factor(analysis, v, ctx=ctx), factor(analysis, w, ctx=ctx)).order

        def unlinked(v, w):
            return factor(analysis, v, ctx=ctx).order

        for program, count in ((linked, 2), (unlinked, 1)):
            text = jax.jit(program).lower(_VALUES, 1.1 * _VALUES).as_text()
            assert sum("diffsoup_klu_factor" in line for line in _klu_calls(text)) == count
        with pytest.raises((FactorContractError, TypeError), match="Factor"):
            after(f, 3)


def test_the_pattern_matvec_applies_the_entries_outside_dirichlet_rows():
    """``pattern_matvec`` applies the pattern's operator, Dirichlet rows dropped, or its
    transpose, to a vector, a block and a batch of lanes, and refuses values of another size."""
    block = jnp.stack([_B, 2.0 * _B + 1.0])
    for bc in (None, [1, 3]):
        analysis, d = _analysis(bc=bc), _matrix(_VALUES)
        if bc is not None:
            d[bc, :] = 0.0
        for transpose, m in ((False, d), (True, d.T)):
            for rhs in (_B, block):
                got = pattern_matvec(analysis, _VALUES, rhs, transpose=transpose)
                np.testing.assert_allclose(got, np.asarray(rhs) @ m.T, **EXACT)
        lanes = pattern_matvec(
            analysis, jnp.stack([_VALUES, 2.0 * _VALUES]), jnp.stack([_B, _B]), transpose=False
        )
        np.testing.assert_allclose(lanes[1], 2.0 * d @ np.asarray(_B), **EXACT)
    with pytest.raises(FactorContractError, match="one per pattern entry"):
        pattern_matvec(_analysis(), _VALUES[:-1], _B, transpose=False)
