"""Recipe handles: identity across same-named recipes and templates, parameter schemas,
structural copies, selection by group names, field slices, and rebuilt boundary signals."""

from __future__ import annotations

import gc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference.approx import PiecewiseLinear
from diff_soup.inference.fitting.inference_problem import InferenceProblem, ParameterGroups
from diff_soup.inference.fitting.series import Component, Series

_SOLVER = ds.JNDF.stock(linear="dense")


def _decay(name="decay_handles", rate=0.4, cells=3):
    spec = ds.problem(name).domain("line", ds.cartesian_grid_1d(cells, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    handle = spec.param("rate", value=rate, learnable=True)
    spec.equation(writes="u", reads=("u",))(lambda q: -q.p.rate * q.u)
    return spec, handle


def _inference(problem, y0, **keywords):
    times = jnp.linspace(0.0, 0.4, 5)
    data = Series("state", times, jnp.exp(-0.3 * times)[:, None] * y0[None, :])
    return InferenceProblem(
        problem, data, solver=_SOLVER, initial_state=inf.FixedInitialState(y0, time=0.0), **keywords
    )


@ds.recipe
def _template(radius=1.0, protocol="decay"):
    spec = ds.problem("template_decay")
    spec.domain("rod", ds.cartesian_grid_1d(3, 0.0, radius, method="fvm"))
    spec.field("u", mass="differential", scale=1.0)
    spec.param("rate", value=0.2, learnable=True)
    spec.param("drive", value=0.0)
    spec.equation(writes="u", reads=("u",))(lambda q: -q.p.rate * q.u + q.p.drive)
    return spec


def _experiment(model, **keywords):
    measurement = inf.Measurement(
        "mean",
        times=jnp.array([0.0, 1.0]),
        values=jnp.array([1.0, 0.8]),
        observe=lambda state: jnp.mean(state),
        likelihood=inf.Gaussian(scale=0.1),
    )
    return inf.Experiment(
        model,
        solver=_SOLVER,
        t_span=(0.0, 1.0),
        initial_state=jnp.ones(3),
        measurements=(measurement,),
        **keywords,
    )


def test_a_template_caches_builds_hands_out_handles_and_binds_experiment_inputs():
    """``@ds.recipe`` caches one build per distinct argument value read, hands out one handle per
    declared name before any build (refusing an undeclared one), and ``inputs=`` binds a
    template argument and a fixed parameter into an experiment's model and id."""
    rate = _template.handle("rate")
    first = _template.build()
    assert first is _template.build(radius=1.0, protocol="unused")
    assert _template.build(radius=2.0) is not first
    assert rate is _template.handle("rate") and rate.identity == ("_template", "rate")
    with pytest.raises(ValueError, match="does not declare.*missing"):
        _template.handle("missing")

    @ds.recipe
    def required(radius):
        return _template.author(radius)

    late = required.handle("rate")
    ds.default_theta(required.build(radius=3.0), {late: 0.8})
    assert late.learnable
    for experiment, expected in (
        (_experiment(_template, inputs={"radius": 2.0, "drive": 0.5}), 0.3),
        (_experiment(_template.build(radius=2.0), inputs={"drive": 0.7}), 0.5),
    ):
        rhs, _ = experiment.model.physics_residual(jnp.ones(3), jnp.zeros(0), jnp.asarray(0.0))
        # measured 5.6e-17 (one rounding of 0.5 - 0.2), CPU VM 2026-10-22
        np.testing.assert_allclose(rhs, np.full(3, expected), atol=1e-14)
        assert "drive=" in experiment.id
    with pytest.raises(ValueError, match="unknown.*template arguments.*declared parameters"):
        _experiment(_template, inputs={"missing": 1.0})


def test_a_schema_holds_one_entry_per_declaration_and_slices_need_no_private_coordinates():
    """Repeated builds of one recipe leave one schema entry naming the latest handle, a pytree
    copy keeps it, and a handle from an unrelated recipe is refused; ``field_slice`` names a
    field's flat rows, a Component reads them, and the inference problem names its theta."""
    for _ in range(5):
        _decay("repeated", cells=4)[0].build()
    gc.collect()
    spec, rate = _decay("repeated", cells=4)
    problem = spec.build()
    schema = _inference(problem, spec.assemble_state({})).parameter_schema
    assert len(schema.entries) == 1 and schema.handles[0].identity == rate.identity
    copied = jax.tree_util.tree_map(lambda leaf: leaf, problem)
    assert copied is not problem
    assert (
        _inference(copied, spec.assemble_state({})).parameter_schema.handles[0].identity
        == rate.identity
    )
    other_spec, other = _decay("unrelated", cells=5)
    other_spec.build()
    with pytest.raises(ValueError, match="comes from a Recipe builder"):
        _inference(
            problem, spec.assemble_state({}), parameters=ParameterGroups(local={"rate": other})
        )
    spec, _ = _decay("sliced")
    problem = spec.build()
    assert problem.field_slice("u") == slice(0, 3)
    np.testing.assert_array_equal(
        Component(problem.field_slice("u")).predict(jnp.arange(5.0), 0.1, "x"), [0.0, 1.0, 2.0]
    )
    assert _inference(problem, spec.assemble_state({})).theta_names == ("rate",)


def _reaction():
    spec = ds.problem("param_and_function").domain(
        "line", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.param("rate", value=0.3, learnable=True)
    reaction = spec.function(
        "reaction", eqx.nn.MLP(1, "scalar", 1, 0, key=jax.random.PRNGKey(1)), learnable=True
    )
    spec.equation(writes="u", reads=("u",))(
        lambda q: -q.p.rate * q.u + jax.vmap(lambda v: q.fn.reaction(jnp.atleast_1d(v)))(q.u)
    )
    return spec.build(), reaction, spec.assemble_state({})


def test_selection_by_group_names_resolves_declarations_first_and_drops_partial_handles():
    """A group name selects a declaration, then a theta leaf path; a selection covering part
    of a declared value leaves that handle out of the schema; an unknown name is refused."""
    problem, reaction, y0 = _reaction()

    def build(groups=None):
        return _inference(problem, y0, **({} if groups is None else {"parameters": groups}))

    assert reaction in build().parameter_schema.handles
    partial = build(ParameterGroups(local={"g": ("weight",)}))
    assert partial.theta_names == ("weight",) and partial.parameter_schema is None
    assert build(ParameterGroups(local={"k": ("rate",)})).theta_names == ("rate",)
    mixed = build(ParameterGroups(local={"g": ("rate", "weight")}))
    assert (mixed.theta_names, mixed.param_group_cols) == (("rate", "weight"), ((0, 1),))
    with pytest.raises(ValueError, match="neither a declared learnable"):
        build(ParameterGroups(local={"g": ("nowhere",)}))


def _signal_recipe(name, value, flux=False):
    """A boundary pinned to a declared constant piecewise-linear signal of ``value``."""
    signal = PiecewiseLinear([0.0, 0.5, 1.0], [value] * 3)
    spec = ds.problem(name).domain("line", ds.cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 0.0 * x)
    spec.function("inlet", signal, learnable=True)
    spec.equation(writes="u", reads=("u",))(lambda q: 0.0 * q.u + 0.0 * q.fn.inlet(q.t))
    spec.bc.dirichlet("u", node_indices=[0], value_fn=signal)
    if flux:
        spec.bc.flux("u", node_indices=[2], value=0.0)
    return spec


def _pinned(problem, spec):
    return float(ds.pdesolve(problem, _SOLVER, (0.0, 1.0), spec.assemble_state({})).ys[-1][0])


def test_same_named_or_rewritten_recipes_keep_their_own_handles_and_signals():
    """Same-named recipes keep their own handles (the other's is refused) and a second build pins
    the boundary to its own signal; an equation set rewritten with with_(replace=) keeps
    the signal of the condition that stays."""
    _signal_recipe("rebuilt_signal", 1.0).build()
    spec = _signal_recipe("rebuilt_signal", 2.0)
    rewritten_spec = _signal_recipe("rewritten_signal", 2.0, flux=True)
    donor = _signal_recipe("donor_signal", 2.0, flux=True).to_equation_set().components[0].bcs[1]
    rewritten = ds.lower(rewritten_spec.to_equation_set().with_(replace=((type(donor), donor),)))
    # measured 0.0 from 2.0 on both, CPU VM 2026-10-22
    np.testing.assert_allclose(
        [_pinned(spec.build(), spec), _pinned(rewritten, rewritten_spec)], 2.0, atol=1e-9
    )
    first_spec, first = _decay(rate=0.15)
    second_spec, second = _decay(rate=0.85)
    first_problem = first_spec.build()
    assert first.identity != second.identity
    leaves = jax.tree_util.tree_leaves(ds.default_theta(first_problem, {first: 0.9}))
    assert any(bool(jnp.allclose(leaf, 0.9)) for leaf in leaves)
    second_spec.build()
    with pytest.raises(ValueError, match="declared on this problem"):
        ds.default_theta(first_problem, {second: 0.5})
