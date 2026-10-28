"""_forward_seam.py — the injected solver seam for the fit layer.

Every inference problem drives a PDE forward through ``pdesolve`` and
differentiates it (LM uses ``jacfwd``).  This module implements the seam
that bridges problems to the injected solver:

* **Normalize the solver problem ONCE, outside the trace.** Structural
  lowering and derivative-plan construction must not be re-run inside the
  residual closure; the identity-stable solver view is obtained once at
  problem-construction time.
* **Ravel only the learnable theta leaves.**  The learnable filter (names /
  a grouped selection / ``None`` for the problem's declared learnables) is
  applied to the problem's ``default_theta()`` pytree; only the surviving
  leaves are flattened into the flat ``x`` vector, one coordinate block per
  distinct array object. Frozen leaves ride along via ``eqx.combine``.
* **Thread the reconstructed theta EXPLICITLY into ``pdesolve``.**  The JNDF
  IFT ``custom_vjp`` differentiates only w.r.t. the explicit ``theta`` argument,
  so the traced leaf must reach the corrector as ``theta=``, not as a
  closed-over problem leaf.
* **``SaveAt`` is built ONCE outside the closure.**  A ``SaveAt(ts=...)``
  array captured inside the jitted residual becomes a tracer and triggers
  an error during config building.
* **Loud-status is a HOST check.**  The condition ``sol.ts[-1] ≈ t_end``
  cannot be asserted inside the jitted residual (tracer→float).  The
  problem checks it on a host-side baseline solve, not per residual call.

Adjoints are resolved when a differentiation seam is first requested. Forward
seams select ``ForwardAdjoint`` when needed; reverse seams preserve a declared
reverse-capable strategy or select checkpointed replay. The adjoint is an
equinox static field, so both swaps use ``dataclasses.replace`` rather than
``eqx.tree_at``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Optional

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from diff_soup._logging import log as logger
from diff_soup.inference.fitting._lane_map import folds_enclosing_batches, lane_map

# Residual value stamped on any failed or non-finite PDE solve. Large enough
# that the optimizer steps away from the failing region without a NaN poisoning
# the Gauss-Newton system; host re-evaluation re-detects it so ``fit`` can
# surface the failure.
_FAILED_SENTINEL = 1e15


def _solution_failed(solution: Any) -> jax.Array:
    """JIT-safe guard for every unsuccessful or non-finite trajectory.

    ``Solution.ok`` recognizes all successful terminal codes (including a
    declared event stop), unlike checks for one particular failure such as
    ``MAX_STEPS``. A finite-value guard additionally protects optimization
    from a backend defect that returns an invalid state with a successful
    status.
    """
    failed = jnp.logical_not(jnp.asarray(solution.ok, dtype=bool))
    for values in (solution.ys, solution.lambdas):
        if values is not None:
            failed = failed | jnp.logical_not(jnp.all(jnp.isfinite(jnp.asarray(values))))
    return failed


# ---------------------------------------------------------------------------
# Solver-problem normalization
# ---------------------------------------------------------------------------


def coerce_forward_problem(problem: Any) -> Any:
    """Return the identity-stable solver view for ``problem``.

    Args:
        problem: A solver problem, solver view, or authored source
            exposing ``solver_problem()``.

    Returns:
        The solver view used by every forward solve.

    Raises:
        TypeError: If ``problem`` supplies no solver problem.
    """
    from diff_soup.problem.implicit.solver_problem import get_or_build_solver_view

    return get_or_build_solver_view(problem)


# ---------------------------------------------------------------------------
# θ partition (the decision-variable selection)
# ---------------------------------------------------------------------------


def _is_name_spec(theta: Any) -> bool:
    """Whether ``theta`` is a name spec (a non-string iterable of strings)."""
    if isinstance(theta, str):
        return True
    try:
        items = list(theta)
    except TypeError:
        return False
    return bool(items) and all(isinstance(i, str) for i in items)


def _path_key_tokens(path: Any) -> tuple:
    """Extract the human-readable key tokens from a JAX tree key path.

    A leaf's key path is a tuple of ``GetAttrKey`` / ``DictKey`` / ``SequenceKey``
    entries; this pulls the attribute names and dict keys (the user-meaningful
    selectors a ``theta=("log_kappa",)`` name matches against).
    """
    tokens: list[str] = []
    for entry in path:
        name = getattr(entry, "name", None)
        if name is not None:
            tokens.append(str(name))
            continue
        key = getattr(entry, "key", None)
        if isinstance(key, str):
            tokens.append(key)
    return tuple(tokens)


def names_to_theta_mask(theta0: Any, names: Any) -> tuple[Any, tuple]:
    """Build a True/False mask over ``theta0`` selecting the named θ leaves.

    The mask is keyed by the leaf's *key path* through the partitioned ``theta0``
    pytree: a requested name matches a leaf iff the name appears as an attribute
    name or dict key anywhere along that leaf's path (e.g. ``"log_kappa"`` matches
    ``…inner.log_kappa``). A Recipe parameter or function keeps its declared name
    on that path, so this resolves authored and declared learnable names
    uniformly without reaching into problem internals (the ``default_theta()``
    pytree is the public seam).

    Args:
        theta0: The solver problem's ``default_theta()`` pytree.
        names: A name (str) or iterable of names.

    Returns:
        ``(mask, matched_names)`` — a bool pytree of ``theta0``'s structure and the
        ordered tuple of θ-leaf names (one per inexact-array leaf, for diagnostics).

    Raises:
        ValueError: If any requested name matches no inexact-array leaf.
    """
    requested = (names,) if isinstance(names, str) else tuple(names)
    req_set = set(requested)
    matched: set[str] = set()
    leaf_names: list[str] = []

    leaves_with_path = jax.tree_util.tree_leaves_with_path(theta0)

    def _leaf_label(path: Any) -> str:
        tokens = _path_key_tokens(path)
        return tokens[-1] if tokens else ""

    # Build the mask leaf-by-leaf so it has the exact structure of theta0.
    def _mask_leaf(path: Any, leaf: Any) -> bool:
        if not eqx.is_inexact_array(leaf):
            return False
        leaf_names.append(_leaf_label(path))
        tokens = set(_path_key_tokens(path))
        hit = bool(tokens & req_set)
        if hit:
            matched.update(tokens & req_set)
        return hit

    mask = jax.tree_util.tree_map_with_path(_mask_leaf, theta0)
    del leaves_with_path

    missing = sorted(req_set - matched)
    if missing:
        available = sorted({n for n in leaf_names if n})
        raise ValueError(
            f"optsolve: theta name(s) {missing} did not match any learnable leaf. "
            f"Available learnable leaf names: {available}. Pass names that appear "
            "on the problem's array fields (or declared spec learnables)."
        )
    return mask, tuple(leaf_names)


# ---------------------------------------------------------------------------
# Per-seam adjoint resolution
# ---------------------------------------------------------------------------


def _declared_compositions(adjoint: Any, lowered_problem: Any) -> frozenset[str]:
    """Return one adjoint's declared optimizer differentiation modes."""
    declare = getattr(adjoint, "compositions", None)
    if declare is None:
        return frozenset()
    return frozenset(declare(lowered_problem))


def resolve_forward_adjoint(forward: Any, lowered_problem: Any = None) -> Any:
    """Resolve the injected solver to a forward-mode-capable adjoint.

    A residual-Jacobian solver (LM) drives ``jacfwd`` through ``pdesolve``,
    but a reverse adjoint does not support forward-mode differentiation. This
    function is called lazily when the forward seam is first requested and
    swaps in ``ForwardAdjoint`` when needed. Forward values are numerically
    unchanged by the swap.

    **Mechanism:** the swap uses ``dataclasses.replace``, not ``eqx.tree_at``
    — the config's ``adjoint`` is an equinox static field (lives in the
    treedef, not as a pytree leaf), so ``eqx.tree_at(lambda c: c.adjoint, …)``
    raises ``AttributeError``.

    Args:
        forward: The injected time-stepping solver.
        lowered_problem: The solve-ready problem used by the adjoint's static
            capability declaration.

    Returns:
        ``forward`` unchanged if its adjoint is already forward-capable; otherwise a
        ``dataclasses.replace`` copy with ``adjoint=ForwardAdjoint()``.

    Raises:
        ConfigurationError: If ``forward`` carries no swappable ``adjoint`` slot and
            its current adjoint is not forward-capable (no forward-capable adjoint
            exists).
    """
    from diff_soup._adjoints import ForwardAdjoint

    adjoint = getattr(forward, "adjoint", None)
    if "forward" in _declared_compositions(adjoint, lowered_problem):
        return forward
    if not dataclasses.is_dataclass(forward) or not hasattr(forward, "adjoint"):
        from diff_soup._errors import ConfigurationError

        raise ConfigurationError(
            "optsolve: the residual-Jacobian solver (Levenberg-Marquardt) drives "
            "forward-mode AD (jacfwd) through pdesolve, but the solver "
            f"{type(forward).__name__} exposes no forward-capable adjoint and no "
            "swappable 'adjoint' slot. Pass a solver whose adjoint is "
            "ForwardAdjoint()."
        )
    swapped = dataclasses.replace(forward, adjoint=ForwardAdjoint())
    logger.info(
        "optsolve: solver adjoint %r is not forward-mode capable; swapped "
        "to ForwardAdjoint for the jacfwd residual-Jacobian path "
        "(forward values unchanged by the swap).",
        type(adjoint).__name__ if adjoint is not None else None,
    )
    return swapped


def resolve_reverse_adjoint(forward: Any, lowered_problem: Any = None) -> Any:
    """Resolve the injected solver to a reverse-mode-capable adjoint.

    A user-supplied checkpoint, backsolve, implicit, or backend-native reverse
    strategy is retained. Otherwise native solvers and Diffrax receive
    ``CheckpointAdjoint``. A solver that declares its supported adjoint types,
    receives its declared reverse member instead.

    Args:
        forward: The injected time-stepping solver.
        lowered_problem: The solve-ready problem used by capability declarations.

    Returns:
        The original solver when its adjoint supports reverse mode, or a
        ``dataclasses.replace`` copy carrying a reverse adjoint.

    Raises:
        ConfigurationError: If the solver has no replaceable ``adjoint`` field
            or declares no reverse-capable adjoint type.
    """
    from diff_soup._adjoints import CheckpointAdjoint
    from diff_soup._errors import ConfigurationError

    adjoint = getattr(forward, "adjoint", None)
    if "reverse" in _declared_compositions(adjoint, lowered_problem):
        return forward
    if not dataclasses.is_dataclass(forward) or not hasattr(forward, "adjoint"):
        raise ConfigurationError(
            "optsolve reverse-mode differentiation requires a solver with a "
            "replaceable reverse-capable adjoint; "
            f"{type(forward).__name__} has no replaceable 'adjoint' field."
        )

    reverse = CheckpointAdjoint()
    supported = tuple(getattr(forward, "supports_adjoints", ()))
    if supported and not any(isinstance(reverse, cls) for cls in supported):
        reverse_types = [
            cls for cls in supported if "reverse" in _declared_compositions(cls(), lowered_problem)
        ]
        if not reverse_types:
            names = ", ".join(sorted(cls.__name__ for cls in supported))
            raise ConfigurationError(
                f"optsolve reverse-mode differentiation cannot use "
                f"{type(forward).__name__}: its supported adjoints ({names}) contain "
                "no reverse-capable strategy."
            )
        reverse = reverse_types[0]()

    swapped = dataclasses.replace(forward, adjoint=reverse)
    logger.info(
        "optsolve: solver adjoint %r is not reverse-mode capable; swapped to "
        "%s for the reverse scalar-loss path (forward values unchanged by the swap).",
        type(adjoint).__name__ if adjoint is not None else None,
        type(reverse).__name__,
    )
    return swapped


# ---------------------------------------------------------------------------
# The forward seam: normalize the problem and ravel theta, then solve frames
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ForwardSeam:
    """Frozen forward-seam handle: solver view + theta-ravel + flat x0.

    Built ONCE at problem construction (eager, outside any trace).  Carries the
    machinery a residual closure needs to map a flat ``x`` to per-frame states
    through ``pdesolve``.

    A plain frozen ``dataclass`` (NOT an ``eqx.Module``): it is a host-side build
    product, not a pytree that flows through JAX transforms — the residual *closures*
    the problems build (which capture this seam) are what get traced.  Storing the
    problem (which references JAX arrays) on an ``eqx.Module`` static field would trip
    equinox's "JAX array set as static" warning (promoted to error by the suite).

    Attributes:
        problem: The cached solver view.
        unflatten: ``flat -> learnable-theta pytree`` (inverse ravel).
        theta_frozen: The non-learnable theta leaves (recombined each call).
        x0: Initial flat learnable-theta vector (numpy, the θ-dtype stream).
        theta_names: Ordered names of the inexact-array θ leaves (for diagnostics).
    """

    problem: Any
    unflatten: Callable
    theta_frozen: Any
    x0: np.ndarray
    theta_names: tuple = ()
    theta_token_sets: tuple = ()
    theta_leaf_ids: tuple = ()

    def theta_of(self, x_flat: jax.Array) -> Any:
        """Reconstruct the full theta pytree from a flat learnable vector."""
        return eqx.combine(self.unflatten(x_flat), self.theta_frozen)


@dataclasses.dataclass(frozen=True)
class GroupedThetaSelection:
    """A θ selection assembled from named parameter groups.

    Two kinds of source select learnable leaves:

    * ``name_sources`` — tuples of learnable names matched against each θ-leaf's key
      path (the same rule a bare ``theta=("k",)`` name spec uses).
    * ``mask_sources`` — boolean pytrees over the *problem* (from
      ``learnables(problem, names)``).  These select by leaf identity, so a declared
      learnable whose public name is not a θ-leaf path token (e.g. a Dirichlet value
      whose leaf field is ``value``) is still resolved correctly.
    * ``leaf_id_sources`` — leaf-identity sets obtained directly from stable Recipe
      parameter/function handles. These do not inspect authored problem IR.

    [build_forward_seam][] unions every source into one θ-leaf filter.

    ``order`` lists the groups in the order their coordinates take in the flat
    decision vector, one ``(leaf_ids, tokens)`` pair per group. A selected leaf
    goes with the first group that owns it, by identity or by a path token;
    leaves within one group, and leaves no group owns, keep their θ-leaf order.
    """

    name_sources: tuple = ()
    mask_sources: tuple = ()
    leaf_id_sources: tuple = ()
    order: tuple = ()


def selected_leaf_ids(pytree: Any, mask: Any) -> frozenset:
    """Return the ``id()`` set of ``pytree`` leaves selected ``True`` by ``mask``.

    ``mask`` is a boolean pytree of ``pytree``'s structure (e.g. from
    [parameters_at][diff_soup.inference.fitting._path_filter.parameters_at]). The solver
    view's ``default_theta()`` shares the array objects partitioned from the
    source problem, so a problem-relative mask resolves to θ-leaf identities.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    mask_leaves = jax.tree_util.tree_leaves(mask)
    return frozenset(id(leaf) for leaf, keep in zip(leaves, mask_leaves) if bool(keep))


def parameter_handle_leaf_ids(handle: Any, *, problem: Any = None) -> frozenset:
    """Return the current ``theta`` leaf identities owned by a Recipe handle.

    When ``problem`` is supplied, resolve through its Recipe-handle registry.
    This keeps a handle usable when a structurally identical Recipe refreshes
    the shared lowering cache with new array objects.

    Raises:
        ValueError: When ``problem`` carries registered Recipe declarations but
            none from the builder that declared ``handle``.
    """
    if problem is not None:
        from diff_soup.inference.fitting.learnables import learnables
        from diff_soup.problem._registry import handle_declarations

        handle_declarations(problem, handle)
        selector = learnables(problem, (getattr(handle, "name", ""),))
        if selector is None:
            return frozenset()
        return selected_leaf_ids(problem, selector.mask)

    kind = getattr(handle, "kind", None)
    if kind in ("scalar", "array"):
        value = handle.value
    elif kind == "function":
        value = handle.fn
    else:
        raise TypeError(
            "parameter groups accept Recipe handles returned by spec.param(...) "
            f"or spec.function(...); got {type(handle).__name__}."
        )
    return frozenset(
        id(leaf) for leaf in jax.tree_util.tree_leaves(value) if eqx.is_inexact_array(leaf)
    )


def build_forward_seam(problem: Any, theta: Any) -> ForwardSeam:
    """Build the forward seam: normalize problem, partition theta, ravel learnables.

    Args:
        problem: A solver problem/view or a source with ``solver_problem()``.
        theta: The learnable spec — ``None`` selects the problem's declared
            learnables, a name tuple resolves via [names_to_theta_mask][], and a
            [GroupedThetaSelection][diff_soup.inference.fitting._forward_seam.GroupedThetaSelection]
            resolves via [_grouped_leaf_filter][].

    Returns:
        A frozen [ForwardSeam][diff_soup.inference.fitting._forward_seam.ForwardSeam].

    Raises:
        ValueError: If the problem declares no learnables and no selection was
            given, or if a spec selects no learnable leaves at all.
        TypeError: If ``theta`` is none of the above.
    """
    solver_problem = coerce_forward_problem(problem)
    source, theta0 = _own_default_theta(problem, solver_problem)
    if isinstance(theta, GroupedThetaSelection):
        leaf_filter = _grouped_leaf_filter(problem, theta0, theta)
    elif _is_name_spec(theta):
        leaf_filter, _all_names = names_to_theta_mask(theta0, theta)
    elif theta is None:
        leaf_filter = _declared_leaf_filter(source, theta0)
    else:
        raise TypeError(
            "optsolve: the theta selection must be None, a tuple of learnable "
            f"names, or a GroupedThetaSelection (built by parameters="
            f"ParameterGroups(...)); got {type(theta).__name__}."
        )
    learn, frozen = eqx.partition(theta0, leaf_filter)
    # One coordinate block per distinct selected array object, in ravel order:
    # the same declared value reached from several closures is one unknown.
    selected = _deduplicated_selection(learn)
    if isinstance(theta, GroupedThetaSelection) and theta.order:
        selected = _in_declaration_order(selected, theta.order)
    if not selected:
        raise ValueError(
            "optsolve: the theta spec selected no learnable leaves; the decision "
            "variable vector is empty. Select the problem's declared learnables, "
            "or pass a leaf predicate or learnable names that resolve to at least "
            "one array leaf."
        )
    # Names / path tokens / identities of the selected leaves, matching the
    # ravel order — used for the identifiability warning message and to map
    # parameter groups onto flat-vector columns.
    names = _selected_theta_names(selected)
    token_sets = _selected_theta_token_sets(selected)
    leaf_ids = _selected_theta_leaf_ids(selected)
    flat, unflatten = _ravel_deduplicated(learn, selected)
    return ForwardSeam(
        problem=solver_problem,
        unflatten=unflatten,
        theta_frozen=frozen,
        x0=np.asarray(flat),
        theta_names=names,
        theta_token_sets=token_sets,
        theta_leaf_ids=leaf_ids,
    )


def _own_default_theta(problem: Any, solver_problem: Any) -> tuple:
    """Return ``(source, theta0)``: the problem whose leaves ``theta0`` holds, and ``theta0``.

    A problem equal in content to one presented earlier is served that one's
    solver view, whose ``default_theta()`` holds the other problem's array
    objects. Selections match leaves by identity on ``problem``, so ``theta0``
    comes from ``problem``'s own ``default_theta()`` when it has one.
    """
    own = getattr(problem, "default_theta", None)
    if problem is not solver_problem and callable(own):
        return problem, own()
    source = getattr(solver_problem, "source", None)
    return (source if source is not None else problem), solver_problem.default_theta()


def _declared_leaf_filter(source: Any, theta0: Any) -> Any:
    """Return the θ mask selecting the declared learnables of ``source``, the owner of ``theta0``'s leaves.

    Membership follows the declaration, not the leaf's type: a parameter is
    fitted because it was declared ``learnable=True`` (or because the model
    names it in ``learnable_paths()``), never because it happens to hold an
    array.  Every occurrence of one declared value is selected, so a parameter
    several closures read stays one unknown.
    """
    from diff_soup.inference.fitting.learnables import declared_learnable_leaf_ids

    ids = declared_learnable_leaf_ids(source)
    if not ids:
        raise ValueError(
            "optsolve: the problem declares no learnable parameters, so there is no "
            "default answer to what is being fitted. Declare the unknowns with "
            "spec.param(..., learnable=True) / spec.function(..., learnable=True), "
            "or state the selection explicitly with "
            "parameters=ParameterGroups(...)."
        )
    return jax.tree_util.tree_map(
        lambda leaf: eqx.is_inexact_array(leaf) and id(leaf) in ids, theta0
    )


def _deduplicated_selection(learn: Any) -> tuple:
    """Return one ``(label, path_tokens, leaf)`` entry per distinct selected array.

    ``learn`` holds the selected θ leaves.  One array object can sit at several
    places in the pytree — a declared parameter read by three equations rides on
    all three closures — and a single object cannot hold two different values,
    so those places share one entry.  The entry keeps the union of their path
    tokens, so a name that appears on any occurrence still selects it, and takes
    its label from the first occurrence's own path.
    """
    order: dict[int, int] = {}
    labels: list[str] = []
    tokens: list[set] = []
    leaves: list = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(learn):
        path_tokens = _path_key_tokens(path)
        index = order.get(id(leaf))
        if index is None:
            order[id(leaf)] = len(leaves)
            labels.append(path_tokens[-1] if path_tokens else "")
            tokens.append(set(path_tokens))
            leaves.append(leaf)
        else:
            tokens[index].update(path_tokens)
    return tuple(zip(labels, tokens, leaves))


def _in_declaration_order(selected: tuple, order: tuple) -> tuple:
    """Stably sort the selected leaves by the first declared group that owns each."""

    def rank(entry):
        _label, tokens, leaf = entry
        for index, (leaf_ids, group_tokens) in enumerate(order):
            if id(leaf) in leaf_ids or (group_tokens and tokens & group_tokens):
                return index
        return len(order)

    return tuple(sorted(selected, key=rank))


def _ravel_deduplicated(learn: Any, selected: tuple) -> tuple:
    """Ravel the deduplicated selection and return ``(flat, unflatten)``.

    ``unflatten`` writes each flat block back to EVERY position that shares the
    array object it came from, so the reconstructed θ keeps one value per
    declared unknown.
    """
    leaves, treedef = jax.tree_util.tree_flatten(learn)
    positions = {id(leaf): index for index, (_label, _tokens, leaf) in enumerate(selected)}
    slots = [positions[id(leaf)] for leaf in leaves]
    flat, unravel = jax.flatten_util.ravel_pytree([leaf for _label, _tokens, leaf in selected])

    def unflatten(x_flat):
        values = unravel(x_flat)
        return jax.tree_util.tree_unflatten(treedef, [values[slot] for slot in slots])

    return flat, unflatten


def _grouped_leaf_filter(problem: Any, theta0: Any, selection: GroupedThetaSelection) -> Any:
    """Union a [GroupedThetaSelection][diff_soup.inference.fitting._forward_seam.GroupedThetaSelection]'s sources into one θ-leaf bool mask.

    Name sources resolve via [names_to_theta_mask][] (path-token match); mask
    sources resolve via leaf identity ([selected_leaf_ids][]) so declared
    learnables whose public name is not a θ-leaf token are selected correctly.
    """
    masks = [jax.tree_util.tree_map(lambda _leaf: False, theta0)]
    for names in selection.name_sources:
        mask, _ = names_to_theta_mask(theta0, names)
        masks.append(mask)
    if selection.mask_sources:
        ids: frozenset = frozenset()
        for problem_mask in selection.mask_sources:
            ids = ids | selected_leaf_ids(problem, problem_mask)
        masks.append(
            jax.tree_util.tree_map(
                lambda leaf: eqx.is_inexact_array(leaf) and id(leaf) in ids, theta0
            )
        )
    if selection.leaf_id_sources:
        ids = frozenset().union(*selection.leaf_id_sources)
        masks.append(
            jax.tree_util.tree_map(
                lambda leaf: eqx.is_inexact_array(leaf) and id(leaf) in ids, theta0
            )
        )
    return jax.tree_util.tree_map(lambda *bs: any(bs), *masks)


def _selected_theta_names(selected: tuple) -> tuple:
    """Per-scalar names of the selected θ leaves, in ravel order.

    The ravel flattens each selected array into its scalar components; this
    expands each leaf's label by its size so the names align 1:1 with the flat
    decision vector (a scalar leaf → one name).  The label is the last token of
    the leaf's path — for a declared parameter or function, its declared name.
    """
    names: list[str] = []
    for label, _tokens, leaf in selected:
        label = label or "theta"
        size = int(np.asarray(leaf).size)
        if size == 1:
            names.append(label)
        else:
            names.extend(f"{label}[{i}]" for i in range(size))
    return tuple(names)


def _selected_theta_token_sets(selected: tuple) -> tuple:
    """Per-scalar path-token sets of the selected θ leaves, in ravel order.

    Each entry is the ``frozenset`` of human-readable key tokens along that
    scalar's leaf path (attribute names and dict keys).  A parameter-group name
    selects a scalar when it appears in that scalar's token set — the same rule
    [names_to_theta_mask][] uses — so groups map to flat-vector columns even
    when the group name is a non-leaf path token (e.g. ``"j0"`` for a leaf at
    ``functions.j0.weight``).
    """
    token_sets: list = []
    for _label, tokens, leaf in selected:
        toks = frozenset(tokens)
        size = int(np.asarray(leaf).size)
        token_sets.extend([toks] * size)
    return tuple(token_sets)


def _selected_theta_leaf_ids(selected: tuple) -> tuple:
    """Per-scalar ``id()`` of the source θ leaf, in ravel order.

    Each scalar carries the identity of the array leaf it was raveled from; a
    parameter group built from a [LearnableSelector][diff_soup.inference.fitting.learnables.LearnableSelector]
    owns a scalar when its leaf identity is in the selector's selected set — the
    identity rule that resolves declared learnables whose name is not a path token.
    """
    leaf_ids: list = []
    for _label, _tokens, leaf in selected:
        size = int(np.asarray(leaf).size)
        leaf_ids.extend([id(leaf)] * size)
    return tuple(leaf_ids)


@dataclasses.dataclass(frozen=True)
class StreamPlan:
    """Host-static plan for slicing one stream out of a shared union solve.

    Attributes:
        name: The stream name.
        frame_indices: Indices into the union save grid for this stream's frames.
        keep_indices: Indices (into this stream's frames) that survive the
            ``frame_mask``, or ``None`` if every frame is observed.
        row_slice: ``(start, stop)`` row range of this stream in the concatenated
            residual vector.
    """

    name: str
    frame_indices: np.ndarray
    keep_indices: Optional[np.ndarray]
    row_slice: tuple


def union_prediction_factory(
    seam: ForwardSeam,
    *,
    streams: tuple,
    y0: jax.Array,
    solver: Any,
    t0: Optional[float] = None,
    decision_to_theta: Optional[Callable[[jax.Array], Any]] = None,
    decision_to_y0: Optional[Callable[[jax.Array], jax.Array]] = None,
    schedule: Any = None,
) -> tuple[Callable[[jax.Array], tuple], tuple, np.ndarray, dict]:
    """Build the single-solve per-stream prediction over the union time grid.

    This is the one pure prediction closure both fitting and design/inference
    consume: exactly ONE ``pdesolve`` over the union of every stream's
    observation times, each stream's frames sliced out of that one trajectory
    and pushed through its observation operator (frame times and the trial
    theta threaded through).

    Streams only need observation times, an optional ``frame_mask``, and an
    ``observe`` method — ``values`` (targets) and ``weight`` may be ``None``,
    so a design-stage declaration with no data yet can predict through the
    same code path a data fit uses.

    Args:
        seam: The built [ForwardSeam][diff_soup.inference.fitting._forward_seam.ForwardSeam].
        streams: Stream declarations in order. Each must expose ``ts``,
            ``frame_mask``, ``observe``; ``values``/``weight`` are optional.
        y0: Initial differential state at the trajectory IC.
        solver: The (already AD-resolved) solver config.
        t0: Optional IC time; defaults to the earliest stream time.
        decision_to_theta: Optional reconstruction for a decision vector that
            carries variables after theta.
        decision_to_y0: Optional reconstruction of a learned initial state from
            the decision vector. The fixed ``y0`` is used when omitted.
        schedule: Optional segment arrays of an operating protocol
            ([ScheduleArrays][diff_soup._schedule.ScheduleArrays]). With them the
            one solve follows the schedule's segments from the first union time
            ([solve_schedule][diff_soup._schedule.solve_schedule]) in place of
            one ``pdesolve``.

    Returns:
        ``(predict, stream_plans, union_ts, solve_probe)`` — ``predict(x)``
        returns ``(predictions, failed)`` where ``predictions`` holds each
        stream's observable-space values at its kept (unmasked) frames and
        ``failed`` flags any unsuccessful terminal solver code as well as
        non-finite states or multipliers in the solution; the per-stream
        [StreamPlan][diff_soup.inference.fitting._forward_seam.StreamPlan] tuple (row slices are zero-length for streams
        without values); the union save grid (numpy); and a ``solve_probe``
        dict whose ``"n"`` counts forward solves performed at trace time (one
        per prediction — the single-solve invariant).
    """
    from diff_soup import SaveAt
    from diff_soup.pdesolve import pdesolve

    all_ts = np.concatenate([np.asarray(s.ts).reshape((-1,)) for s in streams])
    t_ic = float(np.min(all_ts)) if t0 is None else float(t0)
    union_ts = np.unique(np.concatenate([np.asarray([t_ic], dtype=all_ts.dtype), all_ts]))
    t_span = (float(union_ts[0]), float(union_ts[-1]))
    saveat = SaveAt(ts=jnp.asarray(union_ts))

    plans: list[StreamPlan] = []
    row_start = 0
    for s in streams:
        s_ts = np.asarray(s.ts).reshape((-1,))
        idx = np.searchsorted(union_ts, s_ts)
        if idx.size and (idx.max() >= union_ts.size or not np.allclose(union_ts[idx], s_ts)):
            idx = np.array([int(np.argmin(np.abs(union_ts - t))) for t in s_ts])
        if s.frame_mask is None:
            keep = None
            n_kept = s_ts.shape[0]
        else:
            keep = np.where(np.asarray(s.frame_mask).reshape((-1,)).astype(bool))[0]
            n_kept = int(keep.shape[0])
        values = getattr(s, "values", None)
        if values is None:
            row_len = 0
        else:
            v = np.asarray(values)
            per_frame = int(np.prod(v.shape[1:])) if v.ndim > 1 else 1
            row_len = n_kept * per_frame
        plans.append(
            StreamPlan(
                name=s.name,
                frame_indices=idx,
                keep_indices=keep,
                row_slice=(row_start, row_start + row_len),
            )
        )
        row_start += row_len

    adjoint = getattr(solver, "adjoint", None)
    solve_probe = {"n": 0}
    # Bind the per-stream observe/frame-times outside the closure. Frame times
    # feed time-aware observables and the trial theta feeds
    # parameter-dependent observables; operators that read neither ignore them.
    bound = tuple(
        (
            jnp.asarray(p.frame_indices),
            None if p.keep_indices is None else jnp.asarray(p.keep_indices),
            s.observe,
            jnp.asarray(union_ts[p.frame_indices]),
        )
        for p, s in zip(plans, streams)
    )

    def predict(x_flat: jax.Array) -> tuple:
        solve_probe["n"] += 1
        theta = seam.theta_of(x_flat) if decision_to_theta is None else decision_to_theta(x_flat)
        initial_state = y0 if decision_to_y0 is None else decision_to_y0(x_flat)
        if schedule is None:
            sol = pdesolve(
                seam.problem,
                solver,
                t_span,
                initial_state,
                saveat=saveat,
                theta=theta,
                adjoint=adjoint,
            )
        else:
            from diff_soup._schedule import solve_schedule

            sol, _ends = solve_schedule(
                seam.problem,
                solver,
                schedule,
                initial_state,
                t0=t_span[0],
                t_end=t_span[1],
                times=saveat.ts,
                theta=theta,
                adjoint=adjoint,
            )
        ys = sol.ys
        preds = []
        for frame_idx, keep_idx, observe, frame_ts in bound:
            obs_pred = observe(ys[frame_idx], frame_ts, theta)
            if keep_idx is not None:
                obs_pred = obs_pred[keep_idx]
            preds.append(obs_pred)
        failed = _solution_failed(sol)
        return tuple(preds), failed

    return predict, tuple(plans), union_ts, solve_probe


def union_residual_factory(
    seam: ForwardSeam,
    *,
    streams: tuple,
    y0: jax.Array,
    solver: Any,
    t0: Optional[float] = None,
    decision_to_theta: Optional[Callable[[jax.Array], Any]] = None,
    decision_to_y0: Optional[Callable[[jax.Array], jax.Array]] = None,
    schedule: Any = None,
) -> tuple[Callable[[jax.Array], jax.Array], tuple, np.ndarray, dict]:
    """Build a single-solve residual over the union of all streams' times.

    A thin assembly over [union_prediction_factory][]: the shared
    prediction closure produces each stream's kept-frame predictions, and this
    factory subtracts targets and applies the ``1/σ`` weights.

    Residual row order is deterministic: streams in declaration order, then time
    order, then flattened observation order. Any unsuccessful or non-finite
    solver result stamps the whole residual with the sentinel so host
    re-evaluation can detect and surface the failure.

    Args:
        seam: The built [ForwardSeam][diff_soup.inference.fitting._forward_seam.ForwardSeam].
        streams: The problem's [Series][diff_soup.inference.fitting.series.Series]
            streams (declaration order preserved).
        y0: Initial differential state at the trajectory IC.
        solver: The (already AD-resolved) solver config.
        t0: Optional IC time; defaults to the earliest stream time.
        decision_to_theta: Optional theta reconstruction for an extended
            decision vector.
        decision_to_y0: Optional learned-initial-state reconstruction.
        schedule: Optional segment arrays of an operating protocol, passed to
            [union_prediction_factory][diff_soup.inference.fitting._forward_seam.union_prediction_factory].

    Returns:
        ``(residual, stream_plans, union_ts, solve_probe)`` — the residual
        closure ``x -> r``; the per-stream [StreamPlan][diff_soup.inference.fitting._forward_seam.StreamPlan] tuple (for
        diagnostics by name); the union save grid (numpy); and a ``solve_probe``
        dict whose ``"n"`` counts forward solves performed at trace time (one per
        residual evaluation — the single-solve invariant).
    """
    predict, plans, union_ts, solve_probe = union_prediction_factory(
        seam,
        streams=streams,
        y0=y0,
        solver=solver,
        t0=t0,
        decision_to_theta=decision_to_theta,
        decision_to_y0=decision_to_y0,
        schedule=schedule,
    )

    # Host-side per-stream target/noise-model binding at the kept frames.
    bound = []
    for p, s in zip(plans, streams):
        keep = p.keep_indices
        targets = s.values if keep is None else s.values[jnp.asarray(keep)]
        weight = s.weight
        likelihood = getattr(s, "likelihood", None)
        likelihood_context = getattr(s, "likelihood_context", None)
        if (
            weight is not None
            and keep is not None
            and getattr(weight, "ndim", 0) >= 1
            and weight.shape[0] == s.values.shape[0]
        ):
            weight = weight[jnp.asarray(keep)]
        if likelihood is not None and keep is not None:
            likelihood, likelihood_context = likelihood.for_frames(
                jnp.asarray(keep),
                full_prediction_shape=tuple(s.values.shape),
                context=likelihood_context,
            )
        bound.append((targets, weight, likelihood, likelihood_context))
    bound = tuple(bound)

    y0_dtype = jnp.asarray(y0).dtype

    def residual(x_flat: jax.Array) -> jax.Array:
        preds, failed = predict(x_flat)
        rows = []
        for obs_pred, (targets, weight, likelihood, likelihood_context) in zip(preds, bound):
            if likelihood is not None:
                r = likelihood.least_squares_rows(
                    targets,
                    obs_pred,
                    context=likelihood_context,
                )
            else:
                r = obs_pred - targets
            if weight is not None:
                r = weight * r
            rows.append(jnp.reshape(r, (-1,)))
        r_all = jnp.concatenate(rows) if rows else jnp.zeros((0,), dtype=y0_dtype)
        return jnp.where(failed, jnp.full_like(r_all, _FAILED_SENTINEL), r_all)

    return residual, tuple(plans), union_ts, solve_probe


def stack_frame_likelihoods(
    likelihood: Any,
    target_frames: Sequence[int],
    *,
    full_prediction_shape: tuple,
    context: Any,
    values: np.ndarray,
    who: str,
) -> tuple[Any, Any]:
    """Stack per-frame likelihood views on a leading window axis.

    Windowing splits one trajectory into independent per-window observation
    events, so a declared likelihood may only be used if it factorizes over
    frames: the sum of the per-frame squared whitened norms has to equal the
    whole-series squared norm at the same prediction.  Independent families
    (a known-scale Gaussian, count models, per-frame correlated channels)
    satisfy this; a covariance that couples different frames does not, and
    splitting the trace would silently discard that coupling.

    Separability is checked numerically at build time against the declared data
    and a deterministic set of probe predictions — no type test, so a custom
    likelihood that genuinely factorizes is admitted on its behaviour.  The set
    perturbs each scored frame on its own axis, and then every frame at once: a
    single uniform probe certifies nothing, because a cross-frame precision
    block can annihilate one direction while staying active in the rest.

    Args:
        likelihood: The declared likelihood.
        target_frames: Frame indices scored by the kept windows, in window order.
        full_prediction_shape: Shape of the whole-series prediction array.
        context: Optional likelihood context, forwarded to ``for_frames``.
        values: The observed values, used to build the separability probes.
        who: Caller name used in error messages.

    Returns:
        ``(stacked_likelihood, context)`` where the stacked object's frame-aligned
        leaves carry a leading window axis suitable for ``jax.vmap``.

    Raises:
        NotImplementedError: If the likelihood couples different frames, or if its
            per-frame views do not share one structure.
    """
    values = np.asarray(values)
    views = []
    contexts = []
    for frame in target_frames:
        view, view_context = likelihood.for_frames(
            jnp.asarray([int(frame)]),
            full_prediction_shape=tuple(full_prediction_shape),
            context=context,
        )
        views.append(view)
        contexts.append(view_context)

    structures = {jax.tree_util.tree_structure(view) for view in views}
    if len(structures) != 1:
        raise NotImplementedError(
            f"{who}: the declared likelihood produces per-frame views with different "
            "structures, so they cannot be batched over windows. Use SingleShoot, or "
            "declare a likelihood whose frame-aligned parameters have one fixed shape."
        )
    if any(later is not contexts[0] for later in contexts[1:]):
        raise NotImplementedError(
            f"{who}: the declared likelihood returns a per-frame context, which the "
            "window batch cannot carry. Use SingleShoot, or declare a likelihood whose "
            "context is shared across frames."
        )

    stacked = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack([jnp.asarray(leaf) for leaf in leaves]), *views
    )

    # Separability check: at every probe prediction the stacked per-frame whitened
    # norm must equal the norm the same likelihood assigns to those frames jointly.
    # Only the scored frames contribute, so both sides are taken over that subset,
    # and the probe set spans them one frame at a time so no cross-frame block can
    # hide in a direction that is never exercised.
    frames = np.asarray([int(frame) for frame in target_frames])
    subset_view, subset_context = likelihood.for_frames(
        jnp.asarray(frames),
        full_prediction_shape=tuple(full_prediction_shape),
        context=context,
    )
    observed = jnp.asarray(values[frames])
    n_kept = int(frames.shape[0])
    trailing = (1,) * (observed.ndim - 1)
    directions = jnp.concatenate(
        [
            jnp.reshape(jnp.eye(n_kept, dtype=observed.dtype), (n_kept, n_kept) + trailing),
            jnp.ones((1, n_kept) + trailing, dtype=observed.dtype),
        ]
    )
    probes = observed + jnp.asarray(0.125, dtype=observed.dtype) * directions

    def _joint_sq(prediction):
        rows = subset_view.least_squares_rows(observed, prediction, context=subset_context)
        return jnp.sum(jnp.asarray(rows) ** 2)

    def _stacked_sq(prediction):
        rows = jax.vmap(
            lambda view, frame_observed, frame_prediction: view.least_squares_rows(
                frame_observed, frame_prediction, context=contexts[0]
            )
        )(stacked, observed, prediction)
        return jnp.sum(jnp.asarray(rows) ** 2)

    joint_sq = np.asarray(jax.vmap(_joint_sq)(probes), dtype=float)
    per_frame_sq = np.asarray(jax.vmap(_stacked_sq)(probes), dtype=float)
    eps = float(np.finfo(probes.dtype).eps)
    tolerance = max(1e-9, 64.0 * eps) * np.maximum(np.abs(joint_sq), 1.0)
    # A non-finite gap is never "within tolerance": the negation keeps NaN a failure.
    failed = ~(np.abs(per_frame_sq - joint_sq) <= tolerance)
    if bool(np.any(failed)):
        worst = int(np.flatnonzero(failed)[0])
        perturbed = (
            "every scored frame at once"
            if worst == n_kept
            else f"scored frame {int(frames[worst])} alone"
        )
        raise NotImplementedError(
            f"{who}: the declared likelihood couples different frames (perturbing "
            f"{perturbed}, its per-frame whitened norms sum to {per_frame_sq[worst]:.6g} "
            f"against {joint_sq[worst]:.6g} for the same frames scored jointly), so "
            "splitting the trajectory into windows would discard the cross-window "
            "covariance. Use SingleShoot, or declare a likelihood that factorizes over "
            "frames."
        )

    return stacked, contexts[0]


class ObservedWindows(NamedTuple):
    """The kept windows of an observed-start trajectory, as host numbers.

    Window ``k`` starts at ``t0[k]`` from the measured full state ``starts[k]``
    and is scored at ``t1[k]``, the time of frame ``frames[k]`` of the scored
    stream; a window is kept when that frame is observed.

    Attributes:
        frames: Per window, the frame of the scored stream it is scored at.
        t0: Per window, the time it starts at, shape ``(K,)``.
        t1: Per window, the time it is scored at, shape ``(K,)``.
        starts: Per window, the measured state it starts from, one leading
            entry per window.
    """

    frames: tuple
    t0: np.ndarray
    t1: np.ndarray
    starts: np.ndarray


def observed_windows(series: Any, *, scored: Any = None) -> ObservedWindows:
    """The kept windows of an observed-start trajectory, read on the host.

    Window ``i -> i+1`` starts from the measured full state at frame ``i`` of
    ``series`` and is kept when frame ``i+1`` of the scored stream is observed,
    per its ``frame_mask``.

    Args:
        series: The full-state stream whose measured states start the windows.
        scored: The stream the windows are scored on; ``None`` when ``series``
            is.

    Returns:
        The [ObservedWindows][diff_soup.inference.fitting._forward_seam.ObservedWindows].

    Raises:
        ValueError: If the ``frame_mask`` removes every window (no observed target).
    """
    target = series if scored is None else scored
    ts = np.asarray(series.ts)
    ys = np.asarray(series.values)
    n_frames = ts.shape[0]
    if target.frame_mask is None:
        keep_targets = np.ones(n_frames, dtype=bool)
    else:
        keep_targets = np.asarray(target.frame_mask).reshape((-1,)).astype(bool)
    kept = [i for i in range(n_frames - 1) if bool(keep_targets[i + 1])]
    if not kept:
        raise ValueError(
            "Multishoot(Windows.from_data()): frame_mask removed every window (no observed "
            "target frame). At least one frame after the first must be observed."
        )
    return ObservedWindows(
        frames=tuple(i + 1 for i in kept),
        t0=np.asarray([float(ts[i]) for i in kept]),
        t1=np.asarray([float(ts[i + 1]) for i in kept]),
        starts=np.stack([ys[i] for i in kept]),
    )


def batched_transition_residual_factory(
    seam: ForwardSeam,
    *,
    series: Any,
    solver: Any,
    scored: Any = None,
) -> tuple[Callable, Callable, int, list]:
    """Build one vmapped residual over observed-start window transitions.

    Each kept window ``i -> i+1`` starts from the measured full state at frame ``i``
    in ``series`` and scores the predicted endpoint ``u(t_{i+1})``. Every kept window
    runs through a single `jax.vmap`, so the traced program — and its jaxpr
    equation count — is independent of the window count.  (The alternative, one
    jitted closure per window, makes both the jaxpr and the compilation cost grow
    with the number of windows.)  Mapping the windows is also what lets a sparse
    GPU backend factorize them as one batched dispatch instead of a serial sweep.

    The stream that seeds the window starts and the stream that is scored are
    separate declarations.  With ``scored=None`` the seed stream is also the scored
    stream and the endpoint is compared as a full state.  With ``scored`` given, the
    endpoint is mapped through *that* stream's observable and compared against its
    values, so a rank-deficient readout — a pixel average, a partial state — can be
    scored while the window starts still come from a known full state.

    A window ``i -> i+1`` is kept iff its target frame ``i+1`` is observed, per the
    ``frame_mask`` of the scored stream; the start frame is always used as the
    window's initial state. Any unsuccessful or non-finite solver result in a
    window stamps that window's rows with the sentinel so host re-evaluation
    can detect and surface it.

    Args:
        seam: The built [ForwardSeam][diff_soup.inference.fitting._forward_seam.ForwardSeam].
        series: The full-state [Series][diff_soup.inference.fitting.series.Series] whose
            measured states seed each window start.
        solver: The (already AD-resolved) solver config.
        scored: Optional separate [Series][diff_soup.inference.fitting.series.Series]
            supplying the observable, targets, weights, frame mask, and likelihood.
            ``None`` scores the seed stream itself.

    Returns:
        ``(batched_r_fn, batched_rj_fn, n_windows, baseline_windows)`` — the jitted
        residual ``x -> r_flat`` (rows in window order); the jitted ``x ->
        (r_flat, J)`` (one forward-mode pass over the whole map); the kept-window
        count; and a list of ``(t0, t1, y_start)`` host arrays for the host-side
        loud-status baseline solve.

    Raises:
        ValueError: If the ``frame_mask`` removes every window (no observed target).
        NotImplementedError: If the scored stream's likelihood couples frames.
    """
    from diff_soup import SaveAt
    from diff_soup.pdesolve import pdesolve

    target = series if scored is None else scored
    windows = observed_windows(series, scored=scored)
    targets_all = np.asarray(target.values)
    n_frames = np.asarray(series.ts).shape[0]
    weight = target.weight
    observe = None if scored is None else scored.observe
    who = "Multishoot(Windows.from_data())"
    target_frames = list(windows.frames)

    likelihood = getattr(target, "likelihood", None)
    stacked_likelihood = None
    likelihood_context = None
    if likelihood is not None:
        stacked_likelihood, likelihood_context = stack_frame_likelihoods(
            likelihood,
            target_frames,
            full_prediction_shape=targets_all.shape,
            context=getattr(target, "likelihood_context", None),
            values=targets_all,
            who=who,
        )

    t0s = jnp.asarray(windows.t0)
    t1s = jnp.asarray(windows.t1)
    y_starts = jnp.asarray(windows.starts)
    obs_targets = jnp.stack([jnp.asarray(targets_all[frame]) for frame in target_frames])
    # Each kept window i -> i+1 compares against the target frame i+1, so it is
    # weighted by that frame's ``1/σ`` row.  A frame-aligned weight (leading axis ==
    # n_frames) is indexed at the target frame; a scalar or per-state weight broadcasts
    # identically across windows.  Multiplying by the whole ``weight`` would
    # broadcast every frame's weight against one window's residual and inflate the rows.
    has_w = weight is not None and stacked_likelihood is None
    if has_w:
        w_arr = jnp.asarray(weight)
        if w_arr.ndim >= 1 and w_arr.shape[0] == n_frames:
            w_targets = jnp.stack([w_arr[frame] for frame in target_frames])
        else:
            w_targets = jnp.stack([w_arr for _ in target_frames])
    else:
        w_targets = jnp.zeros((len(target_frames),), dtype=obs_targets.dtype)  # unused placeholder
    saveat = SaveAt(t1=True)
    adjoint = getattr(solver, "adjoint", None)

    def _one(x_flat: jax.Array, seg: tuple) -> jax.Array:
        t0_i, t1_i, y_start_i, target_i, w_i, likelihood_i = seg
        theta = seam.theta_of(x_flat)
        sol = pdesolve(
            seam.problem,
            solver,
            (t0_i, t1_i),
            y_start_i,
            saveat=saveat,
            theta=theta,
            adjoint=adjoint,
        )
        end_state = sol.ys[-1]
        # The scored stream's operator maps a frame stack; hand it one frame at
        # the window's end time with the trial theta, as the union residual does.
        predicted = (
            end_state
            if observe is None
            else observe(end_state[None], jnp.reshape(t1_i, (1,)), theta)[0]
        )
        if likelihood_i is not None:
            r = likelihood_i.least_squares_rows(target_i, predicted, context=likelihood_context)
        else:
            r = predicted - target_i
            if has_w:
                r = w_i * r
        r = jnp.reshape(r, (-1,))
        failed = _solution_failed(sol)
        return jnp.where(failed, jnp.full_like(r, _FAILED_SENTINEL), r)

    # An enclosing vmap over parameter points folds into the window axis on a
    # forward-only seam, so the windows' solves keep one matrix-batch axis.
    fold = folds_enclosing_batches(solver)

    def batched_r(x_flat: jax.Array) -> jax.Array:
        def one_window(x, t0, t1, y, target_i, w, likelihood_i):
            return _one(x, (t0, t1, y, target_i, w, likelihood_i))

        likelihood_axis = None if stacked_likelihood is None else 0
        rs = lane_map(
            one_window,
            (None, 0, 0, 0, 0, 0, likelihood_axis),
            x_flat,
            t0s,
            t1s,
            y_starts,
            obs_targets,
            w_targets,
            stacked_likelihood,
            fold=fold,
        )
        return jnp.reshape(rs, (-1,))

    def batched_rj(x_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        def f_aux(p: jax.Array):
            r = batched_r(p)
            return r, r

        jac, r = jax.jacfwd(f_aux, has_aux=True)(x_flat)
        return r, jac

    baseline_windows = [
        (float(t0), float(t1), jnp.asarray(start))
        for t0, t1, start in zip(windows.t0, windows.t1, windows.starts)
    ]
    return jax.jit(batched_r), jax.jit(batched_rj), len(target_frames), baseline_windows


def count_failed_frames(residual_fn: Callable, x_flat: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Host-side detection of sentinel-stamped (failed) residual blocks.

    Re-evaluates ``residual_fn`` at ``x_flat`` and reports whether any element
    reaches the sentinel magnitude (real residuals are finite and far smaller).

    Args:
        residual_fn: The residual closure ``x -> r``.
        x_flat: Flat vector to evaluate at (typically the optimizer's final iterate).

    Returns:
        ``(n_failed, failed_indices)`` — for the single-solve path ``n_failed`` is
        0 or 1 (one trajectory); indices are ``(0,)`` when it failed.
    """
    x = jnp.asarray(x_flat)
    r = np.asarray(residual_fn(x))
    if r.size and np.any(np.abs(r) >= _FAILED_SENTINEL):
        return 1, (0,)
    return 0, ()
