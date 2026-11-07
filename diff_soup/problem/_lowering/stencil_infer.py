"""Build-time stencil inference for the sparse Jacobian topology.

The per-``(write, read)`` Jacobian block of a structured-grid component is
declared from the grid's stencil pattern.  The stencil an equation body
actually exercises can be wider than the built-in radius-1 pattern — e.g. a
composed ``laplacian(laplacian(c))`` reaches two cells per axis — and a
sparse-direct factorization built for an under-declared pattern is
structurally wrong.  This module measures the true reach at lowering time by
numerically probing the assembled residual:

* perturb one cell of one read field (a forward-mode JVP with a one-cell
  tangent) and record which residual cells of each write field respond;
* convert responding cells to per-axis stencil offsets, wrapping only on the
  axes declared periodic;
* probe an interior cell first, then cells in bands near each boundary,
  deepening the bands until the measured reach stops growing (a
  boundary-modified scheme can read farther into the interior than the
  interior stencil suggests), at two random states and two smooth monotone
  ones, and take the union.

Probing measures the dependencies ACTIVE at the probe states: a dependency
whose coefficient is exactly zero at every probe state (and at every probed
cell) stays invisible and is under-declared.  A flux limiter is the common
case: it reads the second upstream cell only where the data are monotone,
which a random state gives at a third of the cells or fewer, so the two
smooth states (rising and falling along every axis) are probed too.
Independent states and the per-cell probes make a miss unlikely for physical
models, but it is the documented residual risk of a numerical-probing
approach.

The union is widened with the radius-1 star so inference can only ever add
entries relative to the default declaration.  Blocks whose measured offsets
stay inside the radius-1 star are omitted from the result, so the default
generators run unchanged for them.  A response farther than
``_GLOBAL_COUPLING_RADIUS`` cells from the probe (after periodic wrapping)
means the dependency is not a local stencil (e.g. an integral of the field);
such blocks are declared dense.  A probe state that produces non-finite
responses is discarded and redrawn; if no clean state is found the default
radius-1 declaration is kept and a warning names the affected component.

Probing evaluates a handful of JVPs of the physics-only residual (boundary
row scatters excluded — their topology is declared by the dedicated BC
cross-block paths) and costs a few residual evaluations per read field at
build time; nothing here runs during the solve.
"""

from __future__ import annotations

import numpy as np

from diff_soup._warnings import warn_user

# A dependency farther than this many cells from the perturbed cell (per
# axis, after periodic wrapping) is treated as global coupling and the block
# is declared dense instead of widening the stencil.
_GLOBAL_COUPLING_RADIUS = 8

# Probe JVPs are evaluated in vmapped chunks of this size (memory is
# ``chunk * state_size`` per evaluation).
_PROBE_CHUNK = 8

# Number of random probe states unioned per problem, and redraw attempts
# allowed when a state produces non-finite responses.
_N_PROBE_STATES = 2
_MAX_STATE_DRAWS = 4

_RNG_SEED = 20260702


def _grid_shape(domain) -> tuple[int, ...] | None:
    """Per-axis cell counts of a structured grid domain, else ``None``.

    ``GridND`` exposes per-axis grids via ``axes``; ``Grid1D`` exposes its
    cell count via ``n``.  Domains without either (FEM meshes, custom
    domains) are not probed and keep their declared pattern.
    """
    axes = getattr(domain, "axes", None)
    if axes is not None:
        return tuple(int(ax.n) for ax in axes)
    n = getattr(domain, "n", None)
    if n is not None:
        return (int(n),)
    return None


def _multi_index(shape: tuple[int, ...], flat: np.ndarray) -> np.ndarray:
    """Row-major multi-indices ``(k, ndim)`` for flat cell indices ``(k,)``."""
    ndim = len(shape)
    strides = np.ones(ndim, dtype=np.int64)
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    flat = np.asarray(flat, dtype=np.int64)
    out = np.empty((flat.size, ndim), dtype=np.int64)
    rem = flat.copy()
    for d in range(ndim):
        out[:, d] = rem // strides[d]
        rem = rem % strides[d]
    return out


def _response_offsets(
    shape: tuple[int, ...],
    periodic_axes: tuple[int, ...],
    probe_cell: int,
    response_cells: np.ndarray,
) -> np.ndarray:
    """Unique stencil offsets measured from one probe.

    A responding residual row depends on the perturbed column, so each
    response contributes the offset ``probe_cell - response_cell`` per axis —
    the column-minus-row convention the pattern generator consumes (a row
    ``r`` is declared to read column ``r + offset``).  Getting this sign
    right matters exactly for asymmetric stencils (e.g. upwind
    compositions): the flipped convention declares the transposed pattern.

    On periodic axes the offset is wrapped to its signed minimum-magnitude
    representative (a response across the domain edge is a wrap neighbour,
    not a long-range dependency).  Non-periodic axes keep the raw signed
    distance so genuine long-range coupling stays visible.

    Returns:
        Integer array ``(n_unique_offsets, ndim)``.
    """
    shape_arr = np.asarray(shape, dtype=np.int64)
    probe_m = _multi_index(shape, np.asarray([probe_cell]))[0]
    resp_m = _multi_index(shape, np.asarray(response_cells, dtype=np.int64))
    diff = probe_m[None, :] - resp_m
    for d in periodic_axes:
        s = int(shape_arr[d])
        diff[:, d] = (diff[:, d] + s // 2) % s - s // 2
    return np.unique(diff, axis=0)


def _probe_cells(shape: tuple[int, ...], depths: tuple[int, ...]) -> np.ndarray:
    """Flat indices of the cells to perturb: interior + per-axis boundary bands.

    One interior cell (the grid center) plus, for each axis, the cells within
    ``depths[axis]`` of either end of that axis (other axes held at the
    center).  Boundary-modified schemes only deviate from the interior
    stencil near a boundary, so these bands are where a wider reach shows.

    Returns:
        Sorted unique flat cell indices.
    """
    shape_arr = np.asarray(shape, dtype=np.int64)
    ndim = len(shape)
    strides = np.ones(ndim, dtype=np.int64)
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape_arr[d + 1]
    center = shape_arr // 2
    cells = {int(center @ strides)}
    for d in range(ndim):
        depth = min(int(depths[d]), int(shape_arr[d]))
        for k in range(depth):
            for idx_d in (k, int(shape_arr[d]) - 1 - k):
                m = center.copy()
                m[d] = idx_d
                cells.add(int(m @ strides))
    return np.asarray(sorted(cells), dtype=np.int64)


def _smooth_state(shape: tuple[int, ...], rising: bool) -> np.ndarray:
    """Per-cell values in ``[1, 1.5]``, convex and monotone along every axis.

    Rising, the ratio of successive differences is below one in the rising
    direction; falling, in the other.  Between them a flux limiter sees a ratio
    in ``(0, 1)`` for either flow direction, where every limiter reads its
    second upstream cell.
    """
    axes = [np.arange(n) / max(n - 1, 1) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    q = sum((g if rising else 1.0 - g) ** 2 for g in grids) / len(shape)
    return 1.0 + 0.5 * q.reshape(-1)


def _star1_offsets(ndim: int) -> set[tuple[int, ...]]:
    """The radius-1 star: the origin plus one step along each axis."""
    star = {(0,) * ndim}
    for d in range(ndim):
        for s in (-1, 1):
            o = [0] * ndim
            o[d] = s
            star.add(tuple(o))
    return star


def _eligible_blocks(eqset):
    """Probe-eligible ``(write, read)`` node-field pairs per grid component.

    Mirrors the block walk of the sparsity builders: a pair is eligible when
    both variables are per-node fields of a component whose domain is a
    structured grid.  Globals and no-DOF read-through aliases carry no local
    columns and are skipped, exactly as in the declared-topology walk.

    Returns:
        A list of ``(comp, shape, periodic_axes, pairs)`` tuples where
        ``pairs`` maps each read Variable to the set of write Variables whose
        equations read it, with the per-node component counts needed to
        address DOFs.
    """
    from diff_soup.problem._ir.equation import MultiEquation
    from diff_soup.problem._ir.variable import GlobalVariable

    from .sparsity import _component_periodic_axes, _node_components

    read_through_ids = {
        id(cp.target) for cp in eqset.couplings if getattr(cp, "read_through", False)
    }
    out = []
    for comp in eqset.components:
        shape = _grid_shape(comp.domain)
        if shape is None or not hasattr(comp.domain, "sparsity_pattern"):
            continue
        num_nodes_attr = getattr(comp.domain, "num_nodes", None)
        if num_nodes_attr is None or int(num_nodes_attr) != int(np.prod(shape)):
            continue
        num_nodes = int(num_nodes_attr)
        pairs: dict = {}
        for eq in comp.equations:
            output_vars = eq.vars if isinstance(eq, MultiEquation) else (eq.var,)
            for out_var in output_vars:
                d_out = _node_components(out_var, num_nodes)
                if d_out is None:
                    continue
                for inp in eq.inputs:
                    if isinstance(inp, GlobalVariable) or id(inp) in read_through_ids:
                        continue
                    d_in = _node_components(inp, num_nodes)
                    if d_in is None:
                        continue
                    pairs.setdefault(id(inp), (inp, d_in, {}))[2][id(out_var)] = (out_var, d_out)
        if pairs:
            out.append((comp, shape, _component_periodic_axes(comp), pairs))
    return out


def _make_batched_jvp(f_of_u, compute_dtype):
    """Compile-once batched forward-mode probe of the residual.

    Returns ``responses(u, lam, t, tangents)`` evaluating ``d/de f(u + e *
    tangent)`` for each row of the host ``tangents`` array.  The state enters
    as arguments (not closure) and every executed batch is zero-padded to
    ``_PROBE_CHUNK`` rows, so ONE jitted program serves every probe of a
    build; padding rows carry a zero tangent and are sliced off the result.
    """
    import jax
    import jax.numpy as jnp

    compute_dtype = jax.dtypes.canonicalize_dtype(compute_dtype)
    batched = jax.jit(
        jax.vmap(
            lambda u, lam, t, v: jax.jvp(lambda uu: f_of_u(uu, lam, t), (u,), (v,))[1],
            in_axes=(None, None, None, 0),
        )
    )

    def responses(u, lam, t, tangents):
        outs = []
        for i in range(0, tangents.shape[0], _PROBE_CHUNK):
            chunk = tangents[i : i + _PROBE_CHUNK]
            n_real = chunk.shape[0]
            if n_real < _PROBE_CHUNK:
                pad = np.zeros((_PROBE_CHUNK - n_real, chunk.shape[1]))
                chunk = np.concatenate([chunk, pad], axis=0)
            r = np.asarray(batched(u, lam, t, jnp.asarray(chunk, dtype=compute_dtype)))
            outs.append(r[:n_real])
        return np.concatenate(outs, axis=0)

    return responses


def _infer_stencil_offsets(
    eqset,
    offset_of,
    state_size: int,
    constraint_dim: int,
    f_of_u,
    compute_dtype,
) -> dict:
    """Measure each node-field Jacobian block's true stencil offset set.

    Probes the assembled physics residual ``f_of_u(u, lam, t)`` with one-cell
    JVP tangents (see the module docstring for the probing strategy) and
    returns, per ``(component, write, read)`` block whose measured reach
    exceeds the radius-1 star, the offset set to generate the declared
    pattern from — or ``None`` for a block whose dependency is not a local
    stencil (declared dense by the caller).  Blocks inside the radius-1 star
    are omitted so the default declaration path runs unchanged for them.

    Args:
        eqset: The equation set being lowered.
        offset_of: Callable mapping ``(comp_name, var_name)`` to the
            variable's offset in the flat state vector.
        state_size: Flat state size (the residual's length).
        constraint_dim: Length of the bordered-global vector ``lam``.
        f_of_u: The physics residual restricted to its state output.
        compute_dtype: Dtype for the probe states.

    Returns:
        Dict mapping ``(comp_name, write_name, read_name)`` to an integer
        offset array ``(n_offsets, ndim)`` or ``None`` (dense).
    """
    import jax
    import jax.numpy as jnp

    compute_dtype = jax.dtypes.canonicalize_dtype(compute_dtype)
    blocks = _eligible_blocks(eqset)
    if not blocks:
        return {}

    rng = np.random.default_rng(_RNG_SEED)
    jvp_responses = _make_batched_jvp(f_of_u, compute_dtype)
    # offsets[(comp, write, read)] -> set of per-axis offset tuples.
    accum: dict[tuple[str, str, str], set] = {}

    def _probe_state(u_np, lam_np, t_val) -> dict | None:
        """All probes for one state; None when a response is non-finite."""
        u = jnp.asarray(u_np, dtype=compute_dtype)
        lam = jnp.asarray(lam_np, dtype=compute_dtype)
        t = jnp.asarray(t_val, dtype=compute_dtype)
        state_accum: dict[tuple[str, str, str], set] = {}

        def _run_probes(comp, shape, periodic_axes, read_var, d_in, writes, cells):
            col_off = offset_of((comp.name, read_var.name))
            # One tangent per (cell, per-node component): a basis tangent per
            # component makes sign cancellation across components impossible.
            # Tangents are built per chunk so probe memory stays bounded at
            # ``_PROBE_CHUNK * state_size`` on large grids.
            probes = [(int(cell), c) for cell in cells for c in range(d_in)]
            for i in range(0, len(probes), _PROBE_CHUNK):
                sub = probes[i : i + _PROBE_CHUNK]
                tangents = np.zeros((len(sub), state_size))
                for k, (cell, c) in enumerate(sub):
                    tangents[k, col_off + cell * d_in + c] = rng.uniform(0.5, 1.5)
                resp = jvp_responses(u, lam, t, tangents)
                if not np.isfinite(resp).all():
                    return False
                for k, (cell, _c) in enumerate(sub):
                    for write_var, d_out in writes.values():
                        row_off = offset_of((comp.name, write_var.name))
                        r = resp[k, row_off : row_off + write_var.size]
                        nodes = np.unique(np.nonzero(r != 0.0)[0] // d_out)
                        if nodes.size == 0:
                            continue
                        offs = _response_offsets(shape, periodic_axes, cell, nodes)
                        key = (comp.name, write_var.name, read_var.name)
                        state_accum.setdefault(key, set()).update(map(tuple, offs.tolist()))
            return True

        def _measured_reach(comp, read_var, ndim):
            """Per-axis max |offset| seen so far for this read field.

            Includes offsets accumulated at earlier states, in case a
            coefficient vanishes at the current one.
            """
            r = np.zeros(ndim, dtype=np.int64)
            for source in (state_accum, accum):
                for key, offs in source.items():
                    if key[0] == comp.name and key[2] == read_var.name:
                        for o in offs:
                            r = np.maximum(r, np.abs(np.asarray(o)))
            return r

        for comp, shape, periodic_axes, pairs in blocks:
            for read_var, d_in, writes in pairs.values():
                # Interior probe first; then bands near each boundary, sized
                # from the measured reach and deepened until the reach stops
                # growing (a boundary-modified scheme can read farther into
                # the interior than the interior stencil suggests, so the
                # bands follow the evidence to a fixed point instead of
                # trusting a one-shot width guess).
                center = _probe_cells(shape, depths=(0,) * len(shape))
                if not _run_probes(comp, shape, periodic_axes, read_var, d_in, writes, center):
                    return None
                probed = set(center.tolist())
                while True:
                    reach = _measured_reach(comp, read_var, len(shape))
                    depths = tuple(int(v) + 2 for v in reach)
                    band = _probe_cells(shape, depths=depths)
                    new_cells = np.asarray(sorted(set(band.tolist()) - probed), dtype=np.int64)
                    if new_cells.size == 0:
                        break
                    if not _run_probes(
                        comp, shape, periodic_axes, read_var, d_in, writes, new_cells
                    ):
                        return None
                    probed.update(new_cells.tolist())
        return state_accum

    def _draw():
        u_np = rng.uniform(0.5, 1.5, size=state_size)
        lam_np = rng.uniform(0.5, 1.5, size=constraint_dim)
        return u_np, lam_np, float(rng.uniform(0.0, 1.0))

    def _accumulate(u_np, lam_np, t_val) -> bool:
        state_accum = _probe_state(u_np, lam_np, t_val)
        if state_accum is None:
            return False
        for key, offs in state_accum.items():
            accum.setdefault(key, set()).update(offs)
        return True

    clean_states = 0
    for _attempt in range(_MAX_STATE_DRAWS):
        if clean_states >= _N_PROBE_STATES:
            break
        clean_states += _accumulate(*_draw())
    for rising in (True, False):
        u_np, lam_np, t_val = _draw()
        for comp, shape, _periodic_axes, pairs in blocks:
            smooth = _smooth_state(shape, rising)
            for read_var, d_in, _writes in pairs.values():
                start = offset_of((comp.name, read_var.name))
                u_np[start : start + smooth.size * d_in] = np.repeat(smooth, d_in)
        _accumulate(u_np, lam_np, t_val)  # a non-finite smooth state is skipped

    if clean_states == 0:
        comp_names = sorted({comp.name for comp, _s, _p, _pairs in blocks})
        warn_user(
            "stencil inference: the residual evaluated to non-finite values at "
            f"every probe state for component(s) {comp_names}; keeping the "
            "default radius-1 stencil declaration. A body whose true "
            "dependency is wider than one cell per axis will be "
            "under-declared; rewrite the residual to be finite at generic "
            "positive states to enable inference.",
        )
        return {}

    result: dict[tuple[str, str, str], np.ndarray | None] = {}
    for key, offs in accum.items():
        ndim = len(next(iter(offs)))
        star = _star1_offsets(ndim)
        if offs <= star:
            continue  # default radius-1 declaration is already correct.
        if any(abs(int(o)) > _GLOBAL_COUPLING_RADIUS for off in offs for o in off):
            result[key] = None  # global coupling: declare the block dense.
            continue
        result[key] = np.asarray(sorted(offs | star), dtype=np.int64)
    return result
