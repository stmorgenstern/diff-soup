# Boundary conditions

This page provides an overview of setting boundary conditions with
`spec.bc`. Each declaration names the field it constrains, selects
where on the boundary the condition applies, and states what it
imposes there.

## Selecting the boundary

Every condition takes exactly one selector:

- `on=`: a tag name, a tag expression, a node-index array, or a
  boolean node mask.
- `tag=` and `node_indices=`: single-form spellings of the same
  selector.
- `axis=` with `side=`: one face of a structured grid, `side="low"`
  or `"high"` along that axis.

On grids, boundary locations are selected by node index, as in the
running example's two pinned ends from
[the Recipe API](recipe-api.md):

```python
spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)
```

- One-dimensional grids also name their outer node: `tag="surface"`
  selects the last node, the surface cell of a spherical or
  cylindrical grid.
- On a two- or three-dimensional grid, a flux, Robin or Neumann
  condition acts through one face. Its cells must lie on that face,
  so `axis=2, side="low"` is the clear way to name it. A selection of
  edge or corner cells alone lies on several faces and is refused.
- On a scalar field, a node index pins the field's one value at that
  node.
- On a vector field, several values live at each node, so nodes are
  not indexed by degree of freedom. `dirichlet` takes `component=k`
  to pin only the k-th value at every selected node, and pins all of
  them when `component=` is not given.

On meshes, boundary regions carry named tags from the
[tagging system](domains.md#the-tagging-system), claimed by name or
by a tag expression built from `ds.tags`:

- `tag="rim"` selects every boundary facet named `rim`.
- `ds.tags(...)` expressions combine with `|` (union) and `-`
  (difference), and `ds.tags("a", "b")` is the union of both names.

The bowl of soup from
[domains and discretizations](domains.md#example-a-bowl-of-soup)
loses heat through its rim while the bowl is held at a fixed
temperature. The difference expression removes the nodes the two
regions share from the pinned set, leaving them governed by the
rim's flux:

```python
spec.bc.flux("u", tag="rim", value=-0.1)                                # heat lost through the rim
spec.bc.dirichlet("u", on=ds.tags("bowl") - ds.tags("rim"), value=1.0)  # bowl held at a fixed temperature
```

## The condition types

For a scalar field \(u\) with outward unit normal \(n\):

- `spec.bc.dirichlet(var, on=..., value=...)` fixes the value,
  \(u = g\).
- `spec.bc.flux(var, on=..., value=...)` prescribes the flux density
  of \(u\) through the boundary, \(-n \cdot F = g\) (\(F\) the flux
  of \(u\)), per unit boundary area and positive into the domain.
  For a diffusive flux this is \(D \, \partial u / \partial n = g\).
- `spec.bc.neumann(var, on=..., value=...)` prescribes the outward
  normal derivative itself, \(\partial u / \partial n = s\),
  whatever the diffusivity. It needs `laplacian` or `div_flux`
  applied to the field itself, `q.u`, which supplies \(D\).
- `spec.bc.robin(var, on=..., alpha=..., beta=..., value=...)`
  states the mixed condition
  \(\alpha u + \beta \, n \cdot \nabla u = g\), assembled as a
  boundary flux with the same sign convention as `flux`. A zero
  `beta` states a Dirichlet condition and is refused. `alpha` and
  `beta` may be callables of time.
- `spec.bc.inflow(var, axis=..., side=..., velocity=v, value=c_in)`
  feeds a grid face, the Danckwerts condition
  \(v c - D \, \partial c / \partial n = v c_{in}\). The face carries
  the total flux \(v c_{in}\) into the domain, so it pairs with
  `div_advect`, which carries no flux in through a face.
- `spec.bc.periodic(var, axes=(0,))` wraps the field on the listed
  axes of its cartesian grid, every axis when `axes` is omitted. The
  other faces take any other condition, and every field sharing the
  grid is periodic on the same axes.

For a vector field, such as a displacement or a velocity:

- `spec.bc.traction(var, on=..., value=...)` prescribes the boundary
  traction \(\sigma \cdot n = t\) (\(\sigma\) the stress tensor),
  with `value=` the traction vector \(t\), on a tagged mesh.
- `spec.bc.symmetry(var, on=...)` imposes \(n \cdot u = 0\): the
  field slides along the boundary, never through it. The nodal
  normal is the measure-weighted mean of the facets meeting at each
  node, so a slanted or curved boundary is imposed as faithfully as
  an axis-aligned one, and the tangential rows keep their assembled
  physics. A scalar field is refused.
- `spec.bc.no_slip(var, on=..., wall_velocity=...)` is shorthand for
  a Dirichlet condition at the wall velocity (zero by default), and
  `spec.bc.slip_wall(var, on=...)` for a symmetry condition.

Swapping a condition changes one line. An insulated left end on the
running example's rod replaces the Dirichlet pin with a zero-flux
condition:

```python
spec.bc.flux("u", node_indices=np.array([0]), value=0.0)
spec.bc.dirichlet("u", node_indices=np.array([n - 1]), value=0.0)
```

A packed bed fed through its low-x face takes a velocity field, a
per-axis dispersion and an inflow:

```python
@spec.equation(writes="c", reads=("c",))
def transport(q):
    return -q.ops.div_advect(q.c, (vx, 0.0, 0.0)) + q.ops.div_flux((D_ax, D_r, D_r), q.c)

spec.bc.inflow("c", axis=0, side="low", velocity=v_in, value=c_feed)
```

## Time-dependent and nonlinear conditions

Every condition takes a constant `value=`, a scalar or an array with
one entry per selected location. Two callback forms generalize it,
and both receive dimensional time
([the two clocks](fields.md#the-two-clocks)). A condition takes
exactly one of the three forms.

`value_fn=lambda t: ...` makes the value time-dependent. Every
condition kind accepts it. A `dirichlet` closure may return an array
to prescribe a spatial profile over the selected nodes.

`fn=lambda q, t: ...` makes the value depend on the state, the
parameters, and the declared functions, the form for nonlinear
kinetics such as reaction fluxes and for a fitted boundary value.
Every condition kind accepts it. It applies on a tagged mesh boundary
and on the cells of one grid face. The closure receives `q`, a view
of the problem restricted to the selected facets, and returns a
scalar or one value per facet:

- `q.<field>` is the field's per-facet value. On a mesh it is the
  face-mean of the nodal values. On a grid each selected cell is its
  own facet and the value is the cell's own. Every field of the
  component the boundary belongs to is readable.
- `q.<global>` reads a declared [global](fields.md#globals), and
  `q.control.<name>` a [control](controls.md).
- `q.p.<name>` and `q.fn.<name>` read the current parameter and
  function values, the fitted values during a fit
  ([learnable parameters](parameters.md)).
- `q.nodal.<field>` reads the field's full nodal array, and `q.ops`
  applies the operator bank to those arrays.
- `q.t` is the time, and `q.geom.normal` and `q.geom.x` are the
  per-facet outward unit normal and centroid.
- A name `q` does not carry fails at build, naming the fields,
  globals, and controls that could be declared.

On a `dirichlet` condition the closure's values at a node shared by
several selected facets are averaged before the node is pinned. By
default a `flux` closure sees one facet at a time, so the flux is
local by construction and its Jacobian stencil is exact. Declare
`cross_facet=True` for a flux that combines values across facets,
such as a boundary-wide reduction. The closure then sees the whole
boundary at once and the Jacobian uses the conservative
boundary-wide pattern.

The bowl of soup radiates heat through its rim, a flux nonlinear in
the surface temperature:

```python
eps = 0.05

spec.bc.flux("u", tag="rim", fn=lambda q, t: -eps * q.u**4, name="rim_loss")
```

Passing `name=` gives the flux a stable handle that the
[`flux_total` constraint](constraints.md) links to, so the same flux
density drives both the boundary condition and its integral closure.
Any of the three forms accepts it. The handle requires a tagged
mesh, which supplies the facets the integral runs over.

A [population](equations.md#populations)'s member boundary
conditions are declared through the family handle,
`members.bc.flux(...)`, with the same three forms. Inside its closure
`q.m.<name>` reads the member's entry of a per-member parameter
([populations](equations.md#populations)). A member condition is
applied or refused, never skipped.

## How conditions are enforced

For the built system \(M(u, t)\,du/dt = f\):

- A Dirichlet condition replaces the field's residual on the
  selected rows with \(u - g\) and zeroes the rows' entries of
  \(M\), so pinned rows are algebraic rows
  ([the variable types](fields.md#the-variable-types)). The diffrax
  backend, which takes only differential rows, instead integrates a
  pinned row through the boundary value's time derivative
  ([the diffrax wrapper](../simulation/solvers.md#the-diffrax-wrapper)).
- A flux or traction condition adds its boundary term to the
  residual. On a mesh it is the surface integral that integration
  by parts exposes
  ([the FEM primer](operators.md#a-primer-on-fem-and-the-weak-form)),
  assembled over the selected facets. On a finite-volume grid the
  flux enters the boundary cell scaled by the face area over the
  cell volume.
- Where a Dirichlet condition and a flux condition select the same
  node, the Dirichlet replacement wins.
- A Neumann condition acts through `laplacian` and `div_flux`
  applied to the field itself, `q.u`. On a grid the boundary takes a
  ghost value \(u + h s\) beyond the face, a ghost node on a
  finite-difference grid. On a mesh the operator adds the boundary
  term \(\int D s N_i \, dS\) with its own \(D\).
- A periodic declaration acts through the operator bank. The wrap
  applies to `laplacian`, `div_flux`, `advect_upwind`, and
  `div_advect`.
  `gradient`, `divergence`, `face_flux`, and `curl` difference
  one-sidedly at the axis ends and are not yet supported on a
  periodic field, so they raise. File an issue for support, or
  define a [custom operator](operators.md#custom-operators) with
  the wrap built in.
