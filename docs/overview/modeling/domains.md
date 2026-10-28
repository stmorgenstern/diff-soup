# Domains and discretizations

This page provides an overview of the spatial domains a field can live
on and the discretizations they support. A domain is registered with
`spec.domain(name, domain)`, and its discretization decides how the
[operator bank](operators.md) assembles every spatial operator.

## The discretizations

- Finite differences (`"fdm"`) approximate derivatives with difference
  stencils at grid nodes.
- Finite volumes (`"fvm"`) evolve cell averages through fluxes across
  cell faces, which keeps the scheme conservative.
- Finite elements state the equations in weak form and integrate them
  over the elements of a mesh.

Grids take `method="fvm"` (the default) or `method="fdm"`. Finite
elements come from using a mesh, structured or not.

## The domain types

- Grids store values on a regular lattice of nodes or cells.
- Structured meshes are regular lattices of elements (quads, hexes, or
  split triangles), built by the mesh factories.
- Unstructured meshes are arbitrary arrangements of elements, for
  complex geometries: a polygon is fit to the shape's boundary and then
  meshed.

Because a boundary condition can vary across different regions of the
boundary, we employ a system for tagging boundary locations. On grids a
location is a node index. On meshes it goes through the tagging system:
pieces of the boundary carry names (`"left"`, `"top"`) on a
`TaggedMesh` and are selected with composable `ds.tags` expressions,
covered with [boundary conditions](boundary-conditions.md).

## Structured domains

### Grids

The grid factories describe regular geometries:

- `cartesian_grid_1d(n, x_min, x_max)`, with optional half-cell
  offsets at either end.
- `cartesian_grid_2d(nx, ny, lx, ly)` and
  `cartesian_grid_3d(nx, ny, nz, lx, ly, lz)`.
- `cylindrical_grid_1d(n, r_max, r_min=0.0)` and
  `spherical_grid_1d(n, r_max, r_min=0.0)`. Curvilinear grids carry
  their metric, so axis regularity (the \(r = 0\) cell of a spherical
  grid) needs nothing from you.
- `grid.node_coordinates()` returns the node coordinates as an array of
  shape `(num_nodes, dim)`, in row-major order over `shape` (x-major).
  A field on a grid is a flat array in the same order. `Grid1D.nodes`
  is the same array without the `dim` axis, and `GridND` has no `nodes`.

`method` selects the discretization: `"fvm"` (the default) or `"fdm"`.
Flow and advection problems specialize grids further, described
[at the end of this page](#flow-and-advection).

### Structured meshes

For finite elements, build a structured mesh of the same geometry and
register it as the domain:

- `rectangle_mesh(nx, ny, lx, ly, ele_type="QUAD4",
  with_boundary_tags=True)`, tagged `"left"`, `"right"`, `"bottom"`,
  `"top"`.
- `box_mesh(nx, ny, nz, lx, ly, lz, ele_type="HEX8", ...)`, tagged
  `"x_min"` through `"z_max"`.
- `interval_mesh(n_cells, x_min, x_max)`, the finite-element
  counterpart of `cartesian_grid_1d`: two-node line elements, the
  two end nodes tagged `"left"` and `"right"`.
- `cylindrical_interval_mesh(n_cells, r_max, r_min=0.0)` and
  `spherical_interval_mesh(n_cells, r_max, r_min=0.0)`, radial line
  meshes with the curvilinear metric weighted into the element
  measure, ends tagged `"center"` and `"surface"`. The arguments
  match the radial grid factories, so a grid call site ports to the
  mesh factory unchanged.

Meshes always discretize by finite elements.

## Unstructured meshes

For objects in complex geometries we fit a polygon to the object's
boundary and mesh it with gmsh, which installs with the `mesh` extra
(`pip install "diff-soup[mesh]"`).

On a mesh, equations are discretized by finite elements: the operator
bank assembles each operator's weak form automatically, so the equation
body does not change.

### Specifying a mesh

Every route returns a mesh you register with `spec.domain`:

- From an image: `mask_to_mesh(binary_mask, mesh_size=...)` converts a
  binary mask (non-zero pixels are inside) to a triangular mesh, with
  `mesh_size` setting the target element size.
- From a polygon: `mask_to_polygon(mask)` extracts a simplified
  boundary polygon from a mask (`pixel_size` sets the physical scale,
  `epsilon_factor` how aggressively the contour is simplified), and
  `polygon_to_mesh` or `polygon_to_tagged_mesh` triangulates it.
- Named solids: `cylinder_mesh`, `sphere_mesh`, and `box_mesh_tet`
  generate tetrahedral meshes directly.
- Structured meshes: `rectangle_mesh` and `box_mesh`, from the
  [structured section](#structured-domains).
- External meshes: `read_mesh` loads a mesh produced by another tool,
  `write_mesh` and `write_vtu` write one back, and `refine_uniform`
  refines uniformly.
- Raw connectivity: `Mesh(points, cells, ele_type)` builds from your
  own points and cells.

The element types are `TRI3` and `QUAD4` in two dimensions and `TET4`
and `HEX8` in three, and `order=2` on the polygon route produces
quadratic `TRI6` elements.

### The tagging system

- A tag is an integer naming a group of boundary edges (or faces),
  optionally with a name string attached.
- `polygon_to_tagged_mesh` always adds a global `"boundary"` tag
  covering every edge. `segment_tags=` (one integer per polygon edge)
  with `segment_names={tag: name}` groups edges under your own names.
- The structured factories name their tags `"left"`, `"right"`,
  `"bottom"`, `"top"` (rectangles) and `"x_min"` through `"z_max"`
  (boxes).
- Declarations select tags by name (`tag="rim"`) or by a composable
  expression: `on=ds.tags("boundary") - ds.tags("rim")` is every
  boundary edge except the rim group.

### Example: a bowl of soup

A fake binarized image of a bowl, with the rim and the bowl tagged
separately:

```python
import numpy as np
import diff_soup as ds
from diff_soup.domains.geometry import mask_to_polygon, polygon_to_tagged_mesh

yy, xx = np.mgrid[0:128, 0:128]
bowl_mask = ((xx - 64) ** 2 + (yy - 64) ** 2 < 56 ** 2) & (yy >= 64)

polygon = mask_to_polygon(bowl_mask)
ys = np.array([p[1] for p in polygon])
mids = 0.5 * (ys + np.roll(ys, -1))                    # midpoint height of each edge
tags = np.where(mids < ys.min() + 1.0, 1, 2).tolist()  # rim edges vs bowl edges

mesh = polygon_to_tagged_mesh(
    polygon, mesh_size=4.0, segment_tags=tags, segment_names={1: "rim", 2: "bowl"}
)
```

## Heat equation with alternative domains and discretizations

The heat equation built in [the Recipe API](recipe-api.md) ports
across domains and discretizations by changing the domain line and the
boundary selectors.

With finite volumes, the recipe's original form:

```python
spec = ds.problem("heat")
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm"))
```

With finite differences:

```python
spec = ds.problem("heat")
spec.domain("rod", cartesian_grid_1d(n, 0.0, 1.0, method="fdm"))
```

With finite elements, the domain is a structured mesh and boundaries
are selected by named tags. The equation body does not change: the
finite-element bank assembles `q.ops.laplacian` as the weak form
automatically:

```python
from diff_soup.domains import rectangle_mesh

mesh = rectangle_mesh(64, 64, 1.0, 1.0, ele_type="QUAD4", with_boundary_tags=True)
spec = ds.problem("heat")
spec.domain("sheet", mesh)
spec.field("u", mass="differential", scale=1.0)

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)             # FEM assembles the weak form

spec.bc.dirichlet("u", on=ds.tags("left") | ds.tags("right") | ds.tags("top") | ds.tags("bottom"), value=0.0)
```

On a sphere with finite volumes, the spherical grid carries the radial
metric (the \(r = 0\) cell needs nothing from you) and the surface node
is pinned:

```python
from diff_soup.domains import spherical_grid_1d

spec = ds.problem("heat_sphere")
spec.domain("ball", spherical_grid_1d(n, 1.0, method="fvm"))
spec.field("u", mass="differential", scale=1.0)

@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.laplacian(q.u)

spec.bc.dirichlet("u", node_indices=np.array([n - 1]), value=0.0)
```

On the unstructured bowl mesh from
[the example above](#example-a-bowl-of-soup), again with the same
equation body, and the rim and bowl carrying different boundary
conditions:

```python
spec = ds.problem("soup")
spec.domain("bowl", mesh)
spec.field("u", mass="differential", scale=1.0)

@spec.equation(writes="u", reads=("u",))
def cool(q):
    return q.ops.laplacian(q.u)             # FEM assembles the weak form

spec.bc.flux("u", tag="rim", value=-0.1)                                    # heat lost through the rim
spec.bc.dirichlet("u", on=ds.tags("boundary") - ds.tags("rim"), value=1.0)  # bowl held at a fixed temperature
```

Every domain carries an operator bank that equations reach as `q.ops`.
The bank, its dispatch on the discretization, and custom operator banks
are covered with [operators](operators.md).

## Flow and advection

Transport-dominated problems need upwind discretizations: information
travels with the flow, and a scheme that ignores the flow direction
oscillates.

### The finite-volume method for advection

For systems with advective terms of the form

$$
\frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0,
$$

where \(F\) is the flux of \(u\), we use the finite-volume
discretization

$$
\frac{d u_i}{d t} =
- \frac{F_{i+1/2}\, A_{i+1/2} - F_{i-1/2}\, A_{i-1/2}}{V_i},
$$

which updates the cell average \(u_i\) through the fluxes at its two
faces (\(A\) the face areas, \(V_i\) the cell volume). The face
fluxes \(F_{i \pm 1/2}\) require special consideration: centered
approximations oscillate when transport dominates, so we construct them
with the upwind methods below.

### Upwind advection on ordinary grids

First-order upwind advection is handled at the operator layer, as
`q.ops.advect_upwind(field, velocity)` on any grid, covered with
[operators](operators.md).

A periodic direction is declared with `spec.bc.periodic` ([boundary
conditions](boundary-conditions.md)).

### Staggered grids for flow

For Navier-Stokes the storage layout itself changes:

- `staggered_grid_2d(nx, ny, lx, ly)` and `staggered_grid_3d(...)` use
  the marker-and-cell (MAC) arrangement: scalars such as pressure live
  at cell centers, and each velocity component lives on the cell faces
  normal to its direction, so an x-velocity array has one extra slot
  along x.
- Every staggered array carries its location explicitly
  (`Location.CELL_CENTER`, `Location.FACE_X`, `Location.FACE_Y`,
  `Location.FACE_Z`), and `staggered.face_shape(location)` gives the array shape at a named
  location.
- The underlying cell grid stays available as `staggered.cell_grid`.
