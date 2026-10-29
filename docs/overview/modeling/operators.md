# Operators

This page provides an overview of the spatial operators available to
equation closures. Each registered domain carries an operator bank,
reached inside a closure as `q.ops`.

## The operator bank

Operators are diff-soup's language for expressing local differential
operators, the terms through which the spatial nodes of a field
become coupled. Each operator carries the sparsity pattern of that
coupling, which the lowering composes into the Jacobian's sparse
structure.

An operator's contract:

- It takes a field's discrete values and returns the discretized
  term, shape-compatible with its input. On a grid the return is the
  term's pointwise value, one per cell or node. On a mesh it is the
  term's weak form, the term integrated against each finite-element
  basis function.
- It is pure and safe under `jit` and automatic differentiation.
- It is defined per domain: each registered grid or mesh carries its
  own bank bound to that geometry, and the discretization decides how
  each operator is realized.

## Operators on grids

A grid carries its discretization from construction: `method="fvm"`
realizes each operator as a finite-volume flux divergence over the
cells, and `method="fdm"` as a finite-difference stencil at the
nodes, with the same call routing to the grid's realization. We
generally recommend FVM: for conservation laws it is the more
natural discretization.

The diffusion and transport terms, on cartesian grids in one, two,
and three dimensions and on the curvilinear one-dimensional grids
with their metric terms included:

- `laplacian(u)`: \(\nabla^2 u\).
- `div_flux(D, u)`: \(\nabla \cdot (D \nabla u)\). A tuple
  `(Dx, Dy, Dz)` gives each axis its own diffusivity, a diagonal
  dispersion tensor, each entry a scalar or one value per cell.
- `divergence(v)`: \(\nabla \cdot v\) for a vector of per-node
  values.
- `advect_upwind(u, velocity)`: the first-order upwind
  \(\mathbf{v} \cdot \nabla u\), with the sign of the velocity
  choosing the upwind side (a backward difference for \(v \ge 0\),
  forward for \(v < 0\)). The velocity is a constant, or a tuple
  with one entry per axis, each a constant or one value per cell.
  The return is the operator itself, so a transport equation writes
  the minus sign: `-q.ops.advect_upwind(q.u, v)`.
- `div_advect(u, velocity, scheme="tvd")`: the conservative form
  \(\nabla \cdot (\mathbf{v} u)\), with the same velocity forms. Each
  interior face carries the face velocity times a face value taken
  from the side the flow comes from, chosen by `scheme` ([advection
  schemes](#advection-schemes)). A boundary face carries only outflow,
  so an inlet is fed by [`spec.bc.inflow`](boundary-conditions.md).
- `curl(u)`: \(\nabla \times u\) of a vector field with one
  component per axis, in two and three dimensions. In two the curl
  is the scalar \(\partial u_y/\partial x - \partial u_x/\partial y\),
  in three the curl vector, pointwise at the cells.

On one-dimensional grids:

- `gradient(u)`: \(\nabla u\).
- `face_flux(u, D)`: the diffusive flux \(-D\,\partial u/\partial x\)
  at the cell faces, one value per face. These are the raw fluxes
  that `div_flux` differences into the per-cell conservation term. A
  prescribed end flux is a boundary condition, declared with
  [`spec.bc.flux`](boundary-conditions.md).

On any grid:

- `mass(u)`: applies the mass matrix, the identity on a grid and the
  consistent-mass integral on a mesh. It is primarily a helper for
  custom residuals that apply the mass weighting explicitly.
- `lumped_mass()`: the integration weights, the cell volumes on a
  grid. `jnp.sum(q.ops.lumped_mass() * f)` integrates a field over
  the domain, the typical use in an integral observable.

<br>

On a grid, operators compose with the
[pointwise interpretation](equations.md#pointwise-interpretation) in
both directions: a linear operator applies to a nonlinear pointwise
expression, and a nonlinearity applies to an operator's result:

```python
q.ops.laplacian(q.u ** 2)          # operator of a pointwise nonlinearity
jnp.tanh(q.ops.gradient(q.u))      # pointwise nonlinearity of an operator
```

### Advection schemes

`div_advect` takes four face values. The cell Péclet number
\(u h / D\) compares transport with dispersion across one cell.

| `scheme` | Face value | Order | Use it for |
|---|---|---|---|
| `"tvd"` (default) | the upstream value plus a limited share of the change to the downstream one | 2 where smooth, 1 at extrema | any cell Péclet number: it creates no new extrema |
| `"weno3"` | the WENO blend of two two-cell reconstructions | 3 where smooth | smooth fields on coarse grids, at a costlier Jacobian |
| `"central"` | the mean of \(v u\) in the two cells | 2 | cell Péclet numbers below 2 everywhere; above 2 it oscillates |
| `"upwind"` | the upstream value | 1 | the most robust form; it adds a numerical dispersion \(u h / 2\) |

`"tvd"` is a MUSCL scheme with the flux limiter `limiter="van_leer"`
(the default), `"koren"` or `"minmod"` (van Leer, 1974, 1979; Koren,
1993). The van Leer limiter is smooth in the ratio of successive
differences. Koren's and minmod switch between linear pieces, which
slows Newton's method: on a 64 by 16 by 16 packed bed, Koren's took
twice the steps of van Leer's and five times the factorizations. `"weno3"` is the third-order WENO reconstruction of Jiang and
Shu (1996), with the smoothness guard \(10^{-10}\) where they used
\(10^{-6}\), so the weights also see jumps in small fields.
Both are upwind-biased by the sign of the face velocity and read two
cells upstream. A face next to a boundary cell lacks the second cell,
so both drop to the upstream value there, which is first order.

The wider stencil reaches the Jacobian: the build widens the sparse
pattern on its own, and a [family](equations.md#populations) member kernel
declares `stencil_radius=2`. On a 64 by 16 by 16 grid one sparse
factorization costs about four times central's on a CPU and a fifth
more on a GPU.

## A primer on FEM and the weak form

Meshes are discretized with the finite element method (FEM). FEM
represents a field through its nodal values,

$$
u(x) = \sum_j u_j \, N_j(x),
$$

where the basis functions \(N_j\) interpolate between the nodes,
each nonzero only on the elements touching its node.

Consider a generic conservation equation in strong form, with
advection at a velocity \(v\), a diffusive flux \(F\), and a
volumetric reaction \(r(u)\):

$$
\frac{\partial u}{\partial t} + \nabla \cdot (u\,v)
= -\nabla \cdot F + r(u) \quad \text{in } \Omega,
$$

with a surface reaction \(s(u)\) setting the total flux through the
boundary:

$$
n \cdot (u\,v + F) = s(u) \quad \text{on } \partial\Omega.
$$

The strong form holds pointwise. The weak form instead requires the
equation to hold in an inner product with every test function: the
equation is multiplied by a test function and integrated over the
domain. The Galerkin choice takes the test functions to be the basis
functions themselves, giving one equation per \(N_i\),

$$
\int_\Omega \frac{\partial u}{\partial t}\, N_i \, d\Omega
+ \int_\Omega \nabla \cdot (u\,v + F)\, N_i \, d\Omega
= \int_\Omega r(u)\, N_i \, d\Omega,
$$

then integrate the divergence term by parts:

$$
\int_\Omega \nabla \cdot (u\,v + F)\, N_i \, d\Omega
= \oint_{\partial\Omega} s(u)\, N_i \, d\Gamma
- \int_\Omega (u\,v + F) \cdot \nabla N_i \, d\Omega.
$$

Integration by parts naturally exposes the flux boundary condition.

Substituting \(u = \sum_j u_j N_j\) gives the final semi-discrete
system, one equation per basis function:

$$
\sum_j M_{ij} \, \frac{du_j}{dt}
= \int_\Omega (u\,v + F) \cdot \nabla N_i \, d\Omega
- \oint_{\partial\Omega} s(u)\, N_i \, d\Gamma
+ \int_\Omega r(u)\, N_i \, d\Omega,
\qquad
M_{ij} = \int_\Omega N_i N_j \, d\Omega.
$$

The integrals on the right are evaluated by quadrature, numerical
integration at sampling points inside each element. The mesh
operators below assemble exactly these integrals.

## FEM nodal values and the weak form

A finite-element field can be represented by its nodal values
\(u_j\), the unknowns in the state vector, and by its quadrature
samples, its values at the quadrature points. In diff-soup, we
default to representing every field in its quadrature-native
representation (the `quad` option of
[`spec.field`](fields.md#declaring-a-field)) in order to be friendly
to nonlinear compositions. To work with nodal values, read
`q.<name>.nodal`. Operators accept either form and consume the nodal
values upon lowering. With this approach, equations can be authored
in strong form and lowered to the weak form at build time.

For weak forms the built-in operators do not cover, three helpers
assemble the primer's integrals directly:

- `q.grad(field)`: the gradient at the quadrature points.
- `q.flux(expr)`: assembles the flux integral
  \(\int \text{expr} \cdot \nabla N_i\).
- `q.source(expr)`: assembles the volumetric integral
  \(\int \text{expr}\, N_i\).

## Operators on meshes

Mesh operators return the weak form's integrated terms where grid
operators return pointwise values.

### Core differential operators

The differential operators cover scalar, vector, and tensor fields:

- `laplacian(u)`: the weak form of \(\nabla^2 u\), integration by
  parts included.
- `div_flux(D, u)`: the weak form of \(\nabla \cdot (D \nabla u)\).
- `grad(u)`: the weak form of \(\nabla u\), for scalar and vector
  fields.
- `div(T)`: the weak form of \(\nabla \cdot T\), for vector and
  tensor fields.
- `curl(u)`: the weak form of \(\nabla \times u\), the scalar curl
  in two dimensions and the curl vector in three.

### Pointwise helpers

Pointwise tensor algebra works at the quadrature samples, on every
discretization:

- `sym(T)` and `skew(T)`: the symmetric and antisymmetric parts.
- `double_dot(A, B)`: the double contraction \(A : B\), including
  the rank-four stiffness case \(C : \varepsilon\).
- `trace(T)`, `det(T)`, `inv(T)`, `transpose(T)`.
- `outer(u, v)` and `tensor_contract(A, B, axes=1)`.
- `scale(scalar_field, tensor)`: a per-point scalar times a constant
  tensor.

The differential operators and pointwise helpers together compose a
mechanics residual:

```python
eps = q.ops.sym(q.grad(q.u))          # strain
sigma = q.ops.double_dot(C4, eps)     # stress
return q.ops.div(sigma)               # momentum residual
```

### Application-specific operators

- Solids: `strain(u)`, the small strain
  \(\operatorname{sym}(\nabla u)\), and
  `linear_elastic_stress(u, E=..., nu=...)`, its linear-elastic
  stress, plus `deformation_gradient(gradu)`
  and `cauchy_stress_from_S_F(S, F)` for finite-strain terms.
- Fluids: `strain_rate(u)` and `viscous_stress_newtonian(u, mu=...)`
  for a Newtonian fluid of viscosity `mu`. For rotation, `curl` is
  the one operator. On a mesh its pointwise reading is the weak
  curl divided by the lumped mass.
- Porous media: `tortuosity_corrected(D, eps, beta=1.5)`, the
  Bruggeman-corrected diffusivity \(D\,\varepsilon^{\beta}\).
- Quadrature machinery: `interp_to_quad(u)` samples a nodal field at
  the quadrature points, and `integrate(expr)` is the weak integral
  \(\int \text{expr} \, N_i\) of a nodal or quadrature expression.

### Boundary operators

On tagged triangle and tetrahedral meshes,
`boundary_integral(field, tag=...)` integrates a field over a tagged
boundary, `boundary_trace(field, tag=...)` restricts it to one, and
`surface_flux(field, tag=..., kinetics=...)` assembles a
kinetics-driven surface flux. Boundary conditions themselves are
declared with `spec.bc`, covered in
[boundary conditions](boundary-conditions.md).

## Custom operators

Operators are methods of the domain's bank, the object a closure
reaches as `q.ops`. A bank holds its bound `domain`, reports a
`discretization_type` tag, and offers exactly the operators it
implements. `provides` reports that set, and calling an operator
outside it raises an error naming the operator and what requires it.
Every method is pure and safe under `jit` and automatic
differentiation, and returns per the bank's convention, pointwise
values on a grid and weak forms on a mesh.

If a required operator is not currently defined in diff-soup, a
custom operator can be defined using two methods.

### Extending the built-in bank

Subclass the concrete `Operators`, which keeps every built-in method,
add the new operator, and register the class when the domain is
declared with `spec.domain(name, domain, operators=...)`.

For example, to implement the curl: for a two-component field
\(u = (u_x, u_y)\) in two dimensions the curl is the scalar

$$
\nabla \times u
= \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}.
$$

On a grid the bank's convention is the pointwise value, differenced
over the grid spacings:

```python
class CurlOnGrid(Operators):
    def scalar_curl(self, u):
        nx, ny = self.domain.shape
        dx = self.domain.axes[0].node_distances[0].astype(u.dtype)
        dy = self.domain.axes[1].node_distances[0].astype(u.dtype)
        ux = u[:, 0].reshape(nx, ny)
        uy = u[:, 1].reshape(nx, ny)
        duy_dx = jnp.gradient(uy, dx, axis=0)
        dux_dy = jnp.gradient(ux, dy, axis=1)
        return (duy_dx - dux_dy).reshape(-1)
```

An operator body is traced, so geometry such as `node_distances` arrives
as a traced float64 array. Compute on it with `jnp` arithmetic, never
with `float()`, `int()`, `.item()` or a Python `if`.
`x.astype(u.dtype)` keeps the arithmetic in float32 under
`Precision.fp32()`.

On a mesh the convention is the integrated weak term
\(\int (\nabla \times u)\, N_i \, d\Omega\), assembled from the
quadrature-point gradient. List the operator in `weak_form` because
it returns an already-integrated term. A subclass adds only its own
names, and the sets union across the class hierarchy:

```python
class CurlOnMesh(Operators):
    weak_form = frozenset({"scalar_curl"})

    def scalar_curl(self, u):
        H = self.quad_gradient(u)                       # H[..., i, j] = du_i/dx_j
        return self.integrate_shape(H[..., 1, 0] - H[..., 0, 1])
```

Registered on the domain, either bank's operator is called in a
closure as `q.ops.scalar_curl(q.u)`.

### A bank for a novel discretization

To extend the domain and discretization types, `AbstractOperator`
can be subclassed to build a custom domain and operator bank.
Implement the operators the bank offers, hold the bound `domain`,
and report a `discretization_type` tag.

For example, a Fourier spectral bank on a uniform periodic
one-dimensional grid. The Laplacian is diagonal in Fourier space, a
multiplication by \(-k^2\):

```python
import jax.numpy as jnp
from diff_soup.domains import AbstractOperator

class FourierSpectral(AbstractOperator):
    @property
    def discretization_type(self):
        return "fourier"

    def laplacian(self, u, *, bcs=None, t=None):
        dx = self.domain.node_distances[0]
        k = 2 * jnp.pi * jnp.fft.fftfreq(u.shape[0], d=dx)
        return jnp.fft.ifft(-(k**2) * jnp.fft.fft(u)).real
```

Sparsity needs no declaration when extending the built-in bank. On
grids the lowering probes the assembled residual at build time and
widens the stencil pattern to the coupling the operators exercise.
On meshes the pattern is the element connectivity.

## References

- G.-S. Jiang and C.-W. Shu. Efficient implementation of weighted ENO
  schemes. Journal of Computational Physics, 126(1):202–228, 1996.
  [doi:10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130).
- B. Koren. A robust upwind discretization method for advection,
  diffusion and source terms. In C. B. Vreugdenhil and B. Koren,
  editors, Numerical Methods for Advection-Diffusion Problems,
  pages 117–138. Vieweg, 1993.
- B. van Leer. Towards the ultimate conservative difference scheme. II.
  Monotonicity and conservation combined in a second-order scheme.
  Journal of Computational Physics, 14(4):361–370, 1974.
  [doi:10.1016/0021-9991(74)90019-9](https://doi.org/10.1016/0021-9991(74)90019-9).
- B. van Leer. Towards the ultimate conservative difference scheme. V.
  A second-order sequel to Godunov's method. Journal of Computational
  Physics, 32(1):101–136, 1979.
  [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1).
