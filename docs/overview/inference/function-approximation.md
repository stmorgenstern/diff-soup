# Function approximation

This page provides an overview of the pre-built function approximators
in diff-soup. The recipe, data, and inference problem are those of
[fitting parameters](fitting.md).

## An overview of function approximation in diff-soup

During inference, unknown functions can be approximated using both
classical and deep learning approaches, through the built-in
approximators on this page or a custom one. In general, any unknown can
be parameterized as an equinox module whose array fields are the
coefficients and declared with `spec.function`
([named functions](../modeling/parameters.md#named-functions)). For
example, to parameterize a reaction rate \(r(u) = e^{a u}\) with the
learnable coefficient \(a\):

```python
class Exponential(eqx.Module):
    rate: jax.Array

    def __call__(self, u):
        return jnp.exp(self.rate * u)

    def learnable_paths(self):
        return ("rate",)


decay = spec.function("decay", Exponential(jnp.array(0.0)), learnable=True)


@spec.equation(writes="u", reads=("u",))
def reaction_diffusion(q):
    return q.p.D * q.ops.laplacian(q.u) - q.fn.decay(q.u) * q.u
```

- The array fields of the module hold the coefficients, and `__call__`
  evaluates the function.
- `learnable_paths()` names the fields the fit moves. A module without
  it has every array field fitted, the weights and biases of a network
  included.
- `spec.function(name, module, learnable=True)` declares the function,
  and `q.fn.<name>` evaluates it inside an equation. `pointwise=True`
  applies a module that maps one sample at every node
  ([named functions](../modeling/parameters.md#named-functions)).
- The fit selects every declared learnable function by default,
  alongside the learnable parameters
  ([pose the inference problem](fitting.md#pose-the-inference-problem)).

The built-in approximators live in `diff_soup.inference.approx` and
fall into three classes by their argument:

- functions of the state variables, for physical operators and
  constitutive relationships,
- functions of position, for fields on spatial domains, and
- functions of time, for time-varying signals.

## Physical operators and constitutive relationships

For a function of the state variables, the classical approximation is
a truncated expansion in orthogonal polynomials,

$$
f(u) \approx \sum_{k=0}^{K} a_k\, P_k(x), \qquad
x = \frac{2\,(u - u_{\min})}{u_{\max} - u_{\min}} - 1,
$$

where \(P_k\) is the Legendre polynomial of degree \(k\) on
\([-1, 1]\), \(x\) is the state mapped onto that interval, and the
coefficients \(a_k\) are the learnable parameters. Orthogonality keeps
the fit well conditioned as the degree grows, and a transform applied to
the raw expansion builds in known structure, such as positivity through
an exponential.

### Orthogonal polynomials in one dimension

`Legendre(coeffs, interval=(0.0, 1.0), transform=None, clip=None)` is
the learnable Legendre expansion of one state variable.

- `coeffs` holds the coefficients \(a_0, \dots, a_K\), one per degree,
  and is the fitted field.
- `interval=` is the range of the physical input, mapped onto
  \([-1, 1]\). The default is `(0.0, 1.0)`.
- `transform=` is a function `transform(u, p)` of the input `u` and the
  raw expansion `p`, applied to the result. The default returns the raw
  expansion. `lambda u, p: jnp.exp(p)` gives a positive function.
- `clip=(lo, hi)` clips the input to that range before evaluation. The
  default evaluates the input as given.

For example, to fit a diffusivity \(D(u) = \exp\big(\sum_k a_k P_k\big)\)
with three coefficients, starting from the constant \(D = 0.1\):

```python
from diff_soup.inference.approx import Legendre

spec = ds.problem("heat")
spec.domain("rod", rod)
spec.field("u", mass="differential", scale=1.0, init=lambda x: x * (1.0 - x))
coeffs = jnp.array([np.log(0.1), 0.0, 0.0])
D_of_u = spec.function("D_of_u", Legendre(coeffs, interval=(0.0, 0.25), transform=lambda u, p: jnp.exp(p)),
                       learnable=True)


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.div_flux(q.fn.D_of_u(q.u), q.u)


spec.bc.dirichlet("u", node_indices=np.array([0, n - 1]), value=0.0)
problem = spec.build()
y0 = spec.assemble_state({})
infproblem = ds.inference.InferenceProblem(
    ds.inference.Experiment(problem, solver, (0.0, 0.1), y0, (measurement,)), parameters=D_of_u)
result = ds.inference.optsolve(infproblem)
```

The expansion evaluates \(\sum_k a_k P_k(x)\), with \(x\) the state
mapped onto \([-1, 1]\) and the coefficients \(a_k\) the fitted values,
and it broadcasts over the nodal array on its own, so it is declared
without `pointwise=True`. `ds.inference.Tikhonov` weights the
coefficient of degree \(k\) by \(1 + k(k + 1)\) by default, which damps
the high-degree coefficients first
([regularization](fitting.md#regularization)).

The raw expansions, for composing a parameterization by hand:

- `LegendreExpansion(params)` evaluates \(\sum_k a_k P_k(x)\) for an
  input already on \([-1, 1]\).
- `ChebyshevExpansion(params)` evaluates \(\sum_k a_k T_k(x)\), with
  \(T_k\) the Chebyshev polynomials, for an input already on
  \([-1, 1]\).
- `MonomialExpansion(params)` evaluates \(\sum_k a_k u^k\).
- `mono_to_legendre(mono_coeffs, n_leg_terms, offset=0.0)` converts
  monomial coefficients on \([0, 1]\) to Legendre coefficients by a
  least-squares fit.

### Sparse orthogonal polynomials in several dimensions

For a function of several state variables, \(f(u_1, \dots, u_d)\), the
sparse orthogonal polynomial expansion runs over products \(P_{k_1}(x_1) \cdots P_{k_d}(x_d)\) with
the total degree \(k_1 + \dots + k_d \le K\), so the number of terms
grows far more slowly than the full tensor product.
`SparseLegendre(*, intervals, degree, truncation="total", coeffs=None,
transform=None, key=None, init_std=0.0)` is the learnable expansion of
several state variables.

- `intervals=` gives the range of each input, one `(lo, hi)` pair per
  variable, each mapped onto \([-1, 1]\).
- `degree=` is the total degree \(K\).
- `truncation="total"` keeps the products with
  \(k_1 + \dots + k_d \le K\). `"hyperbolic"` keeps those with
  \((k_1 + 1) \cdots (k_d + 1) \le K + 1\), fewer mixed terms at the
  same degree.
- `coeffs=` holds one coefficient per retained product, in the order of
  `basis.indices`, the terms sorted by total degree and then
  lexicographically. The default starts every coefficient at zero, and
  `key=` with `init_std=` draws random starting coefficients.
- `transform=` is a function `transform(u_1, ..., u_d, p)` of the inputs
  and the raw expansion, applied to the result.
- `sparse_legendre_weights(basis)` gives the weight \(1 + k(k + 1)\) of
  each term by its total degree \(k\), for `ds.inference.Tikhonov(weights=)`.

For example, to fit a reaction rate that depends on the concentration and
the temperature, \(r(c, T)\), with total degree two:

```python
from diff_soup.inference.approx import SparseLegendre

rate = spec.function("rate", SparseLegendre(intervals=((0.0, 1.0), (250.0, 350.0)), degree=2,
                                            coeffs=jnp.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0])), learnable=True)


@spec.equation(writes="c", reads=("c", "T"))
def reaction(q):
    return -q.fn.rate(q.c, q.T) * q.c
```

### Neural networks

Modern deep learning methods are supported through equinox modules
declared the same way. For example, to learn a reaction term with a
multilayer perceptron (MLP):

```python
network = eqx.nn.MLP("scalar", "scalar", width_size=4, depth=1, activation=jax.nn.tanh,
                     key=jax.random.PRNGKey(0))
reaction = spec.function("reaction", network, learnable=True, pointwise=True)


@spec.equation(writes="u", reads=("u",))
def reaction_diffusion(q):
    return q.p.D * q.ops.laplacian(q.u) + q.fn.reaction(q.u)
```

- A network maps one sample, so it is declared with `pointwise=True`.
- Every weight and bias is fitted.
- The optimizer choice for a network is covered under
  [optimization solvers](solvers.md#recommended-default).

Three modules build known structure into a network:

- `ICNN(in_size, *, width, depth, key, activation="softplus")` is a
  scalar network convex in every input. `activation=` is `"softplus"` or
  `"relu"`.
- `MonotoneMLP(in_size, *, width, depth, key, monotone_in=None,
  activation="tanh")` is a scalar network with a declared sign of the
  derivative in each input. `monotone_in=` has one entry per input, `+1`
  for increasing, `-1` for decreasing, and `0` for unconstrained. The
  default is increasing in every input.
- `Transform(module, fn)` applies `fn` to the output of `module`. `fn` is
  a function, or one of `"exp"`, `"softplus"`, `"abs"`, `"square"`,
  `"even"`, and `"odd"`, the last two averaging the module at \(x\) and
  \(-x\).

The weights are fitted in every case, and a network of one input takes
the nodal array as it is. For example, a reaction rate that increases
with the state:

```python
from diff_soup.inference.approx import MonotoneMLP

rate = spec.function("rate", MonotoneMLP(1, width=8, depth=1, key=jax.random.PRNGKey(0)), learnable=True)


@spec.equation(writes="u", reads=("u",))
def reaction_diffusion(q):
    return q.p.D * q.ops.laplacian(q.u) - q.fn.rate(q.u) * q.u
```

### Neural operators

A neural operator maps a whole field to a field, so a closure that
depends on the profile as a whole, and not on the value at one point, can
be learned.

- `DeepONet(branch_size, trunk_size, *, width=64, depth=3, key,
  n_basis=None)` reads the field at `branch_size` fixed sensors and
  evaluates the output at coordinates of dimension `trunk_size`,
  `q.fn.G(q.u, q.x)` inside an equation.
- `FNO1d(*, modes, width, depth, key, in_channels=1, out_channels=1)` is
  a Fourier neural operator on a uniform one-dimensional grid,
  `q.fn.K(q.u)` inside an equation, with `modes` the number of Fourier
  modes kept, `width` the hidden channels, and `depth` the number of
  spectral layers. `FNO2d` is the same on a uniform two-dimensional grid,
  with `modes` one integer or a pair.
- A Fourier operator is refused on a domain whose grid is not uniform.

For example, a diffusivity that depends on the whole profile, kept
positive through a transform:

```python
from diff_soup.inference.approx import FNO1d, Transform

K = spec.function("K", Transform(FNO1d(modes=3, width=4, depth=1, key=jax.random.PRNGKey(0)), "softplus"),
                  learnable=True)


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.div_flux(q.fn.K(q.u), q.u)
```

## Function approximation on spatial domains

For a function of position, the classical approximation is a
Karhunen-Loeve (KL) expansion of a Gaussian random field (GRF), a sum
of the leading eigenmodes of the field's covariance kernel weighted by
learnable coefficients, with the number of modes kept setting the
resolution of the field.

### Karhunen-Loeve expansions

`KLBasis.from_domain(domain, *, length_scale, sigma=1.0, sigma_psi0=1.0,
n_modes=20, key=None, init_std=0.0)` builds the expansion on the nodes of
a registered grid or mesh by eigendecomposing the covariance matrix over
them, weighted by the mass of each node.

- `domain` is the grid or mesh the field lives on. The basis records the
  domain, and evaluating it on the nodes of another domain is refused.
- `length_scale` is the correlation length of the field, `sigma=` its
  standard deviation, `sigma_psi0=` the standard deviation of the
  constant mode, and `n_modes=` the number of modes kept. The
  covariance kernel is the squared exponential.
- `key=` and `init_std=` draw random starting coefficients. The
  default starts every coefficient at zero.
- Calling the basis, `q.fn.<name>()`, returns the field at every
  node, and `kl.k()` returns its exponential, the positivity
  transform.
- `learnable_paths()` names the coefficients `Z`, so the precomputed
  modes stay fixed in a fit.
- `KLBasis.from_mesh(points, length_scale, ...)` builds the expansion
  from a coordinate array instead, without the mass weighting.
  `KLBasis.from_covariance_operator(A, M, n_modes=)` builds it from a
  discretized covariance operator `A` and a mass matrix `M`, and
  `KLBasis.from_lumped_covariance(covariance_operator, lumped_mass,
  n_modes=)` from a diagonal mass, with the modes orthonormal in that
  mass.
- `GRFBasis.from_domain(domain, *, M, init_std=0.1, key=None)` is the
  special case of a rectangle, with cosine modes of total degree up to
  `M` in place of the computed eigenmodes.

For example, to fit a diffusivity field on the rod with three modes:

```python
from diff_soup.inference.approx import KLBasis

kl = KLBasis.from_domain(rod, length_scale=0.25, sigma=0.5, n_modes=3)
psi = spec.function("psi", kl, learnable=True)


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.div_flux(jnp.exp(q.fn.psi()), q.u)
```

### Coordinate networks

A coordinate network is a function of position, \(x \mapsto f(x)\), with
a network in place of the expansion. Two are provided, and both take one
coordinate or every node at once.

- `FourierFeatureNet(in_size, *, n_frequencies=8, width=32, depth=2, key,
  out_size=1, activation="tanh", transform=None, bounds=None)` feeds the
  sines and cosines \(\sin(2^k \pi x)\) and \(\cos(2^k \pi x)\),
  \(k = 0, \dots, n - 1\), of each coordinate into a multilayer
  perceptron. The network is fitted and the frequencies stay fixed.
- `SIREN(in_size, *, width=32, depth=3, key, out_size=1, omega_0=30.0,
  transform=None, bounds=None)` is a network with the sine activation
  \(\sin(\omega_0 z)\) in every hidden layer and a linear output.
- `bounds=` gives the range of each coordinate, one `(lo, hi)` pair per
  dimension, mapped onto \([-1, 1]\) before the network.
  `from_domain(domain, ...)` takes the dimension and the bounds from a
  registered grid or mesh.
- `transform=` is a function `transform(x, raw)` of the position and the
  raw output, applied to the result, `lambda x, raw: jnp.exp(raw)` for a
  positive field.

For example, to fit a diffusivity field on the rod:

```python
from diff_soup.inference.approx import FourierFeatureNet

D_x = spec.function("D_x", FourierFeatureNet.from_domain(rod, n_frequencies=2, width=5, depth=1, key=jax.random.PRNGKey(0),
                                                         transform=lambda x, raw: 0.2 * jnp.exp(raw)), learnable=True)


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.ops.div_flux(q.fn.D_x(q.x), q.u)
```

## Time-varying signals

For a function of time, such as a boundary value, a forcing term, or a
control input, a signal \(s(t)\) is built on knots
\(t_0 < t_1 < \dots < t_K\) or on a period \(T\), with the values at
the knots or the harmonic amplitudes as the learnable parameters. Each
basis is an equinox module evaluated inside an equation as
`q.fn.<name>(q.t)`.

- `PiecewiseConstant(knots, values)` is the step function
  \(s(t) = v_k\) between \(t_k\) and \(t_{k+1}\), with one value per
  segment.
- `PiecewiseLinear(knots, values)` interpolates linearly between the
  values at the knots.
- `CubicHermite(knots, values, slopes)` is a cubic on each segment
  matched to the value and the slope at both of its knots, so the
  signal is smooth.
- `Fourier(period, cos_coeffs, sin_coeffs, mean)` is the truncated
  series \(s(t) = \bar s + \sum_{k=1}^{n} c_k \cos(2\pi k t / T) + s_k
  \sin(2\pi k t / T)\), periodic with period \(T\).

The knots and the period define the basis and stay fixed in a fit. A time
outside the knots takes the value at the nearest end, and a time
exactly on an interior knot takes the segment to its left, the segment
a solver step ending there was integrated against. A multi-component
signal carries a trailing axis of size \(m\) on its values.

For example, to fit a forcing term that steps once at \(t = 0.05\):

```python
from diff_soup.inference.approx import PiecewiseConstant

knots = jnp.array([0.0, 0.05, 0.1])
drive = spec.function("drive", PiecewiseConstant(knots, jnp.zeros(2)), learnable=True)


@spec.equation(writes="u", reads=("u",))
def heat(q):
    return q.p.D * q.ops.laplacian(q.u) + q.fn.drive(q.t)
```

A signal also drives a boundary value through the callback form of the
condition ([boundary conditions](../modeling/boundary-conditions.md)),
and is fitted there like any other declared function. A fixed trace of
boundary values, as in
[data assimilation](data-assimilation.md#interpolating-boundary-values),
is passed as `value_fn=` instead. For example, to fit the value at the
left end of the rod:

```python
from diff_soup.inference.approx import PiecewiseLinear

inlet = spec.function("inlet", PiecewiseLinear(knots, jnp.zeros(3)), learnable=True)
spec.bc.dirichlet("u", node_indices=np.array([0]), fn=lambda q, t: q.fn.inlet(t))
```
