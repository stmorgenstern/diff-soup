"""Function approximation for ``diff_soup.inference``: expansions, fields, signals.

Provides generic, differentiable polynomial expansions, random-field bases,
and time-signal bases that users compose into physics-specific
parameterisations. [Basis][diff_soup.inference._abc.Basis] is the abstract
interface for coefficient-carrying bases; the concrete expansions below are
plain ``eqx.Module`` pytrees with the same call convention.
[legendre_weights][diff_soup.inference.fitting.regularization.legendre_weights]
supplies the degree-graded Tikhonov weights that pair with ``LegendreExpansion``.

* [Legendre][] — Named learnable Legendre function on a physical interval.
* [LegendreExpansion][] — Legendre polynomial basis on [−1, 1].
* [ChebyshevExpansion][] — Chebyshev polynomial basis on [−1, 1].
* [MonomialExpansion][] — Direct power-law basis.
* [mono_to_legendre][] — Convert monomial coefficients to Legendre.
* [diff_soup.inference.approx.grf][] — Gaussian Random Field (GRF)
  parameterisations via 2-D cosine Fourier modes (differentiable) and RBF
  covariance eigenmodes (sampling utility).
* [PiecewiseConstant][], [PiecewiseLinear][], [CubicHermite][],
  [Fourier][] — learnable time-signal bases for boundary conditions,
  forcing terms, and control inputs.

Usage example::

    expansion = LegendreExpansion(params)
    # User composes physics: mu(c) = log(c/(1-c)) + expansion(2c-1)
    mu = lambda c: jnp.log(c/(1-c)) + expansion(2*c - 1)
    # Positive function: D(c) = exp(expansion(2c-1))
    D = lambda c: jnp.exp(expansion(2*c - 1))
    # Boundary-zero: j0(c) = c*(1-c)*exp(expansion(2c-1))
    j0 = lambda c: c*(1-c) * jnp.exp(expansion(2*c - 1))
"""

from diff_soup.inference._abc import Basis
from diff_soup.inference.approx.chebyshev import ChebyshevExpansion
from diff_soup.inference.approx.coordinate_nets import SIREN, WIRE, FourierFeatureNet
from diff_soup.inference.approx.grf import (
    GRFBasis,
    GRFGenerator,
    KLBasis,
    build_grf_basis,
    grf_mode_list,
)
from diff_soup.inference.approx.legendre import Legendre, LegendreExpansion, mono_to_legendre
from diff_soup.inference.approx.networks import ICNN, MonotoneMLP, Transform
from diff_soup.inference.approx.operators import DeepONet, FNO1d, FNO2d
from diff_soup.inference.approx.polynomial import MonomialExpansion
from diff_soup.inference.approx.signals import (
    CubicHermite,
    Fourier,
    PiecewiseConstant,
    PiecewiseLinear,
)
from diff_soup.inference.approx.sparse_legendre import SparseLegendre, sparse_legendre_weights
from diff_soup.inference.fitting.regularization import legendre_weights

__all__ = [
    # Abstract basis interface
    "Basis",
    "FourierFeatureNet",
    "SIREN",
    "WIRE",
    "ICNN",
    "MonotoneMLP",
    "Transform",
    "DeepONet",
    "FNO1d",
    "FNO2d",
    # Core generic expansions
    "Legendre",
    "LegendreExpansion",
    "ChebyshevExpansion",
    "MonomialExpansion",
    "mono_to_legendre",
    "legendre_weights",
    "SparseLegendre",
    "sparse_legendre_weights",
    # GRF basis
    "GRFBasis",
    "GRFGenerator",
    "KLBasis",
    "build_grf_basis",
    "grf_mode_list",
    # Time-signal bases
    "PiecewiseConstant",
    "PiecewiseLinear",
    "CubicHermite",
    "Fourier",
]


def __getattr__(name: str):
    """Direct callers of the removed neural wrapper to Equinox's MLP."""
    if name == "FluxNet":
        raise TypeError(
            "FluxNet has been removed; use eqx.nn.MLP with callable activations "
            "and spec.function(..., learnable=True, pointwise=True) for nodal inputs"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
