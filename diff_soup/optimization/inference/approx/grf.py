"""grf.py — Gaussian Random Field function approximation.

Two complementary tools:

* [GRFBasis][diff_soup.inference.approx.grf.GRFBasis] — differentiable parameterisation of log k(x,y) as a
  linear combination of 2-D cosine Fourier modes.  The ``coeffs`` field is a
  ``jax.Array`` (learnable via ``eqx.filter_grad``).  The precomputed basis
  matrix is stored as a NumPy array so it is treated as a constant during
  differentiation.

* [GRFGenerator][diff_soup.inference.approx.grf.GRFGenerator] — NumPy/SciPy utility that draws spatially correlated
  random fields from an RBF-covariance Gaussian prior.  Use this to initialise
  or sample GRF realisations (not differentiable).

Fourier-cosine parameterisation
--------------------------------
log k(x, y) = Σ_{m,n}  a_{mn} · cos(m π x / Lx) · cos(n π y / Ly)
k(x, y)     = exp(log k)          — always positive

The mode set is {(m,n) : m + n ≤ M}, giving (M+1)(M+2)/2 modes.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate
import scipy.linalg
from scipy.spatial.distance import cdist


def _domain_data(domain):
    """Extract coordinates, quadrature weights, and a content identity."""
    from diff_soup.problem._ir.content_hash import _domain_id

    points = np.asarray(domain.node_coordinates(), dtype=float)
    mass = getattr(domain, "cell_volumes", None)
    if mass is None:
        from diff_soup.domains.operators import Operators

        mass = Operators(domain).lumped_mass()
    fingerprint = _domain_id(domain)
    return points, np.asarray(mass, dtype=float), fingerprint


def _check_domain_points(basis, coordinates):
    """Reject evaluation arrays incompatible with a domain-bound basis."""
    if coordinates is not None and basis.domain_fingerprint is not None:
        shape = np.shape(coordinates)
        size = shape[0] if shape else 1
        if size != basis.num_points:
            raise ValueError(
                f"basis for domain {basis.domain_name} ({basis.domain_fingerprint}) "
                f"has {basis.num_points} points, but coordinates have {size}; "
                "evaluate on the original domain's coordinate array"
            )


# ---------------------------------------------------------------------------
# Fourier-cosine utilities
# ---------------------------------------------------------------------------


def grf_mode_list(M: int) -> list[tuple[int, int]]:
    """Return 2-D cosine mode pairs (m, n) with m + n ≤ M, in lexicographic order."""
    return [(m, n) for m in range(M + 1) for n in range(M + 1 - m)]


def build_grf_basis(
    pts: np.ndarray,
    Lx: float,
    Ly: float,
    M: int,
) -> np.ndarray:
    """Precompute the Fourier-cosine basis matrix.

    Args:
        pts: Node coordinates, shape ``(n_pts, 2)``.
        Lx: Domain extent along x used for mode normalisation.
        Ly: Domain extent along y used for mode normalisation.
        M: Maximum total mode order (m + n ≤ M).

    Returns:
        numpy.ndarray
            Shape ``(n_pts, n_modes)`` where ``n_modes = len(grf_mode_list(M))``.
            ``basis[i, k] = cos(m_k π x_i / Lx) · cos(n_k π y_i / Ly)``.
    """
    modes = grf_mode_list(M)
    x, y = pts[:, 0], pts[:, 1]
    return np.stack(
        [np.cos(m * np.pi * x / Lx) * np.cos(n * np.pi * y / Ly) for m, n in modes],
        axis=1,
    )


# ---------------------------------------------------------------------------
# GRFBasis — JAX differentiable
# ---------------------------------------------------------------------------


class GRFBasis(eqx.Module):
    """Differentiable GRF parameterisation via 2-D cosine Fourier modes.

    This is an ``eqx.Module`` pytree: ``coeffs`` is a
    ``jax.Array`` leaf (learnable), ``basis`` is a **static** NumPy field
    (excluded from ``filter_grad`` and from the pytree leaves).  This means the
    whole ``GRFBasis`` can be passed through ``eqx.filter_jit`` / ``vmap`` as one
    object without silently freezing ``coeffs`` as a constant.

    Standalone usage::

        grf = GRFBasis.from_mesh(pts, Lx, Ly, M=4)
        log_k = grf.log_k()   # jax.Array (n_pts,)
        k     = grf.k()       # exp(log_k)

    Attributes:
        coeffs: ``jax.Array`` of shape ``(n_modes,)`` — the learnable leaf.
        basis: Precomputed cosine basis matrix ``(n_pts, n_modes)`` — a plain
            NumPy field.  eqx excludes NumPy arrays from ``filter_grad`` by type
            (``is_inexact_array`` is False for NumPy), so ``basis`` is never
            differentiated; it is a non-static field (NOT ``eqx.field(static=True)``
            — equinox warns "JAX array set as static", promoted to error by the
            test suite).
    """

    coeffs: jax.Array
    basis: np.ndarray
    domain_fingerprint: Optional[str] = eqx.field(static=True, default=None)
    domain_name: Optional[str] = eqx.field(static=True, default=None)

    @property
    def num_points(self) -> int:
        """Number of points on which the cosine basis was constructed."""
        return self.basis.shape[0]

    @classmethod
    def from_domain(cls, domain, *, M, init_std=0.1, key=None):
        """Construct total-degree cosine modes using the domain's physical bounds.

        Coordinates are shifted to each lower bound. One coefficient is
        retained for every nonnegative multi-index of total degree at most M.
        The geometry remains fixed when coefficients are fitted.
        """
        import itertools

        if not isinstance(M, int) or M < 0:
            raise ValueError("GRFBasis M must be a nonnegative integer")
        points, _, fingerprint = _domain_data(domain)
        bounds = np.asarray(domain.bounds)
        extent = bounds[:, 1] - bounds[:, 0]
        if np.any(extent <= 0):
            raise ValueError("GRFBasis requires positive extent in every domain dimension")
        scaled = (points - bounds[:, 0]) / extent
        indices = [k for k in itertools.product(range(M + 1), repeat=domain.dim) if sum(k) <= M]
        basis = np.stack([np.prod(np.cos(np.pi * scaled * k), axis=1) for k in indices], axis=1)
        coeffs = (
            jnp.zeros(len(indices))
            if key is None
            else init_std * jax.random.normal(key, (len(indices),))
        )
        return cls(coeffs, basis, domain_fingerprint=fingerprint, domain_name=type(domain).__name__)

    @classmethod
    def from_mesh(
        cls,
        pts: np.ndarray,
        Lx: float,
        Ly: float,
        M: int,
        *,
        init_std: float = 0.1,
        key: Optional[jax.Array] = None,
    ) -> "GRFBasis":
        """Construct from mesh points with randomly initialised coefficients.

        Args:
            pts: Node coordinates ``(n_pts, 2)``.
            Lx: Domain extent along x.
            Ly: Domain extent along y.
            M: Maximum total mode order.
            init_std: Standard deviation for the initial coefficient values.
            key: JAX PRNG key.  If ``None``, coefficients are initialised to zero.
        """
        basis = build_grf_basis(pts, Lx, Ly, M)
        n_modes = basis.shape[1]
        if key is None:
            coeffs = jnp.zeros(n_modes, dtype=jnp.float64)
        else:
            coeffs = init_std * jax.random.normal(key, (n_modes,), dtype=jnp.float64)
        return cls(coeffs=coeffs, basis=basis)

    def log_k(self):
        """Log-k field at all mesh nodes, shape ``(n_pts,)``."""
        return jnp.asarray(self.basis) @ self.coeffs

    def k(self):
        """Positive field ``k = exp(log k)`` at all mesh nodes."""
        return jnp.exp(self.log_k())

    def __call__(self, coordinates=None):
        """Evaluate the log-field, optionally checking a domain coordinate array."""
        _check_domain_points(self, coordinates)
        return self.log_k()

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the fitted coefficient path while keeping the basis fixed."""
        return ("coeffs",)


# ---------------------------------------------------------------------------
# GRFGenerator — NumPy/SciPy sampling utility
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KLBasis — Karhunen-Loève expansion for arbitrary geometries
# ---------------------------------------------------------------------------


def _symmetrize_checked(matrix: np.ndarray, name: str, *, rel_tol: float = 1e-8) -> np.ndarray:
    """Return ``(X + Xᵀ)/2`` after verifying X is square and nearly symmetric.

    Round-off-level asymmetry (relative to the largest entry) is silently
    symmetrized; asymmetry above ``rel_tol`` raises ``ValueError``.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix {name} must be a square 2-D matrix; got shape {matrix.shape}.")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    rel_asym = float(np.max(np.abs(matrix - matrix.T))) / scale
    if rel_asym > rel_tol:
        raise ValueError(
            f"matrix {name} must be symmetric: relative asymmetry "
            f"{rel_asym:.3e} exceeds the tolerance {rel_tol:.0e}; pass a "
            f"symmetric matrix (e.g. (X + X.T) / 2)."
        )
    return 0.5 * (matrix + matrix.T)


class KLBasis(eqx.Module):
    """Differentiable spatial heterogeneity via Karhunen-Loève expansion.

    Represents a log-field ψ(x) = σ_ψ₀ Z₀ + Σᵢ √λᵢ φᵢ(x) Zᵢ where
    (λᵢ, φᵢ) are eigenpairs of the RBF covariance kernel.

    Unlike [GRFBasis][diff_soup.inference.approx.grf.GRFBasis] (cosine modes on rectangles), this works on
    arbitrary unstructured meshes by eigendecomposing the covariance matrix.

    ``Z`` is the coefficient vector and the only fitted quantity;
    [learnable_paths][diff_soup.inference.approx.grf.KLBasis.learnable_paths]
    declares that, so
    ``spec.function("psi", kl, learnable=True)`` selects the coefficients alone
    and never the precomputed mode arrays.  The class is callable — ``kl()``
    returns [psi][diff_soup.inference.approx.grf.KLBasis.psi] — so it can be
    declared directly as a named function
    and evaluated in an equation as ``q.fn.psi()``.

    Typical usage::

        kl = KLBasis.from_mesh(pts, length_scale=1.0, sigma=1.0, n_modes=20)
        psi = kl.psi()    # (n_pts,) log-heterogeneity field
        k   = kl.k()      # (n_pts,) positive heterogeneity exp(psi)

    The precomputed arrays (``basis``, ``modes``, ``mass_matrix`` or
    ``mass_diagonal``,
    ``eigenvalues``) are host-side NumPy metadata rather than JAX arrays, so
    no mode matrix is carried as a traced operand through ``eqx.filter_jit``
    or donated across a device boundary.  They are ordinary (non-static)
    fields because equinox warns when any array — NumPy or JAX — is placed in
    a static field; the declared learnable set, not the field's type, is what
    keeps them out of a fit.

    Attributes:
        Z: Coefficients ``(Z_0, Z_1, …, Z_{N_KL})``, shape ``(n_modes+1,)`` — the
            fitted leaf.  ``Z[0]`` is the constant-mode weight; ``Z[1:]`` weight
            the KL eigenmodes.
        basis: Precomputed ``√λᵢ φᵢ(x)`` at mesh nodes, shape
            ``(n_pts, n_modes)``.
        sigma_psi0: Prior standard deviation of the constant mode (static float).
        modes: Mass-orthonormal eigenmodes Φ (columns), shape
            ``(n_pts, n_modes)``, satisfying ``Φᵀ M Φ = I``.  Set by
            [from_covariance_operator][diff_soup.inference.approx.grf.KLBasis.from_covariance_operator]; ``None`` for the RBF
            constructors.
        mass_matrix: The mass (metric) matrix M of the generalized
            eigenproblem.  Set by [from_covariance_operator][diff_soup.inference.approx.grf.KLBasis.from_covariance_operator]; ``None``
            otherwise.
        mass_diagonal: A lumped diagonal metric stored in O(N) memory. Set by
            [from_lumped_covariance][diff_soup.inference.approx.grf.KLBasis.from_lumped_covariance];
            mutually exclusive with ``mass_matrix``.
        eigenvalues: Leading eigenvalues λ in descending order, shape
            ``(n_modes,)``.  Set by
            [from_covariance_operator][diff_soup.inference.approx.grf.KLBasis.from_covariance_operator];
            ``None`` otherwise.
    """

    Z: jax.Array
    basis: np.ndarray
    sigma_psi0: float = eqx.field(static=True, default=1.0)
    modes: Optional[np.ndarray] = None
    mass_matrix: Optional[np.ndarray] = None
    mass_diagonal: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    domain_fingerprint: Optional[str] = eqx.field(static=True, default=None)
    domain_name: Optional[str] = eqx.field(static=True, default=None)

    @property
    def num_points(self) -> int:
        """Number of points on which the KL modes were constructed."""
        return self.basis.shape[0]

    @classmethod
    def from_domain(
        cls, domain, *, length_scale, sigma=1.0, sigma_psi0=1.0, n_modes=20, key=None, init_std=0.0
    ):
        """Construct RBF covariance modes weighted by the domain's mass metric.

        The domain supplies its node coordinates and cell volumes or lumped
        nodal mass. Modes solve C M v = lambda v and are M-orthonormal.
        Only Z is fitted; the geometry, metric, and modes stay fixed.
        """
        if length_scale <= 0 or sigma < 0 or not isinstance(n_modes, int) or n_modes < 1:
            raise ValueError(
                "KLBasis requires length_scale > 0, sigma >= 0, and positive integer n_modes"
            )
        points, mass, fingerprint = _domain_data(domain)
        covariance = sigma**2 * np.exp(-(cdist(points, points) ** 2) / (2 * length_scale**2))
        basis = cls.from_lumped_covariance(
            covariance, mass, n_modes=min(n_modes, len(points)), sigma_psi0=sigma_psi0
        )
        if key is not None:
            basis = eqx.tree_at(
                lambda b: b.Z, basis, init_std * jax.random.normal(key, basis.Z.shape)
            )
        return cls(
            Z=basis.Z,
            basis=basis.basis,
            sigma_psi0=basis.sigma_psi0,
            modes=basis.modes,
            mass_diagonal=basis.mass_diagonal,
            eigenvalues=basis.eigenvalues,
            domain_fingerprint=fingerprint,
            domain_name=type(domain).__name__,
        )

    @classmethod
    def from_mesh(
        cls,
        mesh_points: np.ndarray,
        length_scale: float,
        sigma: float = 1.0,
        sigma_psi0: float = 1.0,
        n_modes: int = 20,
        *,
        key: Optional[jax.Array] = None,
        init_std: float = 0.0,
    ) -> "KLBasis":
        """Construct KL basis from mesh by eigendecomposing the RBF covariance.

        Args:
            mesh_points: Node coordinates, shape ``(n_pts, 2)`` or ``(n_pts, 3)``.
            length_scale: RBF covariance correlation length (same units as coordinates).
            sigma: Marginal standard deviation of the spatial field (σ_ψ).
            sigma_psi0: Prior standard deviation of the constant (mean) mode.
            n_modes: Number of KL eigenmodes to retain (excluding constant mode).
            key: JAX PRNG key for random initialisation.  ``None`` → zeros.
            init_std: Standard deviation for initial coefficient values.
        """
        # Build covariance matrix
        dists = cdist(mesh_points, mesh_points)
        C = sigma**2 * np.exp(-(dists**2) / (2.0 * length_scale**2))

        # Eigendecompose — keep top n_modes
        n_pts = len(mesh_points)
        n_modes = min(n_modes, n_pts)
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            C, subset_by_index=[n_pts - n_modes, n_pts - 1]
        )
        # Reverse to descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Basis matrix: √λᵢ φᵢ(x), shape (n_pts, n_modes)
        basis = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]

        # Initialize Z coefficients
        if key is None:
            Z = jnp.zeros(n_modes + 1, dtype=jnp.float64)
        else:
            Z = init_std * jax.random.normal(key, (n_modes + 1,), dtype=jnp.float64)

        return cls(Z=Z, basis=basis, sigma_psi0=sigma_psi0)

    @classmethod
    def from_covariance_operator(
        cls,
        A: np.ndarray,
        M: np.ndarray,
        *,
        n_modes: int,
    ) -> "KLBasis":
        """Construct a mass-aware KL basis from a discretized covariance operator.

        Solves the dense generalized symmetric eigenproblem ``A φ = λ M φ``
        with ``M`` the mass (metric) matrix and keeps the ``n_modes`` leading
        eigenpairs (largest λ).  The returned modes are M-orthonormal
        (``Φᵀ M Φ = I``), which makes the expansion consistent under mesh
        refinement: eigenvalues and mode shapes converge to those of the
        continuous operator instead of drifting with the node count.

        The factorization runs host-side via ``scipy.linalg.eigh``
        (which handles the M-orthonormalization natively) — construction is
        not jit-traceable — and the results are stored as host-side NumPy
        metadata; ``psi`` converts to a JAX array at evaluation time.

        Stored fields: ``eigenvalues`` (leading λ, descending),
        ``mass_matrix`` (the symmetrized M), ``modes`` (M-orthonormal Φ), and
        ``basis`` = ``Φ · √λ`` so
        [psi][diff_soup.inference.approx.grf.KLBasis.psi] and
        [k][diff_soup.inference.approx.grf.KLBasis.k] weight the modes
        exactly like the RBF constructor.  ``Z`` is initialised to zeros.

        Typical usage::

            kl = KLBasis.from_covariance_operator(A, M, n_modes=12)
            psi = kl.psi()                       # (n_pts,) log-field
            gram = kl.modes.T @ kl.mass_matrix @ kl.modes   # identity

        Args:
            A: Discretized covariance operator, symmetric, shape ``(n, n)``.
                Round-off asymmetry is symmetrized away; gross asymmetry is
                rejected.
            M: Mass (metric) matrix, symmetric positive definite, same shape
                as ``A`` (e.g. a FEM or quadrature mass matrix).
            n_modes: Number of leading eigenpairs to keep; must satisfy
                ``1 <= n_modes <= n``.

        Returns:
            A [KLBasis][diff_soup.inference.approx.grf.KLBasis] whose
            ``modes``, ``mass_matrix``, and
            ``eigenvalues`` fields are populated.
        """
        A = _symmetrize_checked(np.asarray(A, dtype=float), "A")
        M = _symmetrize_checked(np.asarray(M, dtype=float), "M")
        if A.shape != M.shape:
            raise ValueError(
                f"A and M must have the same shape; got A of shape {A.shape} "
                f"and M of shape {M.shape}."
            )
        try:
            np.linalg.cholesky(M)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "mass matrix M must be positive definite (its Cholesky "
                "factorization failed); provide an SPD metric such as a FEM "
                "or quadrature mass matrix."
            ) from exc
        n = A.shape[0]
        if not 1 <= n_modes <= n:
            raise ValueError(
                f"n_modes must satisfy 1 <= n_modes <= {n} (the matrix size); got {n_modes}."
            )
        eigenvalues, eigenvectors = scipy.linalg.eigh(A, M, subset_by_index=[n - n_modes, n - 1])
        # Reverse to descending order; eigh returns M-orthonormal vectors.
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        basis = np.ascontiguousarray(eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :])
        return cls(
            Z=jnp.zeros(n_modes + 1, dtype=jnp.float64),
            basis=basis,
            sigma_psi0=1.0,
            modes=np.ascontiguousarray(eigenvectors),
            mass_matrix=M,
            mass_diagonal=None,
            eigenvalues=np.ascontiguousarray(eigenvalues),
        )

    @classmethod
    def from_lumped_covariance(
        cls,
        covariance_operator,
        lumped_mass: np.ndarray,
        *,
        n_modes: int,
        solver: str = "auto",
        tol: float = 0.0,
        maxiter: Optional[int] = None,
        ncv: Optional[int] = None,
        sigma_psi0: float = 1.0,
        dtype: Any = jnp.float64,
    ) -> "KLBasis":
        """Construct a mass-aware KL basis without a dense mass matrix.

        For ``M = diag(w)`` and covariance action ``C``, this solves the
        symmetric standard problem

        ``diag(sqrt(w)) C diag(sqrt(w)) v = λ v``

        and returns ``phi = v / sqrt(w)``. Consequently
        ``phi.T @ diag(w) @ phi == I`` while storage is O(Nk + N), never
        O(N²) for the metric. ``covariance_operator`` may be a dense/sparse
        matrix, a SciPy ``LinearOperator``, or a callable accepting one vector
        (and optionally a block of column vectors).

        Args:
            covariance_operator: Covariance matrix/action ``C``.
            lumped_mass: Strictly positive diagonal entries ``w``, shape
                ``(n,)``.
            n_modes: Number of leading eigenpairs.
            solver: ``"auto"`` (dense only for small explicit matrices),
                ``"eigsh"`` (matrix-free Lanczos), or ``"dense"`` (reference).
            tol: ``scipy.sparse.linalg.eigsh`` convergence tolerance. Zero asks
                SciPy for machine-precision convergence.
            maxiter: Optional Lanczos iteration cap.
            ncv: Optional Lanczos subspace width.
            sigma_psi0: Constant-mode prior scale.
            dtype: Runtime coefficient/basis dtype. Eigensolving is performed
                host-side in float64 and cast only at the JAX boundary.

        Returns:
            A [KLBasis][diff_soup.inference.approx.grf.KLBasis] with
            ``mass_diagonal`` populated and ``mass_matrix is None``.
        """
        from scipy.sparse import issparse
        from scipy.sparse.linalg import LinearOperator, aslinearoperator, eigsh

        lumped = np.asarray(lumped_mass, dtype=np.float64)
        if lumped.ndim != 1 or lumped.size == 0:
            raise ValueError("lumped_mass must be a non-empty one-dimensional array.")
        lumped = np.ascontiguousarray(lumped)
        if not np.all(np.isfinite(lumped)) or np.any(lumped <= 0.0):
            raise ValueError("lumped_mass must contain only finite, strictly positive values.")
        n = int(lumped.size)
        if not 1 <= int(n_modes) <= n:
            raise ValueError(f"n_modes must satisfy 1 <= n_modes <= {n}; got {n_modes}.")
        if solver not in ("auto", "eigsh", "dense"):
            raise ValueError(f"solver must be 'auto', 'eigsh', or 'dense'; got {solver!r}.")

        explicit = isinstance(covariance_operator, np.ndarray) or issparse(covariance_operator)
        if explicit:
            covariance = aslinearoperator(covariance_operator)
            if covariance.shape != (n, n):
                raise ValueError(
                    "covariance_operator and lumped_mass dimensions differ: "
                    f"{covariance.shape} versus {(n, n)}."
                )
        elif isinstance(covariance_operator, LinearOperator):
            covariance = covariance_operator
            if covariance.shape != (n, n):
                raise ValueError(
                    "covariance_operator and lumped_mass dimensions differ: "
                    f"{covariance.shape} versus {(n, n)}."
                )
        elif callable(covariance_operator):

            def _matvec(vector):
                value = np.asarray(
                    covariance_operator(np.asarray(vector, dtype=np.float64)),
                    dtype=np.float64,
                )
                if value.shape != (n,):
                    raise ValueError(
                        f"covariance_operator(vector) must return shape {(n,)}; got {value.shape}."
                    )
                return value

            def _matmat(block):
                block = np.asarray(block, dtype=np.float64)
                try:
                    value = np.asarray(covariance_operator(block), dtype=np.float64)
                except (TypeError, ValueError):
                    value = np.column_stack([_matvec(block[:, i]) for i in range(block.shape[1])])
                if value.shape != block.shape:
                    raise ValueError(
                        "covariance_operator(block) must return the same "
                        f"shape as its input; got {value.shape} for {block.shape}."
                    )
                return value

            covariance = LinearOperator(
                (n, n),
                matvec=_matvec,
                matmat=_matmat,
                rmatvec=_matvec,
                dtype=np.dtype(np.float64),
            )
        else:
            raise TypeError(
                "covariance_operator must be a matrix, scipy LinearOperator, "
                "or callable matrix-vector action."
            )

        sqrt_mass = np.sqrt(lumped)

        def _standard_matvec(vector):
            scaled = sqrt_mass * np.asarray(vector, dtype=np.float64)
            return sqrt_mass * np.asarray(covariance.matvec(scaled), dtype=np.float64)

        def _standard_matmat(block):
            scaled = sqrt_mass[:, None] * np.asarray(block, dtype=np.float64)
            return sqrt_mass[:, None] * np.asarray(covariance.matmat(scaled), dtype=np.float64)

        standard = LinearOperator(
            (n, n),
            matvec=_standard_matvec,
            matmat=_standard_matmat,
            rmatvec=_standard_matvec,
            dtype=np.dtype(np.float64),
        )

        use_dense = solver == "dense" or (
            solver == "auto" and explicit and n <= max(256, 4 * int(n_modes))
        )
        if int(n_modes) == n:
            use_dense = True
        if use_dense:
            identity = np.eye(n, dtype=np.float64)
            matrix = np.asarray(standard.matmat(identity), dtype=np.float64)
            matrix = _symmetrize_checked(matrix, "lumped covariance transform")
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                matrix, subset_by_index=[n - int(n_modes), n - 1]
            )
        else:
            # ARPACK otherwise seeds its Lanczos space from NumPy's ambient
            # RNG, so identical immutable inputs can produce byte-different
            # modes in a fresh process. A deterministic, non-constant ramp
            # keeps content-addressed bases and checkpoints reproducible.
            v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
            v0 /= np.linalg.norm(v0)
            eigenvalues, eigenvectors = eigsh(
                standard,
                k=int(n_modes),
                which="LA",
                v0=v0,
                tol=float(tol),
                maxiter=maxiter,
                ncv=ncv,
                return_eigenvectors=True,
            )

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
        # Eigenvector signs are arbitrary. Canonical signs make fresh artifacts
        # stable across LAPACK/ARPACK builds when the eigenspace is separated.
        for column in range(eigenvectors.shape[1]):
            pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
            if eigenvectors[pivot, column] < 0.0:
                eigenvectors[:, column] *= -1.0
        modes = eigenvectors / sqrt_mass[:, None]
        basis = modes * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
        runtime_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(runtime_dtype, jnp.floating):
            raise TypeError(f"KLBasis dtype must be floating; got {dtype!r}.")
        return cls(
            Z=jnp.zeros(int(n_modes) + 1, dtype=runtime_dtype),
            basis=np.ascontiguousarray(basis.astype(np.dtype(runtime_dtype))),
            sigma_psi0=float(sigma_psi0),
            modes=np.ascontiguousarray(modes),
            mass_matrix=None,
            mass_diagonal=np.ascontiguousarray(lumped),
            eigenvalues=np.ascontiguousarray(eigenvalues),
        )

    def psi(self):
        """Log-heterogeneity field ψ(x) at all mesh nodes, shape ``(n_pts,)``."""
        constant = self.sigma_psi0 * self.Z[0]
        spatial = jnp.asarray(self.basis, dtype=self.Z.dtype) @ self.Z[1:]
        return constant + spatial

    def k(self):
        """Positive heterogeneity field ``exp(ψ(x))`` at all mesh nodes."""
        return jnp.exp(self.psi())

    def __call__(self, coordinates=None):
        """Evaluate the log-heterogeneity field.

        Returns the same array as
        [psi][diff_soup.inference.approx.grf.KLBasis.psi].

        Present so a basis can be declared directly as a named learnable
        function (``spec.function("psi", kl, learnable=True)``) and evaluated
        inside an equation as ``q.fn.psi()``.
        """
        _check_domain_points(self, coordinates)
        return self.psi()

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the fitted leaf path(s) of this basis (``"Z"``).

        The precomputed mode arrays are fixed metadata of the chosen
        representation, so the coefficient vector is the whole fitted set.
        """
        return ("Z",)

    def mean_weights(self, node_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Coefficients of a weighted spatial mean of ψ as a row over ``Z``.

        The weighted mean ``Σ_j a_j ψ(x_j) / Σ_j a_j`` is linear in the
        coefficients, so it can be imposed exactly as one equality row rather
        than added as a penalty.  Pass the returned row to a zero-mean gauge to
        normalize a field whose overall level is not identifiable — for example
        when a spatial log-field and a global rate constant carry the same
        multiplicative scale and would otherwise leave an exact null direction
        in the normal equations.

        Args:
            node_weights: Nonnegative per-node weights ``(n_pts,)``, such as
                nodal areas or volumes.  They are normalized to sum to one, so
                the row computes a weighted mean rather than a weighted sum.
                ``None`` uses the lumped mass this basis was built with (the
                domain's nodal volumes), so a basis from ``from_domain`` or
                ``from_lumped_covariance`` needs no weights from the caller.

        Returns:
            A ``(n_modes + 1,)`` NumPy row ``w`` with
            ``w @ Z == Σ_j a_j ψ(x_j) / Σ_j a_j`` for every ``Z``.

        Raises:
            ValueError: If the weights have the wrong length, are negative, or
                sum to zero.
        """
        if node_weights is None:
            if self.mass_diagonal is None:
                raise ValueError(
                    "mean_weights() without node_weights needs a basis built with a lumped "
                    "mass (from_domain or from_lumped_covariance); this basis carries none, "
                    "so pass the nodal weights explicitly."
                )
            node_weights = self.mass_diagonal
        weights = np.asarray(node_weights, dtype=float).reshape((-1,))
        if weights.shape[0] != self.n_pts:
            raise ValueError(
                f"node_weights must have one entry per mesh node ({self.n_pts}); "
                f"got {weights.shape[0]}."
            )
        if np.any(weights < 0.0):
            raise ValueError("node_weights must be nonnegative.")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("node_weights must sum to a positive value.")
        weights = weights / total
        row = np.empty(self.n_modes + 1, dtype=float)
        row[0] = float(self.sigma_psi0)
        row[1:] = weights @ np.asarray(self.basis)
        return row

    @property
    def n_modes(self) -> int:
        """Number of spatial eigenmodes (excluding constant mode)."""
        return self.basis.shape[1]

    @property
    def n_pts(self) -> int:
        """Number of mesh nodes."""
        return self.basis.shape[0]


class GRFGenerator:
    """Draw spatially correlated random fields from an RBF-covariance prior.

    Builds an approximate Karhunen-Loève basis by eigendecomposing the RBF
    covariance on a random subset of the mesh, then interpolates to all nodes
    via RBF interpolation.

    Args:
        mesh_points: All node coordinates, shape ``(n_pts, 2)``.
        length_scale: RBF covariance length scale (same units as coordinates).
        num_modes: Number of eigenmodes to retain.
        rng: A ``numpy.random.Generator``; the global RNG state is never touched.
    """

    @classmethod
    def from_domain(cls, domain, *, length_scale=4.0, num_modes=20, rng=None):
        """Construct mass-weighted RBF modes on every point of a domain.

        Sampling uses the same metric-weighted KL expansion as KLBasis.
        The raw constructor remains the subset-and-interpolation alternative.
        """
        basis = KLBasis.from_domain(domain, length_scale=length_scale, n_modes=num_modes)
        result = cls.__new__(cls)
        result.num_modes = basis.n_modes
        result.num_points = basis.num_points
        result.domain_fingerprint = basis.domain_fingerprint
        result.domain_name = basis.domain_name
        result.mesh_points = np.asarray(domain.node_coordinates())
        result.sample_pts = result.mesh_points
        result._rng = rng if rng is not None else np.random.default_rng()
        result.eigenvalues = basis.eigenvalues
        result.eigenvectors = basis.modes
        result.basis_sample = basis.basis
        result._scipy_interpolate = scipy.interpolate
        return result

    def __init__(
        self,
        mesh_points: np.ndarray,
        length_scale: float = 4.0,
        num_modes: int = 20,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Precompute the approximate KL basis used by ``generate``.

        See [generate][diff_soup.inference.approx.grf.GRFGenerator.generate].

        Args:
            mesh_points: Node coordinates ``(n_pts, 2)``.
            length_scale: RBF covariance length scale.
            num_modes: Number of eigenmodes to retain.
            rng: A ``numpy.random.Generator`` for the sample-point draw; the
                global ``np.random`` state is never used.
                ``None`` defaults to ``np.random.default_rng()``.

        Note:
            NumPy/SciPy only — this class is *not* JIT-stable and should
            not be called inside a JAX trace. Use it offline to seed fields.
        """
        self.num_modes = num_modes
        self.mesh_points = mesh_points
        self._rng = rng if rng is not None else np.random.default_rng()

        sample_size = min(1000, len(mesh_points))
        sample_idx = self._rng.choice(len(mesh_points), sample_size, replace=False)
        self.sample_pts = mesh_points[sample_idx]

        dists = cdist(self.sample_pts, self.sample_pts)
        covariance = np.exp(-(dists**2) / (2.0 * length_scale**2))
        vals, vecs = scipy.linalg.eigh(covariance)

        idx = np.argsort(vals)[::-1]
        self.eigenvalues = vals[idx][:num_modes]
        self.eigenvectors = vecs[:, idx][:, :num_modes]
        self.basis_sample = self.eigenvectors * np.sqrt(np.maximum(self.eigenvalues, 0.0))
        self._scipy_interpolate = scipy.interpolate

    def generate(
        self,
        seed: Optional[int] = None,
        mean: float = 1.0,
        std: float = 0.2,
    ) -> np.ndarray:
        """Draw one realisation of the random field.

        Returns a positive field at all mesh nodes: ``mean * exp(std * z̃)``
        where z̃ is the normalised log-field, clipped to [0.1, 5.0].

        A ``seed`` builds a fresh local
        ``np.random.default_rng(seed)`` (reproducible, reentrant) instead of
        mutating the global ``np.random`` state; ``None`` reuses the generator's
        own ``rng`` (set at construction).

        Args:
            seed: Optional seed for a fresh local generator (reproducible).
            mean: Geometric mean of the returned field.
            std: Standard deviation of the normalised log-field.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        z = rng.normal(0.0, 1.0, self.num_modes)
        log_field_sample = self.basis_sample @ z
        rbf = self._scipy_interpolate.RBFInterpolator(
            self.sample_pts, log_field_sample, kernel="linear"
        )
        log_field = rbf(self.mesh_points)
        log_field_norm = (log_field - log_field.mean()) / (log_field.std() + 1e-8)
        return np.clip(mean * np.exp(std * log_field_norm), 0.1, 5.0)
