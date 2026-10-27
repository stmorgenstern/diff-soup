"""Generation counters and the exact-root certificate for factor reuse.

A solver engine owns carried linearization artifacts — the linearization
point, parameter and control identities, iteration-matrix values, an opaque
numeric factor, and (for bordered systems) the coupling blocks ``B`` / ``C``
/ ``D``. [Generations][diff_soup.nonlinear.generations.Generations] names each artifact with a monotone build
counter that rides the engine's own functional state: plain integer scalars,
no resource objects, so the record is safe under ``jit``, ``vmap``,
``lax.scan``, and checkpoint replay.

Reusing a held numeric factor for an implicit-function derivative is legal
only when the factor is exactly the sensitivity matrix at a converged root.
That authority is an [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate]: it can only be issued by
[diff_soup.nonlinear.generations.certify_exact_root][] — called exactly where a solver decides root
acceptance — never by public construction. A missing certificate never
authorizes reuse; [diff_soup.nonlinear.generations.certificate_authorizes][] is the quiet gate for
callers with a rebuild path, and [diff_soup.nonlinear.generations.assert_certified][] is the loud gate
for callers without one.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "ExactRootCertificate",
    "Generations",
    "assert_certified",
    "certificate_authorizes",
    "certify_exact_root",
    "initial_generations",
]

_CORE_COUNTERS = ("point", "parameters", "control", "matrix", "factor")
_COUPLING_COUNTERS = ("b", "c", "d")


def _as_counter(value) -> jax.Array:
    """Coerce a counter to an int32 scalar."""
    return jnp.asarray(value, dtype=jnp.int32)


class Generations(eqx.Module):
    """Monotone build counters for one engine carry's linearization artifacts.

    Each counter is an int32 scalar that increments exactly when the named
    artifact is rebuilt (see [bump][diff_soup.nonlinear.generations.Generations.bump]); equality of counters is the
    functional identity used to validate reuse. Coupling-block counters
    ``b`` / ``c`` / ``d`` exist only on records built with
    ``initial_generations(coupling_blocks=True)``.

    Attributes:
        point: Linearization-point identity ``(t, y, ydot)``.
        parameters: Parameter (``theta``) identity.
        control: Control identity.
        matrix: Iteration-matrix values build count.
        factor: Numeric-factor build count.
        b: Optional coupling-block ``B`` build count.
        c: Optional coupling-block ``C`` build count.
        d: Optional coupling-block ``D`` build count.
    """

    point: jax.Array
    parameters: jax.Array
    control: jax.Array
    matrix: jax.Array
    factor: jax.Array
    b: Optional[jax.Array] = None
    c: Optional[jax.Array] = None
    d: Optional[jax.Array] = None

    def bump(self, *names: str) -> "Generations":
        """Return a copy with each named counter incremented by one.

        Args:
            names: Counter names among ``point``, ``parameters``, ``control``,
                ``matrix``, ``factor``, and — on a record carrying coupling
                blocks — ``b``, ``c``, ``d``.

        Returns:
            The bumped record; the original is unchanged.

        Raises:
            ValueError: For a name that is not a counter, or for a
                coupling-block name on a record built without coupling blocks.
        """
        fields = {name: getattr(self, name) for name in _CORE_COUNTERS}
        fields.update({name: getattr(self, name) for name in _COUPLING_COUNTERS})
        for name in names:
            if name in _COUPLING_COUNTERS:
                if fields[name] is None:
                    raise ValueError(
                        f"generation counter {name!r} is a coupling-block counter "
                        f"but this record carries none; build the record with "
                        f"initial_generations(coupling_blocks=True) to track "
                        f"coupling-block generations."
                    )
            elif name not in _CORE_COUNTERS:
                raise ValueError(
                    f"unknown generation counter {name!r}; counters are "
                    f"{_CORE_COUNTERS + _COUPLING_COUNTERS}."
                )
            fields[name] = fields[name] + 1
        return Generations(**fields)

    def advance(self, **deltas) -> "Generations":
        """Return a copy with each named counter increased by a given count.

        The traced-count form of
        [bump][diff_soup.nonlinear.generations.Generations.bump]: an
        attempt that performs
        several rebuild events (for example two linear-solver setups) can
        advance a counter by the exact event count in one call, keeping the
        counter equal to the number of times the artifact was actually
        rebuilt.

        Args:
            deltas: ``counter_name=count`` pairs; counts are non-negative
                integers (concrete or traced ``int32`` scalars). Counter
                names follow
                [bump][diff_soup.nonlinear.generations.Generations.bump].

        Returns:
            The advanced record; the original is unchanged.

        Raises:
            ValueError: For a name that is not a counter, or for a
                coupling-block name on a record built without coupling
                blocks.
        """
        fields = {name: getattr(self, name) for name in _CORE_COUNTERS}
        fields.update({name: getattr(self, name) for name in _COUPLING_COUNTERS})
        for name, count in deltas.items():
            if name in _COUPLING_COUNTERS:
                if fields[name] is None:
                    raise ValueError(
                        f"generation counter {name!r} is a coupling-block counter "
                        f"but this record carries none; build the record with "
                        f"initial_generations(coupling_blocks=True) to track "
                        f"coupling-block generations."
                    )
            elif name not in _CORE_COUNTERS:
                raise ValueError(
                    f"unknown generation counter {name!r}; counters are "
                    f"{_CORE_COUNTERS + _COUPLING_COUNTERS}."
                )
            fields[name] = fields[name] + _as_counter(count)
        return Generations(**fields)


def initial_generations(*, coupling_blocks: bool = False) -> Generations:
    """A fresh all-zero [Generations][diff_soup.nonlinear.generations.Generations] record.

    Args:
        coupling_blocks: When true, the record also tracks the coupling-block
            counters ``b`` / ``c`` / ``d`` (bordered / Schur-complement
            systems); otherwise those fields are absent.
    """
    zero = _as_counter(0)
    coupling = {name: zero for name in _COUPLING_COUNTERS} if coupling_blocks else {}
    return Generations(
        point=zero, parameters=zero, control=zero, matrix=zero, factor=zero, **coupling
    )


# Module-private issuance token: holding a reference to this object is what
# authorizes constructing an ExactRootCertificate. Only the issuing helper
# colocated with the root-success logic imports it.
_ISSUER_TOKEN = object()


class ExactRootCertificate(eqx.Module):
    """Attestation that a held factor is exact at a converged root.

    The certificate names the matrix-values and factor generations (and,
    when the issuing carry tracks them, the coupling-block generations) that
    were current when a root solve succeeded, plus the root-success predicate
    itself. It is issued exclusively by
    [certify_exact_root][diff_soup.nonlinear.generations.certify_exact_root];
    direct construction raises, so presenting a certificate always traces
    back to a root-success decision. Validate with
    [certificate_authorizes][diff_soup.nonlinear.generations.certificate_authorizes]
    or [assert_certified][diff_soup.nonlinear.generations.assert_certified]
    — never by reading fields directly.

    Attributes:
        matrix_generation: The matrix-values generation attested exact at the
            root.
        factor_generation: The generation of the factor built from those
            values.
        at_root: The root-success predicate at issuance; a certificate issued
            off-root never authorizes reuse.
        b_generation: Optional coupling-block ``B`` generation at issuance.
        c_generation: Optional coupling-block ``C`` generation at issuance.
        d_generation: Optional coupling-block ``D`` generation at issuance.
    """

    matrix_generation: jax.Array
    factor_generation: jax.Array
    at_root: jax.Array
    b_generation: Optional[jax.Array] = None
    c_generation: Optional[jax.Array] = None
    d_generation: Optional[jax.Array] = None

    def __init__(
        self,
        *,
        matrix_generation,
        factor_generation,
        at_root,
        b_generation=None,
        c_generation=None,
        d_generation=None,
        _issuer=None,
    ):
        """Reject public construction; certificates are issued, never built."""
        if _issuer is not _ISSUER_TOKEN:
            raise TypeError(
                "an ExactRootCertificate cannot be constructed directly: "
                "exact-at-root identity is issued by the root-success helper "
                "certify_exact_root (diff_soup.problem.implicit.root_ift) when a "
                "root solve succeeds. Obtain the certificate from that helper."
            )
        self.matrix_generation = _as_counter(matrix_generation)
        self.factor_generation = _as_counter(factor_generation)
        self.at_root = jnp.asarray(at_root, dtype=bool)
        self.b_generation = None if b_generation is None else _as_counter(b_generation)
        self.c_generation = None if c_generation is None else _as_counter(c_generation)
        self.d_generation = None if d_generation is None else _as_counter(d_generation)


def _issue_certificate(generations: Generations, root_ok) -> ExactRootCertificate:
    """Build a certificate from a generation record (issuing helper only)."""
    return ExactRootCertificate(
        matrix_generation=generations.matrix,
        factor_generation=generations.factor,
        at_root=root_ok,
        b_generation=generations.b,
        c_generation=generations.c,
        d_generation=generations.d,
        _issuer=_ISSUER_TOKEN,
    )


def certify_exact_root(generations: Generations, root_ok) -> ExactRootCertificate:
    """Issue the exact-root certificate for a carry's held linearization.

    This is the only construction path for an
    [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate].
    Call it exactly where a root solve's acceptance is decided, passing that
    decision as ``root_ok`` — the certificate is issued for the generations
    current at that moment and never authorizes reuse once any of them moves.

    Args:
        generations: The carry's
            [Generations][diff_soup.nonlinear.generations.Generations] at the acceptance
            decision — the factor being certified must be the one those
            counters describe.
        root_ok: The root solve's acceptance predicate (traced boolean). A
            certificate issued with a false predicate never authorizes reuse.

    Returns:
        The issued certificate, ready to ride the carry next to its
        generation record.
    """
    return _issue_certificate(generations, root_ok)


def certificate_authorizes(
    certificate: Optional[ExactRootCertificate], generations: Generations
) -> jax.Array:
    """Whether ``certificate`` authorizes reusing the carry's held factor.

    Authorization requires a certificate that was issued at a converged root
    and whose recorded generations equal the carry's **current** counters:
    matrix values, factor, and every coupling-block counter either side
    tracks. A missing certificate, one issued off-root, one attesting other
    matrix values (wrong generation), or one predating a factor or
    coupling-block rebuild (stale) yields false.

    Args:
        certificate: The presented
            [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate],
            or ``None``.
        generations: The carry's current
            [Generations][diff_soup.nonlinear.generations.Generations].

    Returns:
        A traced boolean scalar; concrete ``False`` when ``certificate`` is
        ``None``. Safe under ``jit`` / ``vmap``.
    """
    if certificate is None:
        return jnp.asarray(False)
    ok = jnp.logical_and(
        jnp.asarray(certificate.at_root),
        jnp.logical_and(
            certificate.matrix_generation == generations.matrix,
            certificate.factor_generation == generations.factor,
        ),
    )
    for cert_gen, carry_gen in (
        (certificate.b_generation, generations.b),
        (certificate.c_generation, generations.c),
        (certificate.d_generation, generations.d),
    ):
        if (cert_gen is None) != (carry_gen is None):
            # The certificate and the carry disagree about whether coupling
            # blocks exist at all: never authorize.
            return jnp.asarray(False)
        if cert_gen is not None:
            ok = jnp.logical_and(ok, cert_gen == carry_gen)
    return ok


def assert_certified(value, certificate, generations: Generations):
    """Gate ``value`` on a full certificate attestation, loudly.

    For reuse sites without a rebuild fallback: returns ``value`` unchanged
    when
    [certificate_authorizes][diff_soup.nonlinear.generations.certificate_authorizes]
    holds and raises a runtime error otherwise (including under ``jit``).
    Callers that can rebuild should branch on
    [certificate_authorizes][diff_soup.nonlinear.generations.certificate_authorizes]
    instead.

    Args:
        value: The array (or pytree of arrays) whose consumption depends on
            the reuse being legal.
        certificate: The presented
            [ExactRootCertificate][diff_soup.nonlinear.generations.ExactRootCertificate],
            or ``None``.
        generations: The carry's current
            [Generations][diff_soup.nonlinear.generations.Generations].

    Returns:
        ``value``, unchanged, when the certificate authorizes reuse.

    Raises:
        ValueError: When no certificate was presented — a missing certificate
            never authorizes factor reuse.
        equinox.EquinoxRuntimeError: (at run time, also under ``jit``) when
            the certificate does not attest the current matrix and factor
            generations at a converged root.
    """
    if certificate is None:
        raise ValueError(
            "factor reuse requires an exact-root certificate and none was "
            "presented; a missing certificate never authorizes reuse. Rebuild "
            "the factor from fresh matrix values, or present the certificate "
            "issued by certify_exact_root when this factor was built."
        )
    ok = certificate_authorizes(certificate, generations)
    return eqx.error_if(
        value,
        jnp.logical_not(ok),
        "factor reuse was attempted with a certificate that does not attest "
        "the carry's current matrix and factor generation at a converged root; "
        "reuse requires the certificate issued when the held factor was built "
        "at the accepted root. Rebuild the factor from fresh matrix values.",
    )
