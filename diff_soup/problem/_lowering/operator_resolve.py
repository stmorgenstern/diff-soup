"""The one place lowering turns a domain (and optional factory) into a bank.

``resolve_operator(domain, factory)`` constructs and validates the
[AbstractOperator][diff_soup.domains.operators.base.AbstractOperator] bank
for a domain, substituting the built-in
[Operators][diff_soup.domains.operators.base.Operators]
when no factory is given. It is the single entry point every lowering path uses
instead of constructing a bank directly, so bank construction is validated in
exactly one place before any JAX tracing.

Residual paths call ``.for_residual()`` on the result to obtain the
provenance-tagging view injected as ``q.ops``; host-side paths (mass weights,
observations, constraints, boundary contexts) use the returned bank directly.
"""

from __future__ import annotations

from diff_soup._errors import ConfigurationError
from diff_soup.domains.operators.base import AbstractOperator, Operators

__all__ = ["resolve_operator"]


def resolve_operator(domain, factory=None) -> AbstractOperator:
    """Construct and validate the operator bank bound to ``domain``.

    Args:
        domain: the bound domain the bank operates on.
        factory: an
            [AbstractOperator][diff_soup.domains.operators.base.AbstractOperator]
            subclass or top-level named callable
            ``domain -> AbstractOperator``. ``None`` selects the
            built-in [Operators][diff_soup.domains.operators.base.Operators].

    Returns:
        A validated
        [AbstractOperator][diff_soup.domains.operators.base.AbstractOperator]
        bound to ``domain``.

    Raises:
        ConfigurationError: if the factory fails to construct, returns a
            non-[AbstractOperator][diff_soup.domains.operators.base.AbstractOperator],
            returns a bank bound to a different
            domain, or reports an empty/non-string ``discretization_type``. The
            error is raised before any tracing.
    """
    # (a) Build the bank from the current domain.
    if factory is None:
        bank = Operators(domain)
    else:
        try:
            bank = factory(domain)
        except Exception as exc:
            raise ConfigurationError(
                f"operator factory {factory!r} failed to construct a bank for the "
                f"domain; the factory must be callable as factory(domain) and "
                f"return an AbstractOperator bound to that domain"
            ) from exc

    # (b) The result must be an AbstractOperator.
    if not isinstance(bank, AbstractOperator):
        raise ConfigurationError(
            f"operator factory {factory!r} returned {type(bank).__name__}, which is "
            f"not an AbstractOperator; the factory must return an AbstractOperator "
            f"subclass instance"
        )

    # (c) The bank must be bound to the same domain object (identity).
    # ``domain`` is a declared field of ``AbstractOperator``, so a bank
    # without one cannot be constructed at all; only the binding is checked.
    if bank.domain is not domain:
        raise ConfigurationError(
            f"operator bank {type(bank).__name__} is bound to a different domain "
            f"than the one being lowered; the factory must build the bank from the "
            f"domain it is given"
        )

    # (d) The discretization tag must be a non-empty string.
    try:
        dt = bank.discretization_type
    except Exception as exc:
        raise ConfigurationError(
            f"operator bank {type(bank).__name__} does not expose a readable discretization_type"
        ) from exc
    if not isinstance(dt, str) or not dt:
        raise ConfigurationError(
            f"operator bank {type(bank).__name__} reported discretization_type "
            f"{dt!r}; discretization_type must be a non-empty string (e.g. 'fvm', "
            f"'fdm', 'fem', or a custom tag)"
        )

    # (e) Validated.
    return bank
