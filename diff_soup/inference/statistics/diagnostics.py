"""diagnostics.py — model-adequacy checks for fitted experiments.

Uncertainty statements are conditional on the model being adequate:
[residual_report][] checks each scored channel's standardized residuals
against what its declared likelihood predicts — zero mean, unit scale,
matching total cost, and no serial correlation — and flags plain-words
departures. A flagged report does not say which assumption failed; it says
the declared observation model and the data disagree, so intervals built on
that likelihood inherit the disagreement.

[mcmc_report][] diagnoses MCMC output from the draw arrays alone — no
sampler is rerun. It reports rank-normalized split-Rhat, bulk and tail
effective sample sizes, Monte-Carlo standard errors, and (when the kernel
provided them) acceptance-rate, divergence, and tree-depth summaries,
following Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021).

Everything here runs on the host after prediction; nothing is
differentiated.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from diff_soup.inference.data.deferred import DeferredValues
from diff_soup.inference.statistics.experiment import _channel_likelihood_view

__all__ = [
    "ChannelResiduals",
    "McmcReport",
    "ResidualReport",
    "mcmc_report",
    "residual_report",
]


@dataclasses.dataclass(frozen=True)
class ChannelResiduals:
    """Adequacy summary of one scored channel. Frozen.

    Attributes:
        name: The channel name.
        n_rows: Number of standardized residual rows scored.
        mean: Mean standardized residual (should be near zero).
        std: Standard deviation of the standardized residuals (near one when
            the declared noise scale matches the data).
        cost: Total squared standardized misfit ``Σ z²``.
        reduced_cost: ``cost / n_rows`` (near one under the declared model).
        lag1_autocorrelation: Lag-1 autocorrelation of the standardized
            residuals in frame order (near zero for independent errors).
        standardized: The standardized residual array, frame-ordered.
    """

    name: str
    n_rows: int
    mean: float
    std: float
    cost: float
    reduced_cost: float
    lag1_autocorrelation: float
    standardized: np.ndarray


@dataclasses.dataclass(frozen=True)
class ResidualReport:
    """Residual adequacy evidence for one experiment at one parameter point.

    Attributes:
        channels: Per-channel [ChannelResiduals][] by name.
        flags: Plain-words departures from the declared observation model;
            empty when every check is quiet.
        evaluated_at: Provenance — the flat parameter vector used.
    """

    channels: dict
    flags: tuple
    evaluated_at: np.ndarray


def _lag1(z: np.ndarray) -> float:
    """Lag-1 autocorrelation of a flattened residual sequence."""
    z = z.reshape(-1)
    if z.size < 3:
        return 0.0
    zc = z - z.mean()
    denom = float(np.sum(zc**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(zc[:-1] * zc[1:]) / denom)


def residual_report(problem, fit, *, experiment_id=None) -> ResidualReport:
    """Standardize the residuals of each measurement by its likelihood at a fit, and flag departures from the noise model.

    Under the declared noise model the standardized residuals have mean zero, standard deviation
    one, and no lag-1 autocorrelation.

    Args:
        problem: An [`InferenceProblem`][diff_soup.inference.InferenceProblem] whose measurements
            all declare a likelihood.
        fit: An [`OptResult`][diff_soup.inference.OptResult], a parameter vector, or a mapping
            from every fitted handle to its value.
        experiment_id: The experiment to check, required when the problem has several.

    Returns:
        A `ResidualReport` with one entry per measurement in `channels`, and the departures in
            words in `flags`.
    """
    from diff_soup.inference.statistics._analysis import experiment_view
    from diff_soup.inference.statistics.information import _point
    from diff_soup.inference.statistics.measurement import require_likelihoods

    require_likelihoods(problem, "residual_report")

    experiment, parameters, columns = experiment_view(problem, experiment_id)
    return _residual_report(experiment, _point(problem, fit)[columns], parameters=parameters)


def _residual_report(experiment, x_flat: Any, *, parameters: Any = None) -> ResidualReport:
    """Check each scored channel's residuals against its declared likelihood.

    Runs the experiment's shared prediction closure once at ``x_flat`` and
    standardizes each data-bearing response channel's misfit with its own
    likelihood. Flags name the channel and the failed check: scale (standard
    deviation far from one), bias (mean far from zero), or autocorrelation
    (serial correlation the declared independent errors do not allow).

    Args:
        experiment: The data-bearing experiment declaration.
        x_flat: The flat learnable-theta vector to evaluate at (e.g. a fit's
            converged theta block).
        parameters: Optional parameter groups selecting the learnable theta;
            must match the groups the vector ``x_flat`` was fitted with.

    Returns:
        A [ResidualReport][].

    Raises:
        ValueError: If no response channel has attached data.
    """
    predict = experiment.predictor(parameters=parameters)
    preds, _ = predict(jnp.asarray(x_flat))

    channels: dict = {}
    flags: list = []
    for c in experiment.channels:
        if c.role != "response" or c.name not in experiment.data:
            continue
        observed = experiment.data[c.name]
        if isinstance(observed, DeferredValues):
            observed = observed.materialize()
        if c.mask is not None:
            keep = np.where(np.asarray(c.mask).astype(bool))[0]
            observed = jnp.asarray(observed)[jnp.asarray(keep)]
        likelihood, context = _channel_likelihood_view(c, preds[c.name])
        z = np.asarray(
            likelihood.standardize(
                observed,
                preds[c.name],
                context=context,
            )
        )
        n = int(z.size)
        mean = float(z.mean()) if n else 0.0
        std = float(z.std()) if n else 0.0
        cost = float(np.sum(z**2))
        lag1 = _lag1(z)
        channels[c.name] = ChannelResiduals(
            name=c.name,
            n_rows=n,
            mean=mean,
            std=std,
            cost=cost,
            reduced_cost=cost / n if n else 0.0,
            lag1_autocorrelation=lag1,
            standardized=z,
        )
        if n:
            se_mean = 1.0 / np.sqrt(n)
            if abs(mean) > 4.0 * se_mean:
                flags.append(
                    f"channel {c.name!r}: standardized residual mean {mean:.3f} "
                    "departs from zero — biased predictions or a missing offset"
                )
            if abs(std - 1.0) > 0.5:
                flags.append(
                    f"channel {c.name!r}: standardized residual scale {std:.2f} "
                    "departs from one — the declared noise scale disagrees with "
                    "the data"
                )
            if abs(lag1) > 5.0 * se_mean and abs(lag1) > 0.2:
                flags.append(
                    f"channel {c.name!r}: lag-1 autocorrelation {lag1:.2f} — "
                    "serial correlation the declared independent errors do not "
                    "allow (unmodeled dynamics or correlated noise)"
                )

    if not channels:
        raise ValueError(
            "residual_report needs observed response data; attach it with "
            "Experiment.with_data first."
        )
    return ResidualReport(
        channels=channels,
        flags=tuple(flags),
        evaluated_at=np.asarray(x_flat),
    )


_ESS_MIN = 1e-6


@dataclasses.dataclass(frozen=True, repr=False)
class McmcReport:
    """Convergence and efficiency diagnostics for a set of MCMC draws. Frozen.

    Per-parameter arrays are indexed like the last axis of the draws. Kernel
    statistics (acceptance, divergences, tree depth) are ``None`` when the
    sampler did not report them.

    Attributes:
        names: Per-parameter labels, one per dimension.
        n_chains: Number of chains diagnosed.
        n_draws: Draws per chain (before splitting).
        mean: Posterior mean per parameter, on the original draw scale.
        sd: Posterior standard deviation per parameter (original scale,
            pooled over chains, ``ddof=1``).
        mcse_mean: Monte-Carlo standard error of the mean per parameter:
            ``sd / sqrt(ess_bulk)`` on the original scale.
        rhat: Rank-normalized split-Rhat per parameter. Values near one mean
            the split chains agree; ``nan`` for constant parameters.
        ess_bulk: Bulk effective sample size per parameter (rank-normalized
            split chains, Geyer initial-monotone-positive autocorrelation
            sum). Capped at the total retained draw count.
        ess_tail: Tail effective sample size per parameter — the smaller of
            the effective sample sizes of the 5% and 95% quantile indicator
            functions. Capped like ``ess_bulk``.
        acceptance_rate: Mean acceptance rate over all transitions, or
            ``None`` when the kernel did not report one.
        acceptance_rate_per_chain: Mean acceptance rate per chain, or
            ``None``.
        n_divergent: Total number of divergent transitions, or ``None``.
        divergences_per_chain: Divergent-transition count per chain, or
            ``None``.
        tree_depth_max: Largest per-transition tree depth (or integration-
            step count when the kernel reports that instead), or ``None``.
        tree_depth_mean: Mean per-transition tree depth (or integration-step
            count), or ``None``.
        flags: Plain-words diagnostics failures — parameters whose split-Rhat
            exceeds the threshold or whose bulk/tail effective sample size
            falls below the floor. Empty when every check is quiet.
        rhat_max: The split-Rhat threshold the flags were evaluated against.
        ess_floor: The effective-sample-size floor the flags were evaluated
            against.
    """

    names: tuple
    n_chains: int
    n_draws: int
    mean: np.ndarray
    sd: np.ndarray
    mcse_mean: np.ndarray
    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    acceptance_rate: float | None
    acceptance_rate_per_chain: np.ndarray | None
    n_divergent: int | None
    divergences_per_chain: np.ndarray | None
    tree_depth_max: int | None
    tree_depth_mean: float | None
    flags: tuple
    rhat_max: float
    ess_floor: float

    @property
    def ok(self) -> bool:
        """Whether every parameter passed the Rhat and effective-sample-size checks."""
        return not self.flags

    def summary(self) -> str:
        """Format the per-parameter table and kernel statistics as plain text."""
        width = max(9, *(len(str(n)) for n in self.names))
        lines = [
            f"MCMC report: {self.n_chains} chain(s) x {self.n_draws} draws, "
            f"{len(self.names)} parameter(s)",
            f"{'parameter':<{width}} {'mean':>11} {'sd':>11} {'mcse':>11} "
            f"{'rhat':>7} {'ess_bulk':>9} {'ess_tail':>9}",
        ]
        for j, name in enumerate(self.names):
            lines.append(
                f"{str(name):<{width}} {self.mean[j]:>11.4g} {self.sd[j]:>11.4g} "
                f"{self.mcse_mean[j]:>11.4g} {self.rhat[j]:>7.3f} "
                f"{self.ess_bulk[j]:>9.0f} {self.ess_tail[j]:>9.0f}"
            )
        if self.acceptance_rate is not None:
            per = ", ".join(f"{a:.3f}" for a in self.acceptance_rate_per_chain)
            lines.append(f"acceptance rate: {self.acceptance_rate:.3f} (per chain: {per})")
        if self.n_divergent is not None:
            per = ", ".join(str(int(d)) for d in self.divergences_per_chain)
            lines.append(f"divergences: {self.n_divergent} (per chain: {per})")
        if self.tree_depth_max is not None:
            lines.append(f"tree depth: max {self.tree_depth_max}, mean {self.tree_depth_mean:.2f}")
        if self.flags:
            lines.append("flags:")
            lines.extend(f"  - {flag}" for flag in self.flags)
        else:
            lines.append("flags: none")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return the compact human-readable summary table."""
        return self.summary()


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split each chain in half along the draw axis.

    Args:
        x: Draws of one parameter, shape ``(chain, draw)``.

    Returns:
        Array of shape ``(2 * chain, draw // 2)``. An odd trailing draw is
        dropped so both halves have equal length.
    """
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Replace values by normal scores of their pooled average ranks.

    Uses the fractional-rank transform ``z = Phi^-1((r - 3/8) / (S + 1/4))``
    with ``S`` the pooled value count and ties assigned average ranks.

    Args:
        x: Array of any shape; ranks are pooled over every element.

    Returns:
        Array of the same shape holding the normal scores.
    """
    flat = x.reshape(-1)
    ranks = rankdata(flat, method="average")
    z = ndtri((ranks - 0.375) / (flat.size + 0.25))
    return z.reshape(x.shape)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Per-row autocovariance (denominator ``n``) computed via FFT.

    Args:
        x: Array of shape ``(n_seq, n_len)``.

    Returns:
        Array of shape ``(n_seq, n_len)``; entry ``[m, t]`` is the lag-``t``
        autocovariance of row ``m`` about its own mean.
    """
    n = x.shape[1]
    xc = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(xc, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n]
    return acov.real / n


def _rhat(z: np.ndarray) -> float:
    """Potential-scale-reduction factor of split (rank-normalized) sequences.

    Args:
        z: Array of shape ``(n_seq, n_len)`` — the split chains, already
            rank-normalized.

    Returns:
        ``sqrt(var_plus / W)`` with ``W`` the mean within-sequence variance
        and ``var_plus`` the weighted within/between combination; ``nan``
        when the sequences are constant.
    """
    n = z.shape[1]
    within = float(z.var(axis=1, ddof=1).mean())
    if within <= 0.0:
        return float("nan")
    between_over_n = float(z.mean(axis=1).var(ddof=1))
    var_plus = (n - 1.0) / n * within + between_over_n
    return float(np.sqrt(var_plus / within))


def _ess(z: np.ndarray) -> float:
    """Effective sample size of split sequences via Geyer's truncated sum.

    Averages per-sequence autocovariances, forms the combined-chain
    autocorrelation estimate, truncates at the initial monotone positive
    sequence of paired sums, and returns ``S / (1 + 2 * sum(rho_t))`` with
    ``S`` the total draw count over the split sequences. The estimate is
    clipped to ``[1e-6, S]``: the cap means antithetic sequences report at
    most one effective draw per retained draw.

    Args:
        z: Array of shape ``(n_seq, n_len)`` with at least two sequences —
            the split chains, already rank-normalized.

    Returns:
        The effective sample size; ``nan`` when the sequences are constant.
    """
    m, n = z.shape
    s_total = m * n
    acov = _autocovariance(z)
    mean_acov = acov.mean(axis=0)
    within = mean_acov[0] * n / (n - 1.0)
    var_plus = mean_acov[0] + float(z.mean(axis=1).var(ddof=1))
    if var_plus <= 0.0 or not np.isfinite(var_plus):
        return float("nan")
    rho = 1.0 - (within - mean_acov) / var_plus
    rho[0] = 1.0
    total = 0.0
    prev = np.inf
    for k in range(n // 2):
        pair = rho[2 * k] + rho[2 * k + 1]
        if pair <= 0.0:
            break
        pair = min(pair, prev)
        total += pair
        prev = pair
    tau = -1.0 + 2.0 * total
    ess = s_total / tau if tau > 0.0 else float(s_total)
    return float(np.clip(ess, _ESS_MIN, s_total))


def _tail_ess(xs: np.ndarray) -> float:
    """Minimum ESS of the 5% and 95% pooled-quantile indicator functions.

    Args:
        xs: Split chains of one parameter on the original scale, shape
            ``(n_seq, n_len)``.

    Returns:
        ``min(ESS(x <= q05), ESS(x <= q95))`` where each indicator sequence
        is rank-normalized before the ESS computation.
    """
    values = []
    for q in (0.05, 0.95):
        cut = np.quantile(xs, q)
        indicator = (xs <= cut).astype(float)
        values.append(_ess(_rank_normalize(indicator)))
    return float(np.min(values))


def mcmc_report(draws: Any, *, rhat_max: float = 1.01, ess_floor: float = 400.0) -> McmcReport:
    """Check MCMC draws for convergence and effective sample size, per parameter.

    The diagnostics are the rank-normalized split R-hat and the bulk and tail effective sample
    sizes. Each chain is split in half, so one chain is enough, and every chain needs at least
    eight draws.

    Args:
        draws: The [`Draws`][diff_soup.inference.Draws] of `ds.inference.sample`, or an array
            shaped `(chain, draw, dim)`.
        rhat_max: The largest split R-hat that passes.
        ess_floor: The smallest bulk or tail effective sample size that passes.

    Returns:
        An `McmcReport` with the diagnostics per parameter, the sampler's statistics when
            `draws` carries them, and the failed checks in `flags`.
    """
    positions = np.asarray(getattr(draws, "positions", draws), dtype=float)
    names_in = tuple(getattr(draws, "names", ()) or ())
    info = dict(getattr(draws, "info", {}) or {})

    if positions.ndim != 3:
        raise ValueError(
            "mcmc_report expects draws of shape (chain, draw, dim); got an array "
            f"with {positions.ndim} dimension(s) and shape {positions.shape}."
        )
    n_chains, n_draws, dim = positions.shape
    if n_chains < 1:
        raise ValueError(
            "mcmc_report needs at least one chain; got zero. A single chain is "
            "enough because each chain is split in half before diagnosis."
        )
    if n_draws < 8:
        raise ValueError(
            "mcmc_report needs at least 8 draws per chain (4 per split half) to "
            f"estimate autocorrelations; got {n_draws}."
        )
    if dim < 1:
        raise ValueError("mcmc_report needs at least one parameter dimension; got zero.")
    if names_in and len(names_in) != dim:
        raise ValueError(
            f"draws carry {len(names_in)} parameter names but {dim} parameter "
            "dimension(s); supply one name per dimension or none."
        )
    names = names_in or tuple(f"theta[{i}]" for i in range(dim))

    pooled = positions.reshape(-1, dim)
    mean = pooled.mean(axis=0)
    sd = pooled.std(axis=0, ddof=1)
    rhat = np.empty(dim)
    ess_bulk = np.empty(dim)
    ess_tail = np.empty(dim)
    for j in range(dim):
        xs = _split_chains(positions[:, :, j])
        z = _rank_normalize(xs)
        rhat[j] = _rhat(z)
        ess_bulk[j] = _ess(z)
        ess_tail[j] = _tail_ess(xs)
    mcse_mean = sd / np.sqrt(ess_bulk)

    acceptance_rate = None
    acceptance_rate_per_chain = None
    acc = info.get("acceptance_rate")
    if acc is not None:
        acc = np.asarray(acc, dtype=float).reshape(n_chains, -1)
        acceptance_rate_per_chain = acc.mean(axis=1)
        acceptance_rate = float(acc.mean())

    n_divergent = None
    divergences_per_chain = None
    div = info.get("is_divergent")
    if div is not None:
        div = np.asarray(div).astype(bool).reshape(n_chains, -1)
        divergences_per_chain = div.sum(axis=1)
        n_divergent = int(div.sum())

    tree_depth_max = None
    tree_depth_mean = None
    depth = info.get("tree_depth", info.get("num_integration_steps"))
    if depth is not None:
        depth = np.asarray(depth)
        tree_depth_max = int(depth.max())
        tree_depth_mean = float(depth.mean())

    flags: list = []
    for j, name in enumerate(names):
        if np.isfinite(rhat[j]) and rhat[j] > rhat_max:
            flags.append(
                f"parameter {name!r}: split-Rhat {rhat[j]:.3f} exceeds {rhat_max:g} — "
                "the chains have not mixed to a common distribution"
            )
        if ess_bulk[j] < ess_floor:
            flags.append(
                f"parameter {name!r}: bulk ESS {ess_bulk[j]:.0f} is below {ess_floor:g} — "
                "too few effective draws for a stable posterior mean"
            )
        if ess_tail[j] < ess_floor:
            flags.append(
                f"parameter {name!r}: tail ESS {ess_tail[j]:.0f} is below {ess_floor:g} — "
                "too few effective draws in the tails for stable interval endpoints"
            )

    return McmcReport(
        names=names,
        n_chains=n_chains,
        n_draws=n_draws,
        mean=mean,
        sd=sd,
        mcse_mean=mcse_mean,
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        acceptance_rate=acceptance_rate,
        acceptance_rate_per_chain=acceptance_rate_per_chain,
        n_divergent=n_divergent,
        divergences_per_chain=divergences_per_chain,
        tree_depth_max=tree_depth_max,
        tree_depth_mean=tree_depth_mean,
        flags=tuple(flags),
        rhat_max=rhat_max,
        ess_floor=ess_floor,
    )
