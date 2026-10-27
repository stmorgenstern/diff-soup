"""Labeled local sequential design with fresh linearization and replay."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from diff_soup.inference.statistics.information import InformationResult
from diff_soup.inference.statistics.laplace import LaplacePosterior

__all__ = [
    "CandidateDesign",
    "ExperimentSpace",
    "NextDesign",
    "SequentialState",
    "SequentialUpdate",
    "replay_sequential",
    "sequential_step",
    "sequential_update",
]

_APPROXIMATION = "Laplace/local"
_CRITERION = "local Gaussian information gain"
_CHECKPOINT_VERSION = 1


def _ids(values: Any, *, field: str) -> frozenset:
    """Normalize and validate one set of provenance IDs."""
    try:
        result = frozenset(values)
    except TypeError as error:
        raise TypeError(f"{field} must be an iterable of hashable IDs.") from error
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field} must contain nonempty string IDs.")
    return result


@dataclasses.dataclass(frozen=True)
class CandidateDesign:
    """One candidate protocol and a callback for fresh local information."""

    design_id: str
    payload: Any
    information_at: Callable
    data_ids: frozenset
    conditioning_ids: frozenset

    def __post_init__(self) -> None:
        """Validate candidate labels and the fresh-information callback."""
        if not isinstance(self.design_id, str) or not self.design_id:
            raise ValueError("CandidateDesign.design_id must be a nonempty string.")
        if not callable(self.information_at):
            raise TypeError(
                "CandidateDesign.information_at must be callable so every "
                "sequential step can request fresh information at its mode."
            )
        object.__setattr__(self, "data_ids", _ids(self.data_ids, field="data_ids"))
        object.__setattr__(
            self,
            "conditioning_ids",
            _ids(self.conditioning_ids, field="conditioning_ids"),
        )
        if not self.data_ids:
            raise ValueError("CandidateDesign.data_ids must name response data.")
        overlap = self.data_ids & self.conditioning_ids
        if overlap:
            raise ValueError(
                "CandidateDesign: response data_ids and reusable conditioning_ids "
                f"must be disjoint; overlap={sorted(overlap)}."
            )


@dataclasses.dataclass(frozen=True)
class ExperimentSpace:
    """Finite ordered set of candidate designs."""

    candidates: tuple

    def __post_init__(self) -> None:
        """Freeze and validate the ordered candidate collection."""
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("ExperimentSpace needs at least one candidate design.")
        if any(not isinstance(candidate, CandidateDesign) for candidate in candidates):
            raise TypeError("ExperimentSpace.candidates must contain CandidateDesign objects.")
        design_ids = tuple(candidate.design_id for candidate in candidates)
        if len(set(design_ids)) != len(design_ids):
            raise ValueError("ExperimentSpace candidate design IDs must be unique.")
        object.__setattr__(self, "candidates", candidates)


@dataclasses.dataclass(frozen=True)
class NextDesign:
    """Selected design with the local approximation and provenance attached."""

    design_id: str
    payload: Any
    score: float
    linearized_at: tuple
    approximation: str
    criterion: str
    data_ids: frozenset
    conditioning_ids: frozenset


def _posterior_precision(posterior: LaplacePosterior) -> np.ndarray:
    """Reconstruct the small local precision from its Cholesky factor."""
    factor = np.asarray(posterior.precision_factor, dtype=float)
    return factor @ factor.T


def _posterior_fingerprint(posterior: LaplacePosterior) -> str:
    """Hash the posterior identity needed to resume a checkpoint safely."""
    precision = _posterior_precision(posterior)
    declaration = {
        "coordinate": posterior.coordinate,
        "curvature_kind": posterior.curvature_kind,
        "mode": np.asarray(posterior.mode, dtype=float).tolist(),
        "names": list(posterior.names),
        "precision": precision.tolist(),
    }
    canonical = json.dumps(declaration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert a candidate payload to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(
        f"SequentialState checkpoint payloads must be JSON-compatible; got {type(value).__name__}."
    )


def _selection_checkpoint(selection: NextDesign) -> dict:
    """Lower one selection history record to durable metadata."""
    return {
        "approximation": selection.approximation,
        "conditioning_ids": sorted(selection.conditioning_ids),
        "criterion": selection.criterion,
        "data_ids": sorted(selection.data_ids),
        "design_id": selection.design_id,
        "linearized_at": list(selection.linearized_at),
        "payload": _json_value(selection.payload),
        "score": float(selection.score),
    }


def _selection_from_checkpoint(record: Mapping) -> NextDesign:
    """Restore one selection history record from durable metadata."""
    return NextDesign(
        design_id=str(record["design_id"]),
        payload=record.get("payload"),
        score=float(record["score"]),
        linearized_at=tuple(float(value) for value in record["linearized_at"]),
        approximation=str(record["approximation"]),
        criterion=str(record["criterion"]),
        data_ids=_ids(record["data_ids"], field="checkpoint data_ids"),
        conditioning_ids=_ids(record["conditioning_ids"], field="checkpoint conditioning_ids"),
    )


@dataclasses.dataclass(frozen=True)
class SequentialState:
    """Posterior and exactly-once data ledger for sequential design."""

    posterior: LaplacePosterior
    assimilated_data_ids: frozenset
    conditioning_ids: frozenset
    history: tuple = ()
    iteration: int = 0
    approximation: str = _APPROXIMATION

    def __post_init__(self) -> None:
        """Validate the posterior, ledgers, and iteration/history invariant."""
        if not isinstance(self.posterior, LaplacePosterior):
            raise TypeError("SequentialState.posterior must be a LaplacePosterior.")
        object.__setattr__(
            self,
            "assimilated_data_ids",
            _ids(self.assimilated_data_ids, field="assimilated_data_ids"),
        )
        object.__setattr__(
            self,
            "conditioning_ids",
            _ids(self.conditioning_ids, field="conditioning_ids"),
        )
        history = tuple(self.history)
        if any(not isinstance(selection, NextDesign) for selection in history):
            raise TypeError("SequentialState.history must contain NextDesign records.")
        object.__setattr__(self, "history", history)
        if int(self.iteration) != self.iteration or self.iteration < 0:
            raise ValueError("SequentialState.iteration must be a nonnegative integer.")
        if self.iteration != len(history):
            raise ValueError("SequentialState.iteration must equal the number of history records.")
        if self.approximation != _APPROXIMATION:
            raise ValueError(f"SequentialState.approximation must be {_APPROXIMATION!r}.")

    def to_checkpoint(self) -> dict:
        """Return durable JSON metadata, excluding executable callbacks."""
        return {
            "approximation": self.approximation,
            "assimilated_data_ids": sorted(self.assimilated_data_ids),
            "conditioning_ids": sorted(self.conditioning_ids),
            "history": [_selection_checkpoint(selection) for selection in self.history],
            "iteration": self.iteration,
            "posterior_fingerprint": _posterior_fingerprint(self.posterior),
            "version": _CHECKPOINT_VERSION,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping,
        *,
        posterior: LaplacePosterior,
    ) -> SequentialState:
        """Restore a state after validating the supplied posterior exactly."""
        if not isinstance(checkpoint, Mapping):
            raise TypeError("SequentialState checkpoint must be a mapping.")
        if checkpoint.get("version") != _CHECKPOINT_VERSION:
            raise ValueError(
                "SequentialState checkpoint has an unsupported version; "
                f"expected {_CHECKPOINT_VERSION}."
            )
        expected = checkpoint.get("posterior_fingerprint")
        actual = _posterior_fingerprint(posterior)
        if expected != actual:
            raise ValueError(
                "SequentialState checkpoint posterior fingerprint does not match "
                "the supplied posterior mode/precision."
            )
        history = tuple(
            _selection_from_checkpoint(record) for record in checkpoint.get("history", ())
        )
        return cls(
            posterior=posterior,
            assimilated_data_ids=_ids(
                checkpoint.get("assimilated_data_ids", ()),
                field="checkpoint assimilated_data_ids",
            ),
            conditioning_ids=_ids(
                checkpoint.get("conditioning_ids", ()),
                field="checkpoint conditioning_ids",
            ),
            history=history,
            iteration=int(checkpoint.get("iteration", -1)),
            approximation=str(checkpoint.get("approximation", "")),
        )


@dataclasses.dataclass(frozen=True)
class SequentialUpdate:
    """Posterior refit resulting from observing one selected design."""

    selection: NextDesign
    posterior: LaplacePosterior
    observed_data_ids: frozenset

    def __post_init__(self) -> None:
        """Validate the selected design, new posterior, and observed IDs."""
        if not isinstance(self.selection, NextDesign):
            raise TypeError("SequentialUpdate.selection must be a NextDesign.")
        if not isinstance(self.posterior, LaplacePosterior):
            raise TypeError("SequentialUpdate.posterior must be a LaplacePosterior.")
        object.__setattr__(
            self,
            "observed_data_ids",
            _ids(self.observed_data_ids, field="observed_data_ids"),
        )


def _candidate_information(
    candidate: CandidateDesign,
    posterior: LaplacePosterior,
) -> np.ndarray:
    """Evaluate and validate fresh expected information for one candidate."""
    mode = np.asarray(posterior.mode, dtype=float)
    result = candidate.information_at(posterior.mode)
    if not isinstance(result, InformationResult):
        raise TypeError("CandidateDesign.information_at must return an InformationResult.")
    if result.coordinate != posterior.coordinate:
        raise ValueError(
            "sequential_step: candidate information coordinate does not match "
            "the posterior coordinate."
        )
    reported_at = result.provenance.get("linearized_at")
    if reported_at is None:
        raise ValueError(
            "sequential_step: candidate information lacks linearized_at "
            "provenance, so freshness cannot be verified."
        )
    reported = np.asarray(reported_at, dtype=float)
    if reported.shape != mode.shape or not np.allclose(reported, mode, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "sequential_step: stale candidate linearization; information must "
            "be recomputed at the current posterior mode."
        )
    matrix = result.expected
    if matrix is None:
        raise ValueError(
            "sequential_step: local Gaussian information gain requires expected "
            "candidate information."
        )
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (posterior.dim, posterior.dim):
        raise ValueError(
            "sequential_step: candidate information shape must match the "
            f"posterior dimension {posterior.dim}; got {matrix.shape}."
        )
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise ValueError(
            "sequential_step: expected candidate information must be positive semidefinite."
        )
    return matrix


def _local_information_gain(precision: np.ndarray, addition: np.ndarray) -> float:
    """Compute one-half of the local precision log-determinant gain."""
    sign_before, logdet_before = np.linalg.slogdet(precision)
    sign_after, logdet_after = np.linalg.slogdet(precision + addition)
    if sign_before <= 0 or sign_after <= 0:
        raise ValueError(
            "sequential_step: current and updated local precisions must be positive definite."
        )
    return 0.5 * float(logdet_after - logdet_before)


def sequential_step(
    state: SequentialState,
    experiment_space: ExperimentSpace,
) -> NextDesign:
    """Select the candidate with greatest fresh local Gaussian information gain."""
    if not isinstance(state, SequentialState):
        raise TypeError("sequential_step: state must be a SequentialState.")
    if not isinstance(experiment_space, ExperimentSpace):
        raise TypeError("sequential_step: experiment_space must be an ExperimentSpace.")

    precision = _posterior_precision(state.posterior)
    scored = []
    for candidate in experiment_space.candidates:
        overlap = candidate.data_ids & state.assimilated_data_ids
        if overlap:
            raise ValueError(
                "sequential_step: candidate response data were already "
                f"assimilated and cannot enter twice: {sorted(overlap)}."
            )
        information = _candidate_information(candidate, state.posterior)
        score = _local_information_gain(precision, information)
        scored.append((score, candidate))

    score, candidate = max(scored, key=lambda item: item[0])
    return NextDesign(
        design_id=candidate.design_id,
        payload=candidate.payload,
        score=score,
        linearized_at=tuple(float(value) for value in np.asarray(state.posterior.mode)),
        approximation=_APPROXIMATION,
        criterion=_CRITERION,
        data_ids=candidate.data_ids,
        conditioning_ids=candidate.conditioning_ids,
    )


def sequential_update(
    state: SequentialState,
    update: SequentialUpdate,
) -> SequentialState:
    """Assimilate exactly the selected response IDs and advance one iteration."""
    if not isinstance(state, SequentialState):
        raise TypeError("sequential_update: state must be a SequentialState.")
    if not isinstance(update, SequentialUpdate):
        raise TypeError("sequential_update: update must be a SequentialUpdate.")
    if update.observed_data_ids != update.selection.data_ids:
        raise ValueError(
            "sequential_update: observed data IDs must exactly equal the "
            "selected response data IDs."
        )
    overlap = update.observed_data_ids & state.assimilated_data_ids
    if overlap:
        raise ValueError(
            "sequential_update: response data were already assimilated and "
            f"cannot enter twice: {sorted(overlap)}."
        )
    selected_at = np.asarray(update.selection.linearized_at, dtype=float)
    current_mode = np.asarray(state.posterior.mode, dtype=float)
    if selected_at.shape != current_mode.shape or not np.allclose(
        selected_at, current_mode, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("sequential_update: selection is stale for the current posterior mode.")
    if update.posterior.coordinate != state.posterior.coordinate:
        raise ValueError(
            "sequential_update: updated posterior coordinate must match the sequential state."
        )
    if update.posterior.dim != state.posterior.dim:
        raise ValueError(
            "sequential_update: updated posterior dimension must match the sequential state."
        )

    return SequentialState(
        posterior=update.posterior,
        assimilated_data_ids=state.assimilated_data_ids | update.observed_data_ids,
        conditioning_ids=state.conditioning_ids,
        history=state.history + (update.selection,),
        iteration=state.iteration + 1,
        approximation=_APPROXIMATION,
    )


def replay_sequential(
    state: SequentialState,
    updates: Sequence[SequentialUpdate],
) -> SequentialState:
    """Replay deterministic labeled updates from an initial or restored state."""
    result = state
    for update in updates:
        result = sequential_update(result, update)
    return result
