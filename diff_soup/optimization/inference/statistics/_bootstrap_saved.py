"""What a bootstrap saves and resumes from: its declaration, its record at the fit, its shards."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.problem.parameters import ParameterEntry, parameter_decoder


def _plain(value: Any) -> Any:
    """``value`` as numbers, strings, lists and mappings that JSON holds; another object as its text."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _plain(item) for key, item in value._asdict().items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


# The settings of the solver every replicate's trust region takes, in a bootstrap's fingerprint.
_SOLVER_SETTINGS = (
    "hessian",
    "x_scale",
    "radius_init",
    "radius_max",
    "eta",
    "grad_tol",
    "step_tol",
    "cost_tol",
    "cost_atol",
    "stationarity_tol",
    "noise_trial_limit",
    "noise_relaxed_ratio",
    "max_consecutive_invalid_trials",
    "fixed_indices",
    "constraints",
    "gauges",
)


def _digest(value: Any) -> str:
    """Sixteen hexadecimal characters of a digest of ``value``'s numbers and structure.

    Arrays enter by their type, shape and bytes, other leaves by their text,
    and the tree by its structure, so the digest is the same in every
    process for equal settings.
    """
    hasher = hashlib.sha256()
    leaves, tree = jax.tree_util.tree_flatten(value)
    hasher.update(str(tree).encode())
    for leaf in leaves:
        if isinstance(leaf, (np.ndarray, np.generic, jax.Array)):
            array = np.ascontiguousarray(np.asarray(leaf))
            hasher.update(f"{array.dtype.str}{array.shape}".encode())
            hasher.update(array.tobytes())
        else:
            hasher.update(repr(leaf).encode())
    return hasher.hexdigest()[:16]


def _key_text(key: Any) -> str:
    """A digest of a JAX random key's bits, the same for a typed key and its raw bits."""
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError):
        data = key
    return _digest(np.asarray(data))


def _setting_text(value: Any) -> str:
    """A setting as text: a plain number, string or ``None`` as itself, anything else by its digest."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    return f"digest {_digest(value)}"


def _declaration(
    problem: Any,
    fit: Any,
    key: Any,
    *,
    scheme: str,
    replicates: int,
    steps: Any,
    max_attempts: Any,
    over: str,
    weight_priors: bool,
    solver: Any,
) -> dict:
    """What a bootstrap is declared with, item by item, as text: its fingerprint.

    The study's and its parameter schema's fingerprints, a digest of the
    fit's point (its bytes), a digest of the key, the weights, the
    replicates, the steps, the attempts, the level resampled, whether
    priors are weighed, and every setting of the solver the replicates'
    trust regions take. Two runs with equal fingerprints draw the same
    weights and start every replicate from the same state.
    """
    study = getattr(problem, "study", None)
    items = {
        "study": str(getattr(study, "fingerprint", None)),
        "parameter schema": str(getattr(problem.parameter_schema, "fingerprint", None)),
        "fit point": _digest(np.asarray(fit.params, dtype=np.float64)),
        "key": _key_text(key),
        "weights": str(scheme),
        "replicates": str(int(replicates)),
        "steps": str(steps),
        "max_attempts": str(max_attempts),
        "over": str(over),
        "weight_priors": str(bool(weight_priors)),
    }
    for name in _SOLVER_SETTINGS:
        items[f"solver {name}"] = _setting_text(getattr(solver, name, None))
    return items


def _differing(found: list) -> str:
    """``item A in the file, B here; ...`` for ``(item, A, B)`` triples."""
    return "; ".join(f"{item} {saved} in the file, {here} here" for item, saved, here in found)


class _AtFit(NamedTuple):
    """What a result keeps of the evaluation at the fit, for the replicates and members it holds."""

    point: np.ndarray
    estimates: np.ndarray
    jackknife: np.ndarray
    laplace: np.ndarray
    sandwich: np.ndarray
    influence: np.ndarray
    active_bounds: tuple
    failures: tuple
    fit_rmse: float
    score_balance: float
    inverse: Any  # inverse_columns and inverse_block of the Gauss-Newton matrix at the fit


def _at_fit(products: Any, jackknife: np.ndarray, lane_members: np.ndarray, count: int) -> _AtFit:
    """The evaluation at the fit's products, its leave-one-member-out rows those of ``jackknife``."""
    from diff_soup.optimization.inference.hierarchical._replicates import fit_training_rmse

    held = {int(member) for member in jackknife}
    return _AtFit(
        point=products.point.copy(),
        estimates=products.estimates,
        jackknife=products.jackknife[np.asarray(jackknife, dtype=np.int64)],
        laplace=products.laplace,
        sandwich=products.sandwich,
        influence=products.influence,
        active_bounds=tuple(products.active_bounds),
        failures=tuple(
            failure
            for failure in products.failures
            if failure.kind == "replicate" or int(failure.index) in held
        ),
        fit_rmse=fit_training_rmse(products, lane_members, count),
        score_balance=float(products.score_balance),
        inverse=products,
    )


def _inverse_arrays(solved: Mapping, size: int) -> dict:
    """The solved columns of the inverse Gauss-Newton matrix, ``(size,)`` each, as two arrays."""
    order = sorted(solved)
    return {
        "inverse.columns": np.asarray(order, dtype=np.int64),
        "inverse.values": (
            np.stack([np.asarray(solved[column], dtype=np.float64) for column in order])
            if order
            else np.zeros((0, size))
        ),
    }


def _at_fit_record(at: _AtFit) -> dict:
    """``{"arrays", "texts"}``: the products a checkpoint file keeps, to resume without the evaluation."""
    return {
        "arrays": {
            "point": np.asarray(at.point, dtype=np.float64),
            "estimates": np.asarray(at.estimates, dtype=np.float64),
            "jackknife": np.asarray(at.jackknife, dtype=np.float64),
            "laplace": np.asarray(at.laplace, dtype=np.float64),
            "sandwich": np.asarray(at.sandwich, dtype=np.float64),
            "influence": np.asarray(at.influence, dtype=np.float64),
            **_inverse_arrays(at.inverse.inverse_columns, np.asarray(at.point).size),
        },
        "texts": {
            "active_bounds": _plain(at.active_bounds),
            "failures": _plain(at.failures),
            "fit_rmse": float(at.fit_rmse),
            "score_balance": float(at.score_balance),
        },
    }


def _saved_at_fit(record: Mapping) -> _AtFit:
    """The products a checkpoint file kept, as the result keeps them."""
    from diff_soup.optimization.inference.hierarchical._replicates import OneStepFailure

    arrays, texts = record["arrays"], record["texts"]
    columns = arrays["inverse.columns"]
    return _AtFit(
        point=np.array(arrays["point"], dtype=np.float64, copy=True),
        estimates=arrays["estimates"],
        jackknife=arrays["jackknife"],
        laplace=arrays["laplace"],
        sandwich=arrays["sandwich"],
        influence=arrays["influence"],
        active_bounds=tuple(tuple(int(item) for item in side) for side in texts["active_bounds"]),
        failures=tuple(
            OneStepFailure(
                kind=str(item["kind"]),
                index=int(item["index"]),
                reason=str(item["reason"]),
                message=str(item["message"]),
            )
            for item in texts["failures"]
        ),
        fit_rmse=float(texts["fit_rmse"]),
        score_balance=float(texts["score_balance"]),
        inverse=_SavedInverse(
            {int(column): arrays["inverse.values"][row] for row, column in enumerate(columns)}
        ),
    )


def _resume(writer: Any, on_conflict: str) -> Any:
    """The newest population file's contents when this bootstrap wrote it, else ``None``.

    A file of another declaration (or one that cannot be read) is refused
    under ``on_conflict="error"``, naming every item that differs, and
    discarded with the log under ``"restart"``.
    """
    from diff_soup._errors import ConfigurationError

    path = writer.latest()
    if path is None:
        return None
    try:
        saved = writer.read(path)
    except ValueError as error:
        problem = str(error)
    else:
        found = writer.differences(saved["fingerprint"])
        if not found:
            return saved
        problem = (
            f"{path.name} was written by a bootstrap declared otherwise, its {', '.join(item for item, _, _ in found)} "
            f"differing ({_differing(found)})"
        )
    if on_conflict == "error":
        raise ConfigurationError(
            f"bootstrap checkpoint={str(writer.directory)!r} holds a run this bootstrap cannot "
            f"resume: {problem}. Resume with the settings the files were written with, pass "
            "on_conflict='restart' to discard them and start afresh, or give another directory."
        )
    writer.discard()
    return None


def _shard_of(shard: Any) -> Any:
    """``shard=`` as ``None`` or ``(i, n)``; refused unless ``0 <= i < n``, both integers."""
    from diff_soup._errors import ConfigurationError

    if shard is None:
        return None
    try:
        i, n = shard
    except (TypeError, ValueError):
        i = n = None
    whole = (
        isinstance(i, (int, np.integer))
        and not isinstance(i, bool)
        and _positive_integer(n)
        and 0 <= int(i) < int(n)
    )
    if not whole:
        raise ConfigurationError(
            f"bootstrap shard= is None or a pair (i, n) of integers with 0 <= i < n: shard i of "
            f"n runs the replicates whose number modulo n is i; got {shard!r}."
        )
    return int(i), int(n)


def _directory(checkpoint: Any) -> Any:
    """``checkpoint=`` as ``None`` or a path; refused unless a path, or a file is in the way."""
    from diff_soup._errors import ConfigurationError

    if checkpoint is None:
        return None
    try:
        path = Path(checkpoint)
    except TypeError:
        raise ConfigurationError(
            f"bootstrap checkpoint= is None or the path of a directory; got {checkpoint!r}."
        ) from None
    if path.exists() and not path.is_dir():
        raise ConfigurationError(
            f"bootstrap checkpoint= is a directory, and {path} is a file; give the path of a "
            "directory (it is made when missing)."
        )
    return path


class _SavedHandle(NamedTuple):
    """A parameter of a saved result, by its name and kind."""

    name: str
    kind: Any


class _SavedSchema:
    """The parameter schema of a loaded result: each parameter's name, kind, size and columns.

    A handle is found by its name (and, when two parameters share a name, its
    size). A value is decoded to the shape it had when the result was saved;
    a function is decoded by the handle given, which has to be the study's
    own function handle.
    """

    def __init__(self, fingerprint: str, entries: tuple):
        """The saved schema's fingerprint and its entries, in the schema's order."""
        self.fingerprint = fingerprint
        self.entries = entries

    @property
    def handles(self) -> tuple:
        """The saved parameters, by name and kind, in the schema's order."""
        return tuple(entry.handle for entry in self.entries)

    def entry(self, handle: Any) -> ParameterEntry:
        """The saved entry of the parameter ``handle`` names."""
        name = getattr(handle, "name", None)
        candidates = [entry for entry in self.entries if entry.handle.name == name]
        if len(candidates) > 1:
            size, _ = parameter_decoder(handle)
            candidates = [entry for entry in candidates if entry.size == size]
        if len(candidates) != 1:
            raise KeyError(
                f"parameter handle {name!r} is not one of the saved result's parameters "
                f"{[entry.handle.name for entry in self.entries]}."
            )
        saved = candidates[0]
        decode = saved.decode
        if saved.handle.kind == "function":
            size, decode = parameter_decoder(handle)
            if size != saved.size:
                raise KeyError(
                    f"the function handle {name!r} has {size} coefficients; the saved result's "
                    f"function {name!r} has {saved.size}. Give the study's own handle."
                )
        return ParameterEntry(
            handle=handle, occurrences=saved.occurrences, size=saved.size, decode=decode
        )


def _saved_decode(kind: Any, shape: Any) -> Any:
    """The decoder of a saved parameter: a value reshaped to its saved shape."""
    if kind == "function":

        def decode(values: Any) -> Any:
            raise TypeError(
                "a saved function is decoded through the study's own function handle: ask the "
                "loaded result with that handle."
            )

        return decode
    shape = tuple(int(size) for size in shape)
    return lambda values: jnp.reshape(jnp.asarray(values), shape)


class _SavedInverse:
    """The columns of the inverse Gauss-Newton matrix a saved result holds."""

    def __init__(self, columns: dict):
        """The saved columns, ``(n,)`` each, by coordinate."""
        self._columns = columns

    @property
    def inverse_columns(self) -> Any:
        """The saved columns, by coordinate."""
        return MappingProxyType(dict(self._columns))

    def inverse_block(self, columns: Any) -> np.ndarray:
        """The inverse on ``columns`` x ``columns``, from the saved columns."""
        columns = [int(column) for column in np.ravel(np.asarray(columns, dtype=np.int64))]
        missing = [column for column in columns if column not in self._columns]
        if missing:
            raise ValueError(
                f"this result holds the columns {sorted(self._columns)} of the inverse "
                f"Gauss-Newton matrix, not {missing[:8]}: a result read by load, or made by a "
                "run resumed from a checkpoint, holds the columns solved before it was saved "
                "(the shared coordinates', and those boot.laplace(handle) asked for). Ask "
                "boot.laplace(handle) before saving the result, or run the bootstrap without "
                "resuming."
            )
        if not columns:
            return np.zeros((0, 0))
        return np.stack([self._columns[column][columns] for column in columns], axis=1)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 1
