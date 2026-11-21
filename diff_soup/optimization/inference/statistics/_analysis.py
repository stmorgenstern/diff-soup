"""Bind an experiment's prediction columns to its study's parameter schema."""

import jax.numpy as jnp
import numpy as np

from diff_soup.optimization.inference.statistics.estimation import EstimationProblem


def experiment_view(problem, experiment_id=None):
    """Return an experiment, its parameter selection, and global column indices."""
    if not isinstance(problem, EstimationProblem):
        raise TypeError("analysis requires an InferenceProblem.")
    study = problem.study
    if experiment_id is None:
        if len(study.experiments) != 1:
            raise ValueError("experiment_id is required for a problem with several experiments.")
        index = 0
    else:
        try:
            index = study.experiment_ids.index(experiment_id)
        except ValueError as exc:
            raise ValueError(
                f"unknown experiment_id {experiment_id!r}; expected {study.experiment_ids}."
            ) from exc
    experiment = study.experiments[index]
    parameters = study._parameter_groups(experiment.model)
    child = experiment.as_inference_problem(parameters=parameters)
    predictor = experiment.predictor(parameters=parameters)
    columns = np.full(len(predictor.theta_names), -1, dtype=np.int32)
    for name, local_columns in zip(child.param_group_names, child.param_group_cols):
        entry = next(
            entry for entry in problem.parameter_schema.entries if entry.handle.name == name
        )
        cluster = study.parameters.cluster_for(entry.handle)
        owner = None if cluster is None else study.clusters[cluster][index]
        occurrence = next(item for item in entry.occurrences if item.owner == owner)
        local_columns = np.asarray(local_columns)
        keep = local_columns < len(columns)
        columns[local_columns[keep]] = np.asarray(occurrence.columns)[keep]
    if np.any(columns < 0):
        raise ValueError("experiment prediction columns are absent from the parameter schema.")
    return experiment, parameters, columns


def _output_channel(output):
    """A prediction channel that reads an experiment's output; it is never scored."""
    from diff_soup.optimization.inference.fitting.observables import Observable
    from diff_soup.optimization.inference.statistics.experiment import Channel
    from diff_soup.optimization.inference.statistics.likelihoods import Gaussian

    return Channel(
        name=output.name,
        saveat=output.times,
        observable=Observable(output.name, output.observe, wrt_theta=True),
        likelihood=Gaussian(sigma=1.0),
        role="reference",
    )


def study_predictor(problem, experiment_id=None):
    """Return an experiment's view (as `experiment_view`) and its predictions as a function of the study's whole vector.

    Returns ``(experiment, parameters, columns, predict)``. ``predict`` maps the study's vector ``x`` to ``(predictions, failed)``, as the
    experiment's [Predictor][diff_soup.optimization.inference.statistics.experiment.Predictor]
    does at the experiment's columns of ``x``. An experiment driven by links
    ([Link][diff_soup.optimization.inference.Link]) is predicted as its lane is
    evaluated: each source is solved at its own columns of ``x`` for the outputs
    the links read, and their values fill the driven model's signals, so the
    derivative of a prediction carries the source's parameters through them.
    ``failed`` holds when the experiment's solve or a source's failed.
    """
    experiment, parameters, columns = experiment_view(problem, experiment_id)
    if not experiment.links:
        predictor = experiment.predictor(parameters=parameters)
        return experiment, parameters, columns, lambda x: predictor(x[columns])
    from diff_soup.optimization.inference.statistics.measurement import _signal_values

    sources = {}
    for link in experiment.links:
        sources.setdefault(link.source, []).append(link)
    produced = []
    for source_id, links in sources.items():
        source, source_parameters, source_columns = experiment_view(problem, source_id)
        names = tuple(dict.fromkeys(link.output for link in links))
        outputs = tuple(next(o for o in source.outputs if o.name == name) for name in names)
        predictor = source._delegate._predictor(
            source_parameters, channels=tuple(_output_channel(o) for o in outputs)
        )
        produced.append((predictor, source_columns, links))
    signals = tuple(
        (link.into, tuple(_signal_values(experiment._declarations[link.into]).shape))
        for _, _, links in produced
        for link in links
    )
    driven = experiment._delegate._predictor(parameters, signals=signals)

    def predict(x):
        values, failed = [], jnp.asarray(False)
        for predictor, source_columns, links in produced:
            outputs, source_failed = predictor(x[source_columns])
            failed = failed | source_failed
            values += [jnp.ravel(outputs[link.output]) for link in links]
        predictions, driven_failed = driven(jnp.concatenate([x[columns], *values]))
        return predictions, failed | driven_failed

    return experiment, parameters, columns, predict


def measurement_costs(problem, x):
    """The cost of every scored measurement of every experiment of the study at ``x``.

    Each experiment is predicted by `study_predictor`, links followed, and each
    measurement's rows are formed as its experiment's residual forms them: its
    weight, or the least-squares rows of its likelihood, at its kept frames.
    Returns ``(costs, names)``, ``costs`` half the squared rows of each
    measurement and ``names`` the pairs ``(experiment id, measurement name)``,
    in the order of the study's experiments and their measurements.

    Raises:
        RuntimeError: If the solve of an experiment, or of a source, failed at ``x``.
    """
    x = jnp.asarray(x)
    costs, names = [], []
    for experiment_id in problem.study.experiment_ids:
        experiment, _, _, predict = study_predictor(problem, experiment_id)
        predictions, failed = predict(x)
        if bool(failed):
            raise RuntimeError(
                f"the solve of experiment {experiment_id!r} or of its source failed at the point; "
                "no cost per measurement exists there."
            )
        for series in experiment._delegate._response_series():
            rows = _series_rows(series, predictions[series.name])
            costs.append(0.5 * float(jnp.sum(rows**2)))
            names.append((experiment_id, series.name))
    return tuple(costs), tuple(names)


def _series_rows(series, predicted):
    """The whitened rows of one scored stream at its kept frames, as a residual forms them."""
    keep = (
        None
        if series.frame_mask is None
        else np.flatnonzero(np.asarray(series.frame_mask).reshape(-1).astype(bool))
    )
    targets = series.values if keep is None else jnp.asarray(series.values)[keep]
    weight = series.weight
    likelihood = getattr(series, "likelihood", None)
    context = getattr(series, "likelihood_context", None)
    if (
        weight is not None
        and keep is not None
        and getattr(weight, "ndim", 0) >= 1
        and weight.shape[0] == series.values.shape[0]
    ):
        weight = weight[keep]
    if likelihood is not None:
        if keep is not None:
            likelihood, context = likelihood.for_frames(
                jnp.asarray(keep), full_prediction_shape=tuple(series.values.shape), context=context
            )
        rows = likelihood.least_squares_rows(targets, predicted, context=context)
    else:
        rows = predicted - targets
    if weight is not None:
        rows = weight * rows
    return jnp.reshape(rows, (-1,))
