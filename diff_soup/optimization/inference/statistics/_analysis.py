"""Bind an experiment's prediction columns to its study's parameter schema."""

import numpy as np

from diff_soup.inference.statistics.estimation import EstimationProblem


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
