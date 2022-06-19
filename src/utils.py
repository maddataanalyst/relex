import mlflow as mlflow
import sys
import logging


def prepare_mlflow(experiment_name: str) -> int:
    """
    Prepares an mlflow experiment id and name for later use.

    Parameters
    ----------
    experiment_name: str
        Name of the expriment to be conducted.

    Returns
    -------
    int
        Id of an experiment
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def prepare_default_log() -> logging.Logger:
    """
    Builds a default log for printing trianing outcomes.

    Returns
    -------
    logging.Logger
        A default logger object with stdout handle.

    """
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    return log
