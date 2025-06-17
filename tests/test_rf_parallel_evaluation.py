import logging
import pathlib

import pytest
from mpi4py import MPI

from specialcouscous.rf_parallel import DistributedRandomForest
from specialcouscous.synthetic_classification_data import (
    SyntheticDataset,
    make_classification_dataset,
)
from specialcouscous.utils import set_logger_config, timing

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.fixture(scope="function", autouse=True)
def setup_logger(clean_mpi_tmp_path: pathlib.Path) -> None:
    """
    Set up the logger.

    Parameters
    ----------
    clean_mpi_tmp_path : pathlib.Path
        The temporary path to log to.
    """
    log_file = clean_mpi_tmp_path / f"{pathlib.Path(__file__).stem}.log"
    set_logger_config(log_file=log_file)


@pytest.fixture(scope="session")
def synthetic_dataset(
    n_samples: int = 10000, n_features: int = 100, n_classes: int = 10
) -> tuple[SyntheticDataset, SyntheticDataset]:
    """
    Create the dataset once per test session.

    Parameters
    ----------
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_classes : int
        The number of classes

    Returns
    -------
    tuple[SyntheticDataset, SyntheticDataset]
        The train and test set.
    """
    make_classification_kwargs = {
        "n_clusters_per_class": 1,
        "n_informative": int(0.1 * n_features),
        "n_redundant": int(0.1 * n_features),
        "flip_y": 0.0,
    }
    (train_samples, test_samples, train_targets, test_targets) = (
        make_classification_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            make_classification_kwargs=make_classification_kwargs,
        )
    )
    train_data = SyntheticDataset(x=train_samples, y=train_targets)
    test_data = SyntheticDataset(x=test_samples, y=test_targets)
    return train_data, test_data


@pytest.mark.mpi
@pytest.mark.parametrize("shared_global_model", [True, False])
def test_parallel_synthetic(
    shared_global_model: bool,
    synthetic_dataset: tuple[SyntheticDataset, SyntheticDataset],
    n_trees: int = 100,
    model_seed: int = 0,
) -> None:
    """
    Test the evaluation of parallel random forests.

    Compare the accuracy and running time of the .score() and .evaluate() methods.

    Parameters
    ----------
    shared_global_model : bool
        Whether to use a shared global model or distributed voting for inference.
    synthetic_dataset : tuple[SyntheticDataset, SyntheticDataset]
        The train and test dataset.
    n_trees : int
        The number of trees in the model to test.
    model_seed : int
        The random seed used for model creation.
    """
    comm = MPI.COMM_WORLD
    train_data, test_data = synthetic_dataset

    if comm.rank == 0:
        log.info("Create and train random forest")
    distributed_random_forest = DistributedRandomForest(
        n_trees_global=n_trees,
        comm=comm,
        random_state=model_seed,
        shared_global_model=shared_global_model,
    )
    distributed_random_forest.train(train_data.x, train_data.y)
    if shared_global_model:
        distributed_random_forest.build_shared_global_model()

    if comm.rank == 0:
        log.info("Evaluate on test set")

    # compute accuracy with score method
    with timing.MPITimer(comm, name="score on test set") as t_score:
        score_accuracy = distributed_random_forest.score(
            test_data.x, test_data.y, test_data.n_classes
        )
    # do full evaluation with evaluate method
    with timing.MPITimer(comm, name="evaluate on test set") as t_evaluate:
        distributed_random_forest.evaluate(
            test_data.x, test_data.y, test_data.n_classes, shared_global_model
        )

    if comm.rank == 0:
        log.info(
            f"Test accuracy: score={score_accuracy}, evaluate={distributed_random_forest.acc_global}"
        )
        log.info(
            f"Inference time: score={t_score.elapsed_time_average}, evaluate={t_evaluate.elapsed_time_average}"
        )

    assert score_accuracy == distributed_random_forest.acc_global

    if comm.rank == 0:
        log.info("Evaluate on train set")

    # compute accuracy with score method
    with timing.MPITimer(comm, name="score on train set"):
        score_accuracy = distributed_random_forest.score(
            train_data.x, train_data.y, train_data.n_classes
        )
    # do full evaluation with evaluate method
    with timing.MPITimer(comm, name="evaluate on train set"):
        distributed_random_forest.evaluate(
            train_data.x, train_data.y, train_data.n_classes, shared_global_model
        )

    if comm.rank == 0:
        log.info(
            f"Train accuracy: score={score_accuracy}, evaluate={distributed_random_forest.acc_global}"
        )
        log.info(
            f"Inference time: score={t_score.elapsed_time_average}, evaluate={t_evaluate.elapsed_time_average}"
        )

    assert score_accuracy == distributed_random_forest.acc_global
