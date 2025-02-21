import argparse
import logging
import pathlib

import pytest
from _pytest.fixtures import SubRequest
from mpi4py import MPI

from specialcouscous.scaling_dataset import generate_and_save_dataset
from specialcouscous.train.train_parallel import (
    train_parallel_on_growing_balanced_synthetic_data,
)
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.fixture(params=[0.0, 0.01])
def prepare_dataset_and_args(
    tmp_path: pathlib.Path, request: SubRequest
) -> argparse.Namespace:
    """
    Initialize a default namespace and generate a corresponding dataset, return the namespace.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path to write the data to (set as data_root_path).
    request : SubRequest
        Pytest SubRequest used to pass the parameters (flip_y) to the fixture.

    Returns
    -------
    argparse.Namespace
        The namespace filled with the values used to generate the dataset.
    """
    comm = MPI.COMM_WORLD

    # initialize dataset parameters
    args = argparse.Namespace()
    args.n_samples = 1000
    args.n_features = 100
    args.n_classes = 10
    args.n_train_splits = comm.size
    args.random_state = 9
    args.train_split = 0.75
    args.stratified_train_test = True
    args.n_clusters_per_class = 1
    args.frac_informative = 0.1
    args.frac_redundant = 0.1
    args.flip_y = request.param
    args.data_root_path = tmp_path
    args.override_data = False

    # pre-generate the dataset
    generate_and_save_dataset(args)

    return args


@pytest.mark.mpi
@pytest.mark.parametrize(
    "random_state_model",
    [17, None],
)
@pytest.mark.parametrize(
    "shared_global_model",
    [True, False],
)
def test_parallel_synthetic_scale_data_and_model(
    random_state_model: int,
    shared_global_model: bool,
    clean_mpi_tmp_path: pathlib.Path,
    tmp_path: pathlib.Path,
    prepare_dataset_and_args: argparse.Namespace,
) -> None:
    """
    Test parallel training of random forest on growing synthetic data.

    Parameters
    ----------
    random_state_model: int
        The random state used for the model.
    flip_y: float
        The fraction of samples whose class is assigned randomly.
    shared_global_model: bool
        Whether to build a shared global model.
    clean_mpi_tmp_path : pathlib.Path
        The temporary folder used for storing results.
    """
    # Set up separate logger for Special Couscous.
    set_logger_config(level=logging.INFO)
    comm = MPI.COMM_WORLD

    # Model-related arguments
    n_trees: int = 100  # Number of trees in global random forest classifier
    output_dir: pathlib.Path = clean_mpi_tmp_path  # Directory to write results to
    experiment_id: str = "test_parallel_rf_scale_model_and_data"  # Optional subdirectory name to collect related result in
    save_model: bool = True
    detailed_evaluation: bool = True  # Whether to perform a detailed evaluation on more than just the local test set.

    if comm.rank == 0:
        log.info(
            "**************************************************************\n"
            "* Distributed Random Forest Classification of Synthetic Data *\n"
            "**************************************************************"
        )

    train_parallel_on_growing_balanced_synthetic_data(
        cli_args=prepare_dataset_and_args,
        random_state=prepare_dataset_and_args.random_state,
        random_state_model=random_state_model,
        mpi_comm=comm,
        n_trees=n_trees,
        shared_global_model=shared_global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    comm.barrier()
