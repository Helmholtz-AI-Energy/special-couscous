import glob
import logging
import pathlib
import shutil

import pandas as pd
import pytest
from mpi4py import MPI

from specialcouscous.train import (
    evaluate_parallel_from_checkpoint,
    train_parallel_on_balanced_synthetic_data,
)
from specialcouscous.utils import set_logger_config
from specialcouscous.utils.result_handling import construct_output_path

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.mark.mpi
@pytest.mark.parametrize(
    "random_state_model",
    [17, None],
)
def test_evaluate_from_checkpoint(
    random_state_model: int, mpi_tmp_path: pathlib.Path
) -> None:
    """
    Test parallel evaluation of random forest from pickled model checkpoints.

    First, run parallel training on balanced synthetic data and evaluate model. Then, generate data redundantly, load
    model checkpoints and evaluate loaded model on the regenerated balanced synthetic data.

    Parameters
    ----------
    random_state_model : int
        The random state used for the model.
    mpi_tmp_path : pathlib.Path
        The temporary folder used for storing results.
    """
    n_samples: int = 1000  # Number of samples in synthetic classification data
    n_features: int = 100  # Number of features in synthetic classification data
    n_classes: int = 10  # Number of classes in synthetic classification data
    n_clusters_per_class: int = 1  # Number of clusters per class
    frac_informative: float = (
        0.1  # Fraction of informative features in synthetic classification dataset
    )
    frac_redundant: float = (
        0.1  # Fraction of redundant features in synthetic classification dataset
    )
    random_state: int = 9  # Random state for synthetic data generation and splitting
    train_split: float = 0.75  # Fraction of original dataset used for training
    # Model-related arguments
    n_trees: int = 100  # Number of trees in global random forest classifier
    output_dir: pathlib.Path = mpi_tmp_path  # Directory to write results to
    experiment_id: str = (
        pathlib.Path(
            __file__
        ).stem  # Optional subdirectory name to collect related result in
    )
    save_model: bool = True
    shared_global_model: bool = True
    detailed_evaluation: bool = True  # Whether to perform a detailed evaluation on more than just the local test set.
    log_path: pathlib.Path = mpi_tmp_path  # Path to the log directory
    logging_level: int = logging.INFO  # Logging level
    log_file: pathlib.Path = pathlib.Path(
        f"{log_path}/{pathlib.Path(__file__).stem}.log"
    )

    # Set up separate logger for Special Couscous.
    set_logger_config(
        level=logging_level,  # Logging level
        log_file=log_file,  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        log.info(
            "*************************************************************\n"
            "* Multi-Node Random Forest Classification of Synthetic Data *\n"
            "*************************************************************\nTRAINING"
        )

    train_parallel_on_balanced_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        frac_informative=frac_informative,
        frac_redundant=frac_redundant,
        random_state=random_state,
        random_state_model=random_state_model,
        mpi_comm=comm,
        train_split=train_split,
        n_trees=n_trees,
        shared_global_model=shared_global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    comm.barrier()
    checkpoint_path, _ = construct_output_path(
        output_path=output_dir, experiment_id=experiment_id
    )
    if comm.rank == 0:
        log.info(f"EVALUATION: Checkpoint path is {checkpoint_path}.")

    evaluate_parallel_from_checkpoint(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        frac_informative=frac_informative,
        frac_redundant=frac_redundant,
        random_state=random_state,
        checkpoint_path=checkpoint_path,
        random_state_model=random_state_model,
        mpi_comm=comm,
        train_split=train_split,
        n_trees=n_trees,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
    )
    comm.barrier()
    # Get all result CSV files in output directory.
    result_csv_files = glob.glob(str(checkpoint_path) + "/*.csv")

    # Load result CSV files and convert into dataframe.
    result_csv_dfs = []
    for result_csv_file in result_csv_files:
        # result_df = pd.read_csv(result_csv_file)
        result_csv_dfs.append(pd.read_csv(result_csv_file).fillna(0))

    assert len(result_csv_files) == len(result_csv_dfs) == 2
    # Only compare the following columns of the result dataframes.
    columns_to_compare = [
        "accuracy_local_test",
        "accuracy_local_train",
        "comm_rank",
        "accuracy_global_test",
        "accuracy_global_train",
    ]

    for result_df in result_csv_dfs:
        pd.testing.assert_frame_equal(
            result_df[columns_to_compare], result_csv_dfs[0][columns_to_compare]
        )
    comm.barrier()
    # Remove all files generated during test in temporary directory.
    shutil.rmtree(str(mpi_tmp_path), ignore_errors=True)
