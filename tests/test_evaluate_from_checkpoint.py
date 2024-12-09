import glob
import logging
import pathlib

import pandas as pd
import pytest
from mpi4py import MPI

from specialcouscous.train.train_parallel import (
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
@pytest.mark.parametrize(
    "detailed_evaluation",
    [True, False],
)
@pytest.mark.parametrize(
    "flip_y",
    [0.0, 0.01],
)
@pytest.mark.parametrize(
    "stratified_train_test",
    [True, False],
)
def test_evaluate_from_checkpoint(
    random_state_model: int,
    detailed_evaluation: bool,
    flip_y: float,
    stratified_train_test: bool,
    clean_mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test parallel evaluation of random forest from pickled model checkpoints.

    First, run parallel training on balanced synthetic data and evaluate model. Then, generate data redundantly, load
    model checkpoints and evaluate loaded model on the regenerated balanced synthetic data.

    Parameters
    ----------
    random_state_model : int
        The random state used for the model.
    detailed_evaluation : bool
        Whether to additionally evaluate the model on the training dataset.
    flip_y: float
        The fraction of samples whose class is assigned randomly.
    stratified_train_test: bool
        Whether to stratify the train-test split with the class labels.
    clean_mpi_tmp_path : pathlib.Path
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
    output_dir: pathlib.Path = clean_mpi_tmp_path  # Directory to write results to
    experiment_id: str = (
        pathlib.Path(
            __file__
        ).stem  # Optional subdirectory name to collect related result in
    )
    save_model: bool = True
    shared_global_model: bool = False
    log_path: pathlib.Path = clean_mpi_tmp_path  # Path to the log directory
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
            "**************************************************************\n"
            "* Distributed Random Forest Classification of Synthetic Data *\n"
            "**************************************************************\nTRAINING"
        )

    train_parallel_on_balanced_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        make_classification_kwargs={
            "n_clusters_per_class": n_clusters_per_class,
            "n_informative": int(frac_informative * n_features),
            "n_redundant": int(frac_redundant * n_features),
            "flip_y": flip_y,
        },
        random_state=random_state,
        random_state_model=random_state_model,
        mpi_comm=comm,
        train_split=train_split,
        stratified_train_test=stratified_train_test,
        n_trees=n_trees,
        shared_global_model=shared_global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    comm.barrier()
    checkpoint_path, base_filename = construct_output_path(
        output_path=output_dir, experiment_id=experiment_id
    )
    if comm.rank == 0:
        log.info(f"EVALUATION: Checkpoint path is {checkpoint_path}.")
    evaluate_parallel_from_checkpoint(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        make_classification_kwargs={
            "n_clusters_per_class": n_clusters_per_class,
            "n_informative": int(frac_informative * n_features),
            "n_redundant": int(frac_redundant * n_features),
            "flip_y": flip_y,
        },
        random_state=random_state,
        checkpoint_path=checkpoint_path,
        random_state_model=random_state_model,
        mpi_comm=comm,
        train_split=train_split,
        stratified_train_test=stratified_train_test,
        n_trees=n_trees,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        experiment_id=experiment_id,
    )
    comm.barrier()
    # Get all result CSV files in output directory.
    result_csv_files = glob.glob(str(checkpoint_path) + "/*_results.csv")

    # Load result CSV files and convert into dataframe.
    result_csv_dfs = []
    for result_csv_file in result_csv_files:
        # result_df = pd.read_csv(result_csv_file)
        result_csv_dfs.append(pd.read_csv(result_csv_file).fillna(0))

    assert len(result_csv_files) == len(result_csv_dfs) == 2
    # Only compare the following columns of the result dataframes.
    if detailed_evaluation:
        columns_to_compare = [
            "accuracy_local_test",
            "accuracy_local_train",
            "comm_rank",
            "accuracy_global_test",
            "accuracy_global_train",
        ]
    else:
        columns_to_compare = [
            "accuracy_local_test",
            "comm_rank",
            "accuracy_global_test",
        ]

    for result_df in result_csv_dfs:
        pd.testing.assert_frame_equal(
            result_df[columns_to_compare], result_csv_dfs[0][columns_to_compare]
        )
    comm.barrier()
