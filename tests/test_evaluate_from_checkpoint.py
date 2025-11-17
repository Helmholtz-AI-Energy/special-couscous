import glob
import logging
import pathlib

import pandas as pd
import pytest
from mpi4py import MPI

from specialcouscous.utils.evaluation_metrics import accuracy_score
from specialcouscous.train.train_parallel import (
    evaluate_parallel_from_checkpoint_balanced_synthetic_data,
    evaluate_parallel_from_checkpoint_synthetic_data,
    train_parallel_on_balanced_synthetic_data,
    train_parallel_on_synthetic_data,
)
from specialcouscous.utils import set_logger_config
from specialcouscous.utils.result_handling import construct_output_path

log = logging.getLogger("specialcouscous")  # Get logger instance.


def compare_confusion_matrices(confusion_csv_files: list[str]) -> float:
    """
    Load and compare confusion matrices from given paths.

    Parameters
    ----------
    confusion_csv_files : list[str]
        The paths to the confusion matrices to compare.

    Returns
    -------
    float
        The derived accuracy.
    """
    assert len(confusion_csv_files) == 2
    confusion_csv_dfs = []
    for confusion_csv_file in confusion_csv_files:
        confusion_csv_dfs.append(
            pd.read_csv(confusion_csv_file, header=None, index_col=False)
        )
    assert len(confusion_csv_dfs) == 2
    pd.testing.assert_frame_equal(confusion_csv_dfs[0], confusion_csv_dfs[1])
    return accuracy_score(confusion_csv_dfs[0].to_numpy())


@pytest.mark.mpi
@pytest.mark.parametrize(
    "random_state_model",
    [17, None],
)
@pytest.mark.parametrize(
    "flip_y",
    [0.0, 0.01],
)
@pytest.mark.parametrize(
    "stratified_train_test",
    [True, False],
)
@pytest.mark.parametrize(
    "globally_imbalanced, locally_imbalanced",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_evaluate_from_checkpoint_synthetic(
    globally_imbalanced: bool,
    locally_imbalanced: bool,
    random_state_model: int,
    flip_y: float,
    stratified_train_test: bool,
    clean_mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test parallel evaluation of random forest from pickled model checkpoints.

    First, run parallel training on synthetic data and evaluate model. Then, generate data redundantly, load model
    checkpoints and evaluate loaded model on the regenerated balanced synthetic data. Compare the resulting confusion
    matrices and results files.

    Parameters
    ----------
    globally_imbalanced : bool
        Whether the class distribution of the entire dataset is imbalanced.
    locally_imbalanced : bool
        Whether to use an imbalanced partition when assigning the dataset to ranks.
    random_state_model : int
        The random state used for the model.
    flip_y: float
        The fraction of samples whose class is assigned randomly.
    stratified_train_test: bool
        Whether to stratify the train-test split with the class labels.
    clean_mpi_tmp_path : pathlib.Path
        The temporary folder used for storing results.
    """
    # --- SYNTHETIC DATASET GENERATION SETTINGS ---
    n_samples: int = 1000  # Number of samples
    n_features: int = 100  # Number of features
    n_classes: int = 10  # Number of classes
    n_clusters_per_class: int = 1  # Number of clusters per class
    frac_informative: float = 0.1  # Fraction of informative features
    frac_redundant: float = 0.1  # Fraction of redundant features
    random_state: int = 9  # Random state for data generation and splitting
    mu_data: float = 10.0  # μ parameter of the Skellam-distributed global dataset
    peak: int = n_classes // 2  # Position (class index) of the distribution's peak
    mu_partition: float = 10.0  # μ parameter of the Skellam-distributed partitioning
    train_split: float = 0.75  # Fraction of original dataset used for training
    shared_test_set: bool = True  # Shared test set

    # --- MODEL SETTINGS ---
    n_trees: int = 100  # Number of trees in global random forest classifier
    output_dir: pathlib.Path = clean_mpi_tmp_path  # Directory to write results to
    experiment_id: str = pathlib.Path(__file__).stem  # Optional results subdirectory
    save_model: bool = True
    shared_global_model: bool = False
    log_path: pathlib.Path = clean_mpi_tmp_path  # Path to the log directory
    logging_level: int = logging.INFO  # Logging level
    log_file: pathlib.Path = pathlib.Path(
        f"{log_path}/{pathlib.Path(__file__).stem}.log"
    )
    detailed_evaluation = False
    # Configure logger.
    set_logger_config(
        level=logging_level,  # Logging level
        log_file=log_file,  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    comm = MPI.COMM_WORLD

    # --- FIRST: TRAINING TO GENERATE CHECKPOINTS + EVALUATION ---
    if comm.rank == 0:
        log.info(
            "**************************************************************\n"
            "* Distributed Random Forest Classification of Synthetic Data *\n"
            "**************************************************************\nTRAINING"
        )
    train_parallel_on_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        globally_balanced=not globally_imbalanced,
        locally_balanced=not locally_imbalanced,
        shared_test_set=shared_test_set,
        random_state=random_state,
        random_state_model=random_state_model,
        mu_partition=mu_partition,
        mu_data=mu_data,
        peak=peak,
        make_classification_kwargs={
            "n_clusters_per_class": n_clusters_per_class,
            "n_informative": int(frac_informative * n_features),
            "n_redundant": int(frac_redundant * n_features),
            "flip_y": flip_y,
        },
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

    # --- SECOND: RE-EVALUATE FROM LOADED CHECKPOINTS OF FIRST RUN ---
    checkpoint_path, base_filename = construct_output_path(
        output_path=output_dir, experiment_id=experiment_id
    )
    if comm.rank == 0:
        log.info(f"EVALUATION: Checkpoint path is {checkpoint_path}.")

    evaluate_parallel_from_checkpoint_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        globally_balanced=not globally_imbalanced,
        locally_balanced=not locally_imbalanced,
        mu_partition=mu_partition,
        mu_data=mu_data,
        peak=peak,
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
        output_dir=output_dir,
        experiment_id=experiment_id,
    )
    comm.barrier()

    # --- COMPARE CONFUSION MATRICES AND CALCULATE ACCURACIES ---
    # Compare all corresponding confusion matrix CSV files in output directory and calculate accuracies.
    local_test_accuracy = compare_confusion_matrices(
        glob.glob(
            str(checkpoint_path) + f"/*_confusion_matrix_test_rank_{comm.rank}.csv"
        )
    )  # Compare rank-local test confusion matrices and calculate corresponding accuracy.
    global_test_accuracy = compare_confusion_matrices(
        glob.glob(str(checkpoint_path) + "/*_confusion_matrix_test_global.csv")
    )  # Compare global test confusion matrices and calculate corresponding accuracy.

    # --- COMPARE RESULT CSV FILES ---
    # Get all result CSV files in output directory.
    result_csv_files = glob.glob(str(checkpoint_path) + "/*_results.csv")
    # Load result CSV files and convert to dataframes.
    result_csv_dfs = []
    for result_csv_file in result_csv_files:
        result_csv_dfs.append(pd.read_csv(result_csv_file).fillna(0))
    # There should be two files: one from the first run and another one from the second run started from the first run's
    # checkpoints.
    assert len(result_csv_files) == len(result_csv_dfs) == 2
    # Only compare the following columns of the result dataframes.
    columns_to_compare = [
        "accuracy_local_test",
        "comm_rank",
        "accuracy_global_test",
    ]

    # Assert that two dataframes' values in the respective columns are equal.
    pd.testing.assert_frame_equal(
        result_csv_dfs[0][columns_to_compare], result_csv_dfs[1][columns_to_compare]
    )

    # --- COMPARE ACCURACIES ---
    # Compare manually calculated accuracies from result CSV files to accuracies calculated from confusion matrices.
    df = result_csv_dfs[0]
    assert (
        df.loc[df["comm_rank"] == str(float(comm.rank)), "accuracy_local_test"].values[
            0
        ]
        == local_test_accuracy
    )
    assert (
        df.loc[df["comm_rank"] == "global", "accuracy_global_test"].values[0]
        == global_test_accuracy
    )
    comm.barrier()


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
def test_evaluate_from_checkpoint_balanced_synthetic(
    random_state_model: int,
    detailed_evaluation: bool,
    flip_y: float,
    stratified_train_test: bool,
    clean_mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test parallel evaluation of random forest from pickled model checkpoints.

    First, run parallel training on balanced synthetic data and evaluate model. Then, generate data redundantly, load
    model checkpoints and evaluate loaded model on the regenerated balanced synthetic data. Compare the resulting
    confusion matrices and results files.

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
    # --- SYNTHETIC DATASET GENERATION SETTINGS ---
    n_samples: int = 1000  # Number of samples
    n_features: int = 100  # Number of features
    n_classes: int = 10  # Number of classes
    n_clusters_per_class: int = 1  # Number of clusters per class
    frac_informative: float = 0.1  # Fraction of informative features
    frac_redundant: float = 0.1  # Fraction of redundant features
    random_state: int = 9  # Random state for data generation and splitting
    train_split: float = 0.75  # Fraction of original dataset used for training

    # --- MODEL SETTINGS ---
    n_trees: int = 100  # Number of trees in global random forest classifier
    output_dir: pathlib.Path = clean_mpi_tmp_path  # Directory to write results to
    experiment_id: str = pathlib.Path(__file__).stem  # Optional results subdirectory
    save_model: bool = True
    shared_global_model: bool = False
    log_path: pathlib.Path = clean_mpi_tmp_path  # Path to the log directory
    logging_level: int = logging.INFO  # Logging level
    log_file: pathlib.Path = pathlib.Path(
        f"{log_path}/{pathlib.Path(__file__).stem}.log"
    )

    # Configure logger.
    set_logger_config(
        level=logging_level,  # Logging level
        log_file=log_file,  # Logging path
        log_to_stdout=True,  # Print log on stdout.
        log_rank=False,  # Do not prepend MPI rank to logging messages.
        colors=True,  # Use colors.
    )

    comm = MPI.COMM_WORLD

    # --- FIRST: TRAINING TO GENERATE CHECKPOINTS + EVALUATION ---
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

    # --- SECOND: RE-EVALUATE FROM LOADED CHECKPOINTS OF FIRST RUN ---
    checkpoint_path, base_filename = construct_output_path(
        output_path=output_dir, experiment_id=experiment_id
    )
    if comm.rank == 0:
        log.info(f"EVALUATION: Checkpoint path is {checkpoint_path}.")
    evaluate_parallel_from_checkpoint_balanced_synthetic_data(
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

    # --- COMPARE CONFUSION MATRICES AND CALCULATE ACCURACIES ---
    # Compare all corresponding confusion matrix CSV files in output directory and calculate accuracies.
    if detailed_evaluation:  # Additionally consider train accuracies.
        local_train_accuracy = compare_confusion_matrices(
            glob.glob(
                str(checkpoint_path) + f"/*_confusion_matrix_train_rank_{comm.rank}.csv"
            )
        )  # Compare rank-local train confusion matrices and calculate corresponding accuracy.
        global_train_accuracy = compare_confusion_matrices(
            glob.glob(str(checkpoint_path) + "/*_confusion_matrix_train_global.csv")
        )  # Compare global train confusion matrices and calculate corresponding accuracy.

    local_test_accuracy = compare_confusion_matrices(
        glob.glob(
            str(checkpoint_path) + f"/*_confusion_matrix_test_rank_{comm.rank}.csv"
        )
    )  # Compare rank-local test confusion matrices and calculate corresponding accuracy.
    global_test_accuracy = compare_confusion_matrices(
        glob.glob(str(checkpoint_path) + "/*_confusion_matrix_test_global.csv")
    )  # Compare global test confusion matrices and calculate corresponding accuracy.

    # --- COMPARE RESULT CSV FILES ---
    # Get all result CSV files in output directory.
    result_csv_files = glob.glob(str(checkpoint_path) + "/*_results.csv")
    # Load result CSV files and convert to dataframes.
    result_csv_dfs = []
    for result_csv_file in result_csv_files:
        result_csv_dfs.append(pd.read_csv(result_csv_file).fillna(0))
    # There should be two files: one from the first run and another one from the second run started from the first run's
    # checkpoints.
    assert len(result_csv_files) == len(result_csv_dfs) == 2
    # Only compare the following columns of the result dataframes.
    if detailed_evaluation:  # Compare both test and train accuracies.
        columns_to_compare = [
            "accuracy_local_test",
            "accuracy_local_train",
            "comm_rank",
            "accuracy_global_test",
            "accuracy_global_train",
        ]
    else:  # Compare only test accuracies.
        columns_to_compare = [
            "accuracy_local_test",
            "comm_rank",
            "accuracy_global_test",
        ]

    # Assert that two dataframes' values in the respective columns are equal.
    pd.testing.assert_frame_equal(
        result_csv_dfs[0][columns_to_compare], result_csv_dfs[1][columns_to_compare]
    )

    # --- COMPARE ACCURACIES ---
    # Compare manually calculated accuracies from result CSV files to accuracies calculated from confusion matrices.
    df = result_csv_dfs[0]
    if detailed_evaluation:  # Additionally compare train accuracies.
        assert (
            df.loc[
                df["comm_rank"] == str(float(comm.rank)), "accuracy_local_train"
            ].values[0]
            == local_train_accuracy
        )
        assert (
            df.loc[df["comm_rank"] == "global", "accuracy_global_train"].values[0]
            == global_train_accuracy
        )

    assert (
        df.loc[df["comm_rank"] == str(float(comm.rank)), "accuracy_local_test"].values[
            0
        ]
        == local_test_accuracy
    )
    assert (
        df.loc[df["comm_rank"] == "global", "accuracy_global_test"].values[0]
        == global_test_accuracy
    )
    comm.barrier()
