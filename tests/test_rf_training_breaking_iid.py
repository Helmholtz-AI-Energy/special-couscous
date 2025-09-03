import logging
import pathlib

import pytest
from mpi4py import MPI

from specialcouscous.train.train_parallel import train_parallel_on_synthetic_data
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.mark.mpi
@pytest.mark.parametrize(
    "shared_global_model, shared_test_set, globally_imbalanced, locally_imbalanced",
    [
        # (True, False, True, True),
        # (True, False, True, False),
        # (True, False, False, True),
        # (True, False, False, False),
        # (True, True, True, True),
        # (True, True, True, False),
        # (True, True, False, True),
        # (True, True, False, False),
        (False, True, True, True),
        (False, True, True, False),
        (False, True, False, True),
        (False, True, False, False),
    ],
)
@pytest.mark.parametrize("random_state_model", [17, None])
@pytest.mark.parametrize("flip_y", [0.0, 0.01])
@pytest.mark.parametrize("stratified_train_test", [True, False])
def test_breaking_iid(
    random_state_model: int,
    shared_global_model: bool,
    shared_test_set: bool,
    globally_imbalanced: bool,
    locally_imbalanced: bool,
    flip_y: float,
    stratified_train_test: bool,
    clean_mpi_tmp_path: pathlib.Path,
) -> None:
    """
    Test parallel training of random forest on imbalanced synthetic data.

    Parameters
    ----------
    random_state_model : int
        The random state used for the model.
    shared_global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    shared_test_set : bool
        Whether the test set is shared across all subforests (True) or private to each rank (False).
        If shared_global_model == False, the test set needs to be shared.
    globally_imbalanced : bool
        Whether the class distribution of the entire dataset is imbalanced.
    locally_imbalanced : bool
        Whether to use an imbalanced partition when assigning the dataset to ranks.
    flip_y: float
        The fraction of samples whose class is assigned randomly.
    stratified_train_test: bool
        Whether to stratify the train-test split with the class labels.
    clean_mpi_tmp_path : pathlib.Path
        The temporary folder used for storing results.
    """
    n_samples: int = 1000  # Number of samples in synthetic classification data
    n_features: int = 100  # Number of features in synthetic classification data
    n_classes: int = 3  # Number of classes in synthetic classification data
    n_clusters_per_class: int = 1  # Number of clusters per class
    frac_informative: float = (
        0.1  # Fraction of informative features in synthetic classification dataset
    )
    frac_redundant: float = (
        0.1  # Fraction of redundant features in synthetic classification dataset
    )
    random_state: int = 0  # Random seed used in synthetic dataset generation

    # Model-related arguments
    n_trees: int = 100  # Number of trees in global random forest classifier
    train_split: float = 0.75  # Fraction of data in the train set
    output_dir: pathlib.Path = clean_mpi_tmp_path  # Directory to write results to
    output_label: str = ""  # Optional label for the output files
    experiment_id: str = (
        "test_breaking_iid"  # Optional subdirectory name to collect related result in
    )
    mu_data: float = (
        2.0  # The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the Skellam distribution.
    )
    peak: int = (
        n_classes // 2
    )  # The position (class index) of the distribution's peak (i.e., the most frequent class).
    mu_partition: float = (
        10.0  # The μ = μ₁ = μ₂, μ ∊ [0, ∞] parameter of the Skellam distribution.
    )
    detailed_evaluation: bool = True
    save_model: bool = True
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
            "**********************************************************************\n"
            "* Distributed Random Forest Classification of Non-IID Synthetic Data *\n"
            "**********************************************************************"
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
        comm=MPI.COMM_WORLD,
        train_split=train_split,
        stratified_train_test=stratified_train_test,
        n_trees=n_trees,
        shared_global_model=shared_global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        output_label=output_label,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    comm.barrier()
