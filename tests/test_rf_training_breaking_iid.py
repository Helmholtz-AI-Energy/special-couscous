import logging
import pathlib
import shutil

import pytest
from mpi4py import MPI

from specialcouscous.train import train_parallel_on_synthetic_data
from specialcouscous.utils import set_logger_config

log = logging.getLogger("specialcouscous")  # Get logger instance.


@pytest.mark.mpi
@pytest.mark.parametrize(
    "global_model, private_test_set, globally_imbalanced, locally_imbalanced",
    [
        (True, True, True, True),
        (True, True, True, False),
        (True, True, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, False, True, False),
        (True, False, False, True),
        (True, False, False, False),
        (False, False, True, True),
        (False, False, True, False),
        (False, False, False, True),
        (False, False, False, False),
    ],
)
def test_breaking_iid(
    global_model: bool,
    private_test_set: bool,
    globally_imbalanced: bool,
    locally_imbalanced: bool,
) -> None:
    """
    Test parallel training of random forest on imbalanced synthetic data.

    Parameters
    ----------
    global_model : bool
        Whether the local models are all-gathered to one global model shared by all ranks after training.
    private_test_set : bool
        Whether the test set is private (not shared across subforests). If global_model == False, the test set needs to
        be shared.
    globally_imbalanced : bool
        Whether the class distribution of the entire dataset is imbalanced.
    locally_imbalanced : bool
        Whether to use an imbalanced partition when assigning the dataset to ranks.
    """
    n_samples: int = 1000  # Number of samples in synthetic classification data
    n_features: int = 100  # Number of features in synthetic classification data
    n_classes: int = 3  # Number of classes in synthetic classification data
    random_state_data: int = 0  # Random seed used in synthetic dataset generation

    # Model-related arguments
    n_trees: int = 100  # Number of trees in global random forest classifier
    random_state_model: int = (
        0  # Random seed used to initialize random forest classifier
    )
    train_split: float = 0.75  # Fraction of data in the train set
    output_dir: pathlib.Path = pathlib.Path(
        "./results"
    )  # Directory to write results to
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
    log_path: pathlib.Path = pathlib.Path("./")  # Path to the log directory
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
            "*********************************************************************\n"
            "* Multi-Node Random Forest Classification of Non-IID Synthetic Data *\n"
            "*********************************************************************"
        )
    train_parallel_on_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        globally_balanced=not globally_imbalanced,
        locally_balanced=not locally_imbalanced,
        shared_test_set=not private_test_set,
        random_state_data=random_state_data,
        random_state_model=random_state_model,
        mu_partition=mu_partition,
        mu_data=mu_data,
        peak=peak,
        comm=MPI.COMM_WORLD,
        train_split=train_split,
        n_trees=n_trees,
        global_model=global_model,
        detailed_evaluation=detailed_evaluation,
        output_dir=output_dir,
        output_label=output_label,
        experiment_id=experiment_id,
        save_model=save_model,
    )
    if comm.rank == 0:
        log_file.unlink()
        shutil.rmtree(output_dir)
